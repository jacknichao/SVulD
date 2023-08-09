import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None, contrast_ids=None):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
                vec = (outputs * input_ids.ne(1).unsqueeze(-1)).sum(1) / input_ids.ne(1).sum(-1).unsqueeze(-1)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                
                loss = F.cross_entropy(logits, labels)

                if self.args.r_drop and self.args.do_train:
                    
                    # keep dropout twice
                    outputs2 = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
                    logits2 = self.classifier(outputs2)

                    # cross entropy loss for classifier
                    ce_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits2, labels))
                    
                    kl_loss = compute_kl_loss(logits, logits2)
                    
                    # carefully choose hyper-parameters
                    loss = ce_loss + 0.1 * kl_loss

                if self.args.simcse and self.args.do_train:
                    
                    # keep dropout twice
                    outputs2 = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
                    logits2 = self.classifier(outputs2)
                    vec2 = (outputs2 * input_ids.ne(1).unsqueeze(-1)).sum(1) / input_ids.ne(1).sum(-1).unsqueeze(-1)
                    
                    kl_loss = compute_kl_loss(logits, logits2)
                    simcse_loss = simcse_unsup_loss(torch.cat((vec, vec2), dim=0), device=self.args.device)
                    loss = loss + 0.1 * kl_loss + 0.2 * simcse_loss
                
                if self.args.simct and self.args.do_train:
                    
                    # dropout for contrast
                    outputs2 = self.encoder.roberta(contrast_ids, attention_mask=contrast_ids.ne(1), output_attentions=output_attentions)[0]
                    logits2 = self.classifier(outputs2)
                    vec2 = (outputs2 * contrast_ids.ne(1).unsqueeze(-1)).sum(1) / contrast_ids.ne(1).sum(-1).unsqueeze(-1)
                    
                    kl_loss = compute_kl_loss(logits, logits2)
                    simct_loss = simct_unsup_loss(vec, vec2, labels)
                    
                    # carefully choose hyper-parameters
                    loss = loss + 0.1 * kl_loss + 0.2 * simct_loss
            
                return loss, prob, vec
            else:
                return prob
            
            
def simct_unsup_loss(vec, contrast, label):
    
    label = (1 - label) * 2 - 1
    loss = F.cosine_embedding_loss(vec, contrast, label)
    return loss


def simcse_unsup_loss(vec, device, temp=0.05):
    
    # label: [0, batch_size], [1, batch_size+1],....,[batch_size-1, 2*batch_size-1]
    label = torch.arange(vec.shape[0], device=device)
    label = torch.roll(label, shifts=int(len(label)/2))

    # [batch_size*2, 1, 768] * [1, batch_size*2, 768] = [batch_size*2, batch_size*2]
    sim = F.cosine_similarity(vec.unsqueeze(1), vec.unsqueeze(0), dim=-1)

    sim = sim - torch.eye(vec.shape[0], device=device) * 1e12
    sim = sim / temp

    # cross entropy loss for similiarity
    loss = F.cross_entropy(sim, label)
    return torch.mean(loss)


def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss