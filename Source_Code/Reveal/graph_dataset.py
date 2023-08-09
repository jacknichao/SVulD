import copy

import sys

import numpy as np
import torch
import json
from imblearn.over_sampling import SMOTE
import pickle


class DataEntry:
    def __init__(self, dataset, feature_repr, label, _id=None, meta_data=None):
        self.dataset = dataset
        assert isinstance(self.dataset, DataSet)
        self.features = copy.deepcopy(feature_repr)
        self.label = label
        self.meta_data = meta_data
        self._id = _id
        pass

    def __repr__(self):
        return str(self.features) + '\t' + str(self.label)

    def __hash__(self):
        return str(self.features).__hash__

    def is_positive(self):
        return self.label == 1


class DataSet:
    def __init__(self, batch_size, hdim):
        self.train_entries = []
        self.valid_entries = []
        self.test_entries = []

        self.train_batch_indices = []
        self.valid_batch_indices = []
        self.test_batch_indices = []

        self.batch_size = batch_size
        self.hdim = hdim
        self.positive_indices_in_train = []
        self.negative_indices_in_train = []

    def initialize_dataset(self, balance=True, output_buffer=sys.stderr):
        if isinstance(balance, bool) and balance:
            entries = []
            train_features = []
            train_targets = []
            for entry in self.train_entries:
                train_features.append(entry.features)
                train_targets.append(entry.label)
            train_features = np.array(train_features)
            train_targets = np.array(train_targets)
            print(f"DataSet train len(features) before smote ={len(train_features)}")
            smote = SMOTE(random_state=1000)
            features, targets = smote.fit_resample(train_features, train_targets)
            print(f"DataSet train len(features) after smote={len(features)}") 
            for feature, target in zip(features, targets):
                entries.append(DataEntry(self, feature.tolist(), target.item()))
            self.train_entries = entries
        elif isinstance(balance, list) and len(balance) == 2:
            entries = []
            for entry in self.train_entries:
                if entry.is_positive():
                    for _ in range(balance[0]):
                        entries.append(
                            DataEntry(self, entry.features, entry.label, entry.meta_data)
                        )
                else:
                    if np.random.uniform() <= balance[1]:
                        entries.append(
                            DataEntry(self, entry.features, entry.label, entry.meta_data)
                        )
            self.train_entries = entries
            pass
        for tidx, entry in enumerate(self.train_entries):
            if entry.label == 1:
                self.positive_indices_in_train.append(tidx)
            else:
                self.negative_indices_in_train.append(tidx)

        self.initialize_train_batches()
        if output_buffer is not None:
            print('Number of Train Entries %d #Batches %d' % \
                  (len(self.train_entries), len(self.train_batch_indices)), file=output_buffer)
        self.initialize_valid_batches()
        if output_buffer is not None:
            print('Number of Valid Entries %d #Batches %d' % \
                  (len(self.valid_entries), len(self.valid_batch_indices)), file=output_buffer)
        self.initialize_test_batches()
        if output_buffer is not None:
            print(sum([i.label for i in self.test_entries]))
            print('Number of Test Entries %d #Batches %d' % \
                  (len(self.test_entries), len(self.test_batch_indices)), file=output_buffer)

    def add_data_entry(self, feature, label, _id=None, part='train'):
        assert part in ['train', 'valid', 'test']
        entry = DataEntry(self, feature, label, _id)
        if part == 'train':
            self.train_entries.append(entry)
        elif part == 'valid':
            self.valid_entries.append(entry)
        else:
            self.test_entries.append(entry)

    def initialize_train_batches(self, shuffle=True):
        self.train_batch_indices = self.create_batches(self.batch_size, self.train_entries, shuffle)
        return len(self.train_batch_indices)
        pass

    def clear_test_set(self):
        self.test_entries = []

    def initialize_valid_batches(self, batch_size=-1, shuffle=False):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batch_indices = self.create_batches(batch_size, self.valid_entries, shuffle)
        return len(self.valid_batch_indices)
        pass

    def initialize_test_batches(self, batch_size=-1, shuffle=False):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batch_indices = self.create_batches(batch_size, self.test_entries, shuffle)
        return len(self.test_batch_indices)
        pass

    def get_next_train_batch(self):
        if len(self.train_batch_indices) > 0:
            indices = self.train_batch_indices.pop()
            features, targets,_ = self.prepare_data(self.train_entries, indices)
            same_class_features = self.find_same_class_data(ignore_indices=indices)
            different_class_features = self.find_different_class_data(ignore_indices=indices)
            return features, targets, same_class_features, different_class_features
        raise ValueError('Initialize Train Batch First by calling dataset.initialize_train_batches()')
        pass

    def get_next_valid_batch(self):
        if len(self.valid_batch_indices) > 0:
            indices = self.valid_batch_indices.pop()
            return self.prepare_data(self.valid_entries, indices)
        raise ValueError('Initialize Valid Batch First by calling dataset.initialize_valid_batches()')
        pass

    def get_next_test_batch(self):
        if len(self.test_batch_indices) > 0:
            indices = self.test_batch_indices.pop()
            return self.prepare_data(self.test_entries, indices)
        raise ValueError('Initialize Test Batch First by calling dataset.initialize_test_batches()')
        pass

    def create_batches(self, batch_size, entries, shuffle=False):
        _batches = []
        if batch_size == -1:
            batch_size = self.batch_size
        total = len(entries)
        indices = np.arange(0, total - 1, 1)
        if shuffle:
            np.random.shuffle(indices)
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + batch_size
            if c_end > end:
                c_end = end
            _batches.append(indices[curr:c_end])
            curr = c_end
        return _batches

    def prepare_data(self, _entries, indices):
        batch_size = len(indices)
        features = np.zeros(shape=(batch_size, self.hdim))
        targets = np.zeros(shape=(batch_size))
        ids = np.zeros(shape=(batch_size))
        for tidx, idx in enumerate(indices):
            entry = _entries[idx]
            assert isinstance(entry, DataEntry)
            targets[tidx] = entry.label
            ids[tidx] = entry._id
            for feature_idx in range(self.hdim):
                features[tidx, feature_idx] = entry.features[feature_idx]
        return torch.FloatTensor(features), torch.LongTensor(targets), torch.LongTensor(ids),
        pass

    def find_same_class_data(self, ignore_indices):
        positive_indices_pool = set(self.positive_indices_in_train).difference(ignore_indices)
        negative_indices_pool = set(self.negative_indices_in_train).difference(ignore_indices)
        return self.find_triplet_loss_data(
            ignore_indices, negative_indices_pool, positive_indices_pool)

    def find_different_class_data(self, ignore_indices):
        positive_indices_pool = set(self.negative_indices_in_train).difference(ignore_indices)
        negative_indices_pool = set(self.positive_indices_in_train).difference(ignore_indices)
        return self.find_triplet_loss_data(
            ignore_indices, negative_indices_pool, positive_indices_pool)

    def find_triplet_loss_data(self, ignore_indices, negative_indices_pool, positive_indices_pool):
        indices = []
        for eidx in ignore_indices:
            if self.train_entries[eidx].is_positive():
                indices_pool = positive_indices_pool
            else:
                indices_pool = negative_indices_pool
            indices_pool = list(indices_pool)
            indices.append(np.random.choice(indices_pool))
        features, _,_ = self.prepare_data(self.train_entries, indices)
        return features


def create_dataset(ggnn_output, batch_size=32, output_buffer=sys.stderr):
    if output_buffer is not None:
        print('Reading Train data from', file=output_buffer)
    train_data, train_labels = ggnn_output[0] 
    print(len(train_data),len(train_labels))
    hdim = len(train_data[0])
    print('hdim:', hdim)
    dataset = DataSet(batch_size=batch_size, hdim=hdim)
    for i in range(len(train_data)):
        dataset.add_data_entry(train_data[i].cpu().numpy(), train_labels[i].cpu().numpy(), part='train')

    if output_buffer is not None:
        print('Reading Valid data from', file=output_buffer)
    valid_data, valid_labels = ggnn_output[1]
    print(len(valid_data),len(valid_labels))
    for i in range(len(valid_data)):
        dataset.add_data_entry(valid_data[i].cpu().numpy(), valid_labels[i].cpu().numpy(), part='valid')

    if output_buffer is not None:
        print('Reading Test data from', file=output_buffer)
    test_data, test_labels = ggnn_output[2]
    print(len(test_data),len(test_labels))

    for i in range(len(test_data)):
        dataset.add_data_entry(test_data[i].cpu().numpy(), test_labels[i].cpu().numpy(), part='test')

    return dataset
