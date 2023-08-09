import re
import urllib.request
import urllib.error
from bs4 import BeautifulSoup
import ssl
import os
import numpy as np


def main():
    
    if not os.path.exists('./dataset/spider/answers.npy'):
        Comments = dict()
        Scores = dict()
        Accepted = dict()
    else:
        Comments = np.load('./dataset/spider/answers.npy', allow_pickle=True).tolist()
        Scores = np.load('./dataset/spider/scores.npy', allow_pickle=True).tolist()
        Accepted = np.load('./dataset/spider/accepted.npy', allow_pickle=True).tolist()

    posts = np.load('./dataset/spider/postIds.npy')
    
    for post in posts:
        if post in Comments.keys() and len(Comments[post]) != 0:
            continue
        html = ask_url_get_html('https://stackoverflow.com/questions/%s' % post)
        Accepted[post] = -1
        try:
            page_info, score, accept = get_data_from_html(html, answer_num=5)
            Comments[post] = page_info[1:]
            Scores[post] = score
            for idx, item in enumerate(accept):
                if 'accepted-answer' in item:
                    Accepted[post] = idx
                    
            np.save('./dataset/spider/answers.npy', Comments)
            np.save('./dataset/spider/scores.npy', Scores)
            np.save('./dataset/spider/accepted.npy', Accepted)
        except:
            pass


def ask_url_get_html(url):

    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"
    }

    request = urllib.request.Request(url, headers=head)
    html = None
    
    try:
        response = urllib.request.urlopen(request, context=ssl._create_unverified_context())
        html = response.read().decode("utf-8")
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
            html = e.code
        if hasattr(e, "reason"):
            print(e.reason)
    return html


def get_data_from_html(html, answer_num):
    
    find_description = re.compile('">(.*)</div>', re.DOTALL)
    find_score = re.compile('data-score="(.*?)"', re.DOTALL)
    find_accept = re.compile('answer js-answer(.*?)"', re.DOTALL)
    
    score = re.findall(find_score, html)
    accpet = re.findall(find_accept, html)
    
    soup = BeautifulSoup(html, "html.parser")
    item = str(soup.find_all("div", class_="s-prose js-post-body", limit=answer_num+2))
    description = re.findall(find_description, item)
    description = description[0].split('</div>, <div class="s-prose js-post-body" itemprop="text">')
    return description, score, accpet


if __name__ == '__main__':
    main()