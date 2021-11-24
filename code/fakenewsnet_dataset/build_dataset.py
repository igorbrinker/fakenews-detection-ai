import json
import os
import numpy as np
import pandas as pd

def load_data(news_source, truthfulness):
    temp = []
    
    for i in os.listdir('{}/{}/'.format(news_source, truthfulness)):
        try:
            with open('{}/{}/{}/news content.json'.format(news_source, truthfulness, i), 'rb') as file:
                article = json.load(file)
                
            tweets = []
            for j in os.listdir('{}/{}/{}/tweets/'.format(news_source, truthfulness, i)):
                with open('{}/{}/{}/tweets/{}'.format(news_source, truthfulness, i, j), 'rb') as file2:
                    tweet = json.load(file2)
                tweets.append(tweet)
                
        except FileNotFoundError:
            continue
        except NotADirectoryError:
            continue
            
        temp.append({'id': i, 'article': article, 'tweets': tweets})
            
    return temp

def gather_twitter_stats(tweets):
    follower_counts = []
    friends_counts = []
    favorite_counts = []
    retweet_counts = []
    statuses_counts = []
    verified_counter = 0
    
    for tweet in tweets:
        follower_counts.append(tweet['user']['followers_count'])
        friends_counts.append(tweet['user']['friends_count'])
        favorite_counts.append(tweet['favorite_count'])
        retweet_counts.append(tweet['retweet_count'])
        statuses_counts.append(tweet['user']['statuses_count'])
        if tweet['user']['verified']:
            verified_counter += 1
            
    return {
        'followers_mean': np.mean(follower_counts),
        'followers_std': np.std(follower_counts),
        'followers_median': np.median(follower_counts),
        'followers_sum': np.sum(follower_counts),
        'friends_mean': np.mean(friends_counts),
        'friends_std': np.std(friends_counts),
        'friends_median': np.median(friends_counts),
        'friends_sum': np.sum(friends_counts),
        'favorites_mean': np.mean(favorite_counts),
        'favorites_std': np.std(favorite_counts),
        'favorites_median': np.median(favorite_counts),
        'favorites_sum': np.sum(favorite_counts),
        'retweets_mean': np.mean(retweet_counts),
        'retweets_std': np.std(retweet_counts),
        'retweets_median': np.median(retweet_counts),
        'retweets_sum': np.sum(retweet_counts),
        'statuses_mean': np.mean(statuses_counts),
        'statuses_std': np.mean(statuses_counts),
        'statuses_median': np.median(statuses_counts),
        'statuses_sum': np.sum(statuses_counts),
        'verified_count': verified_counter
    }

def process_example(ex, label):
    uid = ex['id']
    article_text = ex['article']['text']
    article_title = ex['article']['title']
    article_source = ex['article']['source']
    
    temp = {'id': uid, 'title': article_title, 'text': article_text, 'source': article_source, 'label': label}
    temp.update(gather_twitter_stats(ex['tweets']))
    
    return temp

pf_fake = load_data('politifact', 'fake')
pf_fake = [process_example(x, 'fake') for x in pf_fake]
pf_real = load_data('politifact', 'real')
pf_real = [process_example(x, 'real') for x in pf_real]

df = pd.DataFrame(pf_fake)
df = df.append(pf_real)

df.reset_index(inplace=True, drop=True)
df.to_csv('dataset.csv', index=False)