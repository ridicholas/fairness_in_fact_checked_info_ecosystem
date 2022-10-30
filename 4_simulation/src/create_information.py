#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:54:03 2022

@author: tdn897
"""

import pandas as pd
import networkx as nx
import random
import numpy as np
from numpy.random import choice
from sklearn.metrics import pairwise_distances
#import progressbar
import argparse
import operator
import os
import time
from scipy.stats import beta
import collections
#making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#having trouble importing this from the other folder so just copied and pasted it here for now
def subset_graph(G, communities=None):
    """
    If communities is not None, only return graph of nodes in communities subset.

    param G: input graph
    param communities: list of int
    """

    #filter graph to desired community subset
    comm_list = nx.get_node_attributes(G, 'Community')
    nodes = list(G.nodes)
    G2 = G.copy()
    if communities is not None:
        for node in nodes:
            if comm_list[node] not in communities:
                G2.remove_node(node)
    
    #nx.write_gexf(G2, outpath)

    return G2

def scale(x):
    '''
    Normalizes a vector by dividing each element by the vector max.
    '''
    x = np.array(x)
    return(x/np.max(x))



def create_simulation_network(G: nx.digraph, perc_nodes_to_use: float, numTopics: int, perc_bots: float, impactednesses: list, sentiments: list):
    '''
    Will create a network for simulation using input graph and provided community level attitudes towards topics

    :param G: input digraph
    :param perc_nodes_to_use: percentage of nodes from G you want to keep for simulation
    :param numTopics: number of topics in use this simulation round
    :perc_bots: what percentage of nodes will be bots
    :param impactednesses:  len(numTopics) list of dictionaries. impactednesses[i] contains dictionary where keys are communities and values are the community's 
                            impactedness value towards topic i.
                            (this value will be used as mean for drawing distribution)
    :param sentiments: len(numTopics) list of dictionaries. sentiments[i] contains dictionary where keys are communities and values are the community's 
                            sentiment value towards topic i.
                            (this value will be used as mean for drawing distribution)
    :return: returns new network where nodes have impactedness, sentiments, and are bots
    
    '''

    to_remove = random.sample(list(G.nodes), int(len(G.nodes)*(1-perc_nodes_to_use)))
    to_keep = list(set(list(G.nodes)) - set(to_remove))
    G.remove_nodes_from(to_remove)

    # Randomly select who will be a bot
    num_bots = int(np.round(len(to_keep)*perc_bots))
    bot_names = random.sample(to_keep, num_bots) #might want to make it so we sample a certain number from each community instead

    for node, data in G.nodes(data=True):

        #set things that matter if you are bot or not
        if node in bot_names:
            data['lambda'] = np.random.uniform(0.1,0.75)
            data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
            data['inbox'] = []
            data['kind'] = 'bot'
            data['mentioned_by'] = []
        else:
            data['lambda'] = np.random.uniform(0.001,0.75)
            data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
            data['inbox'] = []
            data['mentioned_by'] = []

        #set everything else
        data['impactedness'] = {}
        data['sentiment'] = {}

        for topic in range(numTopics):
            data['impactedness'][topic] = np.max([0, np.random.normal(loc=impactednesses[topic][data['Community']], scale=0.1)]) #making it a gaussian for now
            data['sentiment'][topic] = np.max([0, np.random.normal(loc=sentiments[topic][data['Community']], scale=0.1)]) #making it a gaussian for now
        
        data['belief'] = np.array(list(data['sentiment'].values())).mean() #make belief an average of sentiments? then what we are interested in are changes in belief due to misinfo?


        
        if data['belief'] < 0.2: #this might need to be adjusted depending on how belief figures look
            data['kind'] = 'beacon'
        else:
            data['kind'] = 'normal'

    ## Remove self_loops and isololates
    G.remove_edges_from(list(nx.selfloop_edges(G, data=True)))
    G.remove_nodes_from(list(nx.isolates(G)))
    
    return(G)

   
#quick test to see if it works with 3 communities and 3 topics, uncomment below to run with test

print('running....')
path = '/Users/tdn897/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/data/nodes_with_community.gpickle'
num_topics = 4

impactednesses = [{3: 0.5, 56: 0.5, 43: 0.5},
                  {3: 0.8, 56: 0.1, 43: 0.1},
                  {3: 0.1, 56: 0.8, 43: 0.1},
                  {3: 0.1, 56: 0.1, 43: 0.8}]

sentiments = [{3: 0.5, 56: 0.5, 43: 0.5}, 
              {3: 0.4, 56: 0.7, 43: 0.7}, 
              {3: 0.7, 56: 0.4, 43: 0.7},
              {3: 0.7, 56: 0.7, 43: 0.4}]


impactednesses_by_com = {3: [0.5,0.8,0.1,0.1],
                         56:[0.5,0.1,0.8,0.1],
                         43:[0.5,0.1,0.1,0.8]}



t = time.time()
G = nx.read_gpickle(path)
subG = subset_graph(G, communities=[3,56,43])
sampleG = create_simulation_network(G=subG, 
                              perc_nodes_to_use = 0.1,
                              numTopics = num_topics, 
                              perc_bots = 0.05, 
                              impactednesses = impactednesses, 
                              sentiments = sentiments)

print('making network took: {} seconds'.format(time.time() - t))


def calculate_sentiment_rankings(G: nx.DiGraph, topics: list):

    '''
    This function returns a pandas DataFrame with all nodes' percentile rankings of deviation from mean sentiment across all topics.
    This ranking is multiplied by -1 if they have a negative deviation and by +1 if they have a positive deviation,
    creating a range of possible values [-1,1].
    
    This pandas dataframe is used as an input to modify the distribution from which agents draw their quality of information when tweeting. 
    A higher rank value in the dataframe results in a higher probability of creating misinformation. 
    This should be intuitive... if someone's sentiment is already high, they are
    more likely to create misinformation. If someone's sentiment is low, they are more likely to produce anti-misinformation.
    
    One potential issue here is if sentiment is tightly clustered for all agents, this will sort of artificially make some agents produce more/less misinformation in that case.
    '''
    all_node_sentiments = nx.get_node_attributes(G, 'sentiment')
    rankings = pd.DataFrame(index = all_node_sentiments.keys())

    for topic in topics:
        node_sentiments = [all_node_sentiments[key][topic] for key in all_node_sentiments.keys()]
        sent_mean = np.mean(node_sentiments)
        deviations = [np.absolute(i - sent_mean) for i in node_sentiments]
        rankings['sentiment' + str(topic)] = node_sentiments
        rankings['deviation' + str(topic)] = deviations
        rankings['rank' + str(topic)] = np.where(rankings['sentiment' + str(topic)] < sent_mean,
                                                 -1*rankings['deviation' + str(topic)].rank(method='max')/len(rankings),
                                                 rankings['deviation' + str(topic)].rank(method='max')/len(rankings))
        
    return rankings



def choose_topic(data: dict):
    topic_probs = [i / sum(data['impactedness'].values()) for i in data['impactedness'].values()]
    topic = choice(np.arange(0, len(topic_probs)), p=topic_probs)
    return topic

def choose_info_quality(node: str, rankings: pd.DataFrame, topic: int, agent_type: str):
    
    '''
    For each (non-bot) type, we draw from a beta distribution with beta(5 - a, 5 + a), and we shift the parameters according
    to their percentile rankings of sentiment deviation from the mean, such that those with low sentiment 
    produce more anti-misinformation, and those with high sentiment produce more misinformation, but noise is always the most common
    info type produced.
    
    Because B(a,b) is bounded by (0, 1), we can just use thirds as cut points to effectively give different 
    probabilistic weight to information quality in {-1, 0, 1}.
    
    Bots produce misinformation 80% of the time
    '''
    if agent_type != 'bot':
        deviation_rank = rankings.loc[node].loc['rank' + str(topic)]
        raw = beta.rvs(5 + deviation_rank, 5 - deviation_rank, size=1)
        value = np.where(raw < 0.3333, -1, np.where(raw >= 0.3333 and raw < 0.666, 0, 1))[0]
    else: 
        value = np.where(np.random.uniform(size=1) > 0.2, 1, 0)[0]
    return value


def choose_claim(value: int):
    '''
    Within topics, there is a high-dimensional array of "potential claims". This (topic, claim) pair 
    is the main feature we will use to train the fact-checking algorithm. Claims are partitioned by the quality of information
    so that we don't have agents posting {-1,0,1} all relative to the same claim.'
    
    Parameters
    ----------
    value : quality of informaiton {-1, 0, 1} if anti-misinformation, noise, misinformation
    Returns
    -------
    claim number : (0-33) if anti-misinfo, (34-66) if noise, (66-100) if misinfo.

    '''
    
    if value == -1:
        claim = random.sample(list(range(0,33)), k=1)[0]
    elif value == 0:
        claim = random.sample(list(range(33,66)), k=1)[0]
    elif value == 1:
        claim = random.sample(list(range(66,100)), k=1)[0]
    return claim



all_info = {'topic':[],'claim':[],'value':[],'node-origin':[],'node-community':[], 'time-origin':[]}
topics = list(range(num_topics))
t = 1
rankings = calculate_sentiment_rankings(G = sampleG, topics = topics)

'''
This is an example run, where we create information for every node in the dataset.
Below, we explore the distributions produced by topic, value, and community. 
The results seem promising, we seem to have introduced a lot of high quality stochasticity.
'''


for node, data in sampleG.nodes(data=True):
    topic = choose_topic(data = data)
    value = choose_info_quality(node = node, rankings = rankings, topic = topic, agent_type = data['kind'])
    claim = choose_claim(value = value)
    all_info['topic'].append(topic)
    all_info['value'].append(value)
    all_info['claim'].append(claim)
    all_info['node-community'].append(data['Community'])
    all_info['node-origin'].append(node)
    all_info['time-origin'].append(t)



dict(collections.Counter(all_info['topic']))
dict(collections.Counter(all_info['value']))

pd.crosstab(all_info['topic'], all_info['node-community'])
pd.crosstab(all_info['topic'], [all_info['value'], all_info['node-community']], rownames = ['topic'], colnames = ['value', 'community'])



## What is the quality of the data?


## Construct Information


