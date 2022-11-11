#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:23:24 2022

@author: tdn897
"""

import pandas as pd
import networkx as nx
import random
import numpy as np
from numpy.random import choice
from sklearn.metrics import pairwise_distances
import progressbar
import argparse
import operator
import os
import time
from scipy.stats import beta, rankdata
import pickle
# making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# having trouble importing this from the other folder so just copied and pasted it here for now




def scale(x):
    '''
    Normalizes a vector by dividing each element by the vector max.
    '''
    x = np.array(x)
    return(x/np.max(x))


def percentile(x):
    x = np.array(x)
    ranks = rankdata(x)
    return(ranks/len(x))


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
            data['kind'] = 'normal'

        #set everything else
        data['impactedness'] = {}
        data['sentiment'] = {}

        for topic in range(numTopics):
            data['impactedness'][topic] = np.max([0, np.random.normal(loc=impactednesses[topic][data['Community']], scale=0.1)]) #making it a gaussian for now
            data['sentiment'][topic] = np.max([0, np.random.normal(loc=sentiments[topic][data['Community']], scale=0.1)]) #making it a gaussian for now
        
        data['belief'] = np.array(list(data['sentiment'].values())).mean() #make belief an average of sentiments? then what we are interested in are changes in belief due to misinfo?


        
        if data['kind'] != 'bot':
            if data['belief'] < 0.2: #this might need to be adjusted depending on how belief figures look
                data['kind'] = 'beacon'



    ## Remove self_loops and isololates
    G.remove_edges_from(list(nx.selfloop_edges(G, data=True)))
    G.remove_nodes_from(list(nx.isolates(G)))
    
    return(G)



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
        deviations = [np.absolute(i - 0.5) for i in node_sentiments]
        rankings['sentiment' + str(topic)] = node_sentiments
        rankings['deviation' + str(topic)] = deviations
        rankings['rank' + str(topic)] = np.where(rankings['sentiment' + str(topic)] < 0.5,
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


def choose_claim(value: int, num_claims: int):
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
        claim = random.sample(list(range(0,int(num_claims/3))), k=1)[0]
    elif value == 0:
        claim = random.sample(list(range(int(num_claims/3),int(num_claims/3)*2)), k=1)[0]
    elif value == 1:
        claim = random.sample(list(range(int(num_claims/3)*2,num_claims)), k=1)[0]
    return claim

def subset_graph(G, communities=None):
    """
    If communities is not None, only return graph of nodes in communities subset.

    param G: input graph
    param communities: list of int
    """

    # filter graph to desired community subset
    comm_list = nx.get_node_attributes(G, 'Community')
    nodes = list(G.nodes)
    G2 = G.copy()
    if communities is not None:
        for node in nodes:
            if comm_list[node] not in communities:
                G2.remove_node(node)

    #nx.write_gexf(G2, outpath)

    return G2

def retweet_behavior(topic, value, topic_sentiment, creator_prestige):
    if value == -1:
        retweet_perc = (1 - topic_sentiment)*creator_prestige
    elif value == 0:
        retweet_perc = (0.5)*creator_prestige
    elif value == 1:
        retweet_perc = topic_sentiment*creator_prestige
    return retweet_perc

   

print('running....')
#path = '/Users/tdn897/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/data/nodes_with_community.gpickle'
path = '../../data/nodes_with_community.gpickle'
outpath_info = '../output/all_info.pickle'
outpath_node_info = '../output/node_info.pickle'
outpath_node_time_info = '../output/node_time_info.pickle'
num_topics = 4



'''
First topic (row) impacts every group the same, the other topics each impact 
one group significantly more than the others
'''


impactednesses = [{3: 0.5, 56: 0.5, 43: 0.5},
                  {3: 0.8, 56: 0.1, 43: 0.1},
                  {3: 0.1, 56: 0.8, 43: 0.1},
                  {3: 0.1, 56: 0.1, 43: 0.8}]

'''
For the first topic (which everyone cares about equally), belief is roughly average (50%). 
For the other topics, if a community is more impacted by a topic, we assume that their 
average belief is lower, indicating that they have more knowledge of the truth than the 
other communities that are not impacted  by the topic.
'''
sentiments = [{3: 0.5, 56: 0.5, 43: 0.5}, 
              {3: 0.2, 56: 0.8, 43: 0.8}, 
              {3: 0.8, 56: 0.2, 43: 0.8},
              {3: 0.8, 56: 0.8, 43: 0.2}]


if os.path.isfile('../output/simulation_net.gpickle'):
        print('found simulation network file, loading from file')
        with open("../output/simulation_net.gpickle", "rb") as f:
            sampleG = pickle.load(f)
else:
    print('simulation net file not found, creating it now')
    G = nx.read_gpickle(path)
    subG = subset_graph(G, communities=[3,56,43])
    sampleG = create_simulation_network(G=subG, 
                              perc_nodes_to_use = 0.1,
                              numTopics = num_topics, 
                              perc_bots = 0.05, 
                              impactednesses = impactednesses, 
                              sentiments = sentiments)
    del G
    del subG

    nx.write_gpickle(sampleG, '../output/simulation_net.gpickle')


NUM_CLAIMS = 100 #this is number of claims per topic per timestep

def run(G, runtime):
    '''
    This executes a single run of the twitter_sim ABM model
    '''
    ##################################
    global_perception = 0.00000001
    num_topics = 4

    # Create scale free network

    '''
    Prestige is degree - this makes nodes more likely to retweet info that comes from high degree users
    '''
    prestige_values = percentile(list(dict(G.degree()).values()))
    nodes = list(G.nodes())
    prestige = {nodes[i]: prestige_values[i] for i in range(len(nodes))}
    # Initialize objects to collect results
    all_info = {}
    # This will capture the unique-ids of each tweet read by each node.
    node_read_tweets = {node:[] for node in G.nodes()}
    node_read_tweets_by_time = {node:{t: [] for t in range(runtime)} for node in G.nodes()}
    
    
    topics = list(range(num_topics))
    rankings = calculate_sentiment_rankings(G = G, topics = topics)


    bar = progressbar.ProgressBar()
    for step in bar(range(runtime)):
        # Loop over all nodes
        '''
        Users and Information interact
        '''
        for node, data in G.nodes(data=True):
            # Check if User logs on for this Time Step
           # print(node + '\n\n\n')
            if data['wake'] == step:
                # Get new 'wake' time
                #print('\n\n\n They are awake! - ' + str(node) + '\n\n\n')

                data['wake'] = data['wake'] + \
                    np.round(1 + np.random.exponential(scale=1 / data['lambda']))
                
                '''
                Tweeting Behavior
                '''
                
                if data['kind'] == 'bot':
                    chance = 1 # bots tweet every time they are awake
                elif data['kind'] != 'bot':
                    # humans tweet proportionally to their degree
                    chance = prestige[node]
                    
                new_tweets = []
                if chance > np.random.uniform():
                    #print('Here we go!\n\n\n Node ' + str(node) + '\n\n\n')
                    num_tweets = np.random.randint(1,10)
                    for i in range(num_tweets):
                        topic = choose_topic(data = data)
                        value = choose_info_quality(node = node, rankings = rankings, topic = topic, agent_type = data['kind'])
                        claim = choose_claim(value = value, num_claims=NUM_CLAIMS)
                        unique_id = str(topic) + '-' + str(claim) + '-' + str(node) + '-' + str(step)
                        all_info.update({unique_id: {'topic':topic,'value':value,'claim':claim,'node-origin':node,'time-origin':step}})
                        new_tweets.append(unique_id)
                #else:
                    #print('No Go \n\n\n Node ' + str(node) + '\n\n\n')
                               
                   
                '''
                    
                Read tweets, update beliefs, and re-tweet
                    
                '''               

                retweets = []
                if len(data['inbox']) > 0:
                    number_to_read = min(random.randint(4, 20), len(data['inbox'])) #should this be fully random?
                    read_tweets = data['inbox'][-number_to_read:]
                    retweet_perc = []
                    new_retweets = []
                    for read_tweet in read_tweets:
                        if read_tweet not in node_read_tweets[node]:
                            topic = all_info[read_tweet]['topic']
                            value = all_info[read_tweet]['value']
                            topic_sentiment = data['sentiment'][topic]
                            creator_prestige = prestige[all_info[read_tweet]['node-origin']]

                            '''
                            update beliefs here
                            '''
                            ## update beliefs
                            # if (perc + global_perception) > 0:
                            #     new_belief = data['belief'] + \
                            #         (perc + global_perception) * (1-data['belief'])
                            # else:
                            #     new_belief = data['belief'] + \
                            #         (perc + global_perception) * (data['belief'])
                            #data['belief'] = new_belief
                            ## 
                            '''
                            retweet behavior
                            '''
                            perc = retweet_behavior(topic = topic, value=value, topic_sentiment=topic_sentiment,creator_prestige=creator_prestige)
                            retweet_perc.append(perc)
                            new_retweets.append(read_tweet) 
                            # updates the tweets that nodes have read
                            node_read_tweets[node].append(read_tweet)
                            node_read_tweets_by_time[node][step].append(read_tweet)
                            
                    for i in range(len(new_retweets)):
                        if retweet_perc[i] > np.random.uniform():
                            retweets.append(new_retweets[i])
                    # clear inbox
                    data['inbox'] = []
                
                

                    '''
                    Pass information on to followers
                    '''
                new_tweets.extend(retweets)
                if len(new_tweets) > 0:
                    predecessors = G.predecessors(node)
                    for follower in predecessors:
                        G.nodes[follower]['inbox'].extend(new_tweets)


    return all_info, node_read_tweets, node_read_tweets_by_time



all_info, node_read_tweets, node_read_tweets_by_time = run(G = sampleG, runtime = 1000)

with open(outpath_info, 'wb') as file:
    pickle.dump(all_info, file, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(outpath_node_info, 'wb') as file:
    pickle.dump(node_read_tweets, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_node_time_info, 'wb') as file:
    pickle.dump(node_read_tweets_by_time, file, protocol=pickle.HIGHEST_PROTOCOL)
