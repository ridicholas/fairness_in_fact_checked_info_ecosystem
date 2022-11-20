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
import progressbar
import os
from scipy.stats import beta, rankdata
import pickle
# making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))






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
            data['lambda'] = np.random.uniform(0.5,0.75)
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
        data['num_read'] = {}

        for topic in range(numTopics):
            data['impactedness'][topic] = np.max([0, np.random.normal(loc=impactednesses[topic][data['Community']], scale=0.1)]) #making it a gaussian for now
            data['sentiment'][topic] = np.max([0, np.random.normal(loc=sentiments[topic][data['Community']], scale=0.1)]) #making it a gaussian for now
            data['num_read'][topic] = np.max([0, np.random.normal(impactednesses[topic][data['Community']]*100, scale = 10)])

        data['belief'] = np.array(list(data['sentiment'].values())).mean() #make belief an average of sentiments? then what we are interested in are changes in belief due to misinfo?



        if data['kind'] != 'bot':
            if data['belief'] < 0.2: #this might need to be adjusted depending on how belief figures look
                data['kind'] = 'beacon'



    ## Remove self_loops and isololates
    G.remove_edges_from(list(nx.selfloop_edges(G, data=True)))
    G.remove_nodes_from(list(nx.isolates(G)))

    bb = nx.betweenness_centrality(G)
    degrees = dict(G.in_degree())
    nx.set_node_attributes(G, bb, "centrality")
    nx.set_node_attributes(G, degrees, "degree")

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
        median = np.median(node_sentiments)
        deviations = [np.absolute(i - median) for i in node_sentiments]
        rankings['sentiment' + str(topic)] = node_sentiments
        rankings['deviation' + str(topic)] = deviations
        rankings['rank' + str(topic)] = np.where(rankings['sentiment' + str(topic)] < median,
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

def create_claims(num_claims):

    def type_func(x):
        if x < int(num_claims/3):
            return 'anti-misinfo'
        elif x >= int(num_claims/3) and x < int((2*num_claims)/3):
            return 'noise'
        else:
            return 'misinfo'    
    
    def virality_func(x):
        if x == 'anti-misinfo':
            return 1 + np.random.beta(a=3, b=7, size=1)[0]
        elif x == 'noise':
            return 1 + np.random.beta(a=1, b=9, size=1)[0]
        else:
            return 1 + np.random.beta(a=6,b=4,size=1)[0]
        
    claims = pd.DataFrame(data=[i for i in range(num_claims)], columns = ['claim_id'])
    claims['type'] = claims['claim_id'].apply(type_func)
    claims['virality'] = claims['type'].apply(virality_func) 
    
    c = claims.set_index('claim_id')
    c_dict = c.to_dict('index')
    return c_dict
    
    
    

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

def retweet_behavior(topic, value, topic_sentiment, creator_prestige, claim_virality):
    if value == -1:
        retweet_perc = np.min([1, (1 - topic_sentiment)*creator_prestige*claim_virality])
    elif value == 0:
        retweet_perc = np.min([1, (0.5)*creator_prestige*claim_virality])
    elif value == 1: # misinfo is retweeted 70% more than real news
        retweet_perc = np.min([1, topic_sentiment*creator_prestige*claim_virality])
    return retweet_perc


def update_topic_sentiment(current_sentiment, tweet_value, tweet_impactedness, num_read, learning_rate):
    # iterative mean updating of beliefs
    new_sentiment = ((num_read - 1)/num_read)*current_sentiment + (1/num_read)*tweet_value*(1+tweet_impactedness)
    difference = learning_rate*(new_sentiment - current_sentiment)
    return current_sentiment + difference



print('running....')
#path = '/Users/tdn897/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/data/nodes_with_community.gpickle'
path = '../../data/nodes_with_community.gpickle'
outpath_info = '../output/all_info.pickle'
outpath_node_info = '../output/node_info.pickle'
outpath_community_sentiment = '../output/community_sentiment.pickle'
outpath_node_time_info = '../output/node_time_info.pickle'
outpath_checkworthy = '../output/checkworthy_data.pickle'
subset_graph_file = '../output/simulation_net.gpickle'
num_topics = 4
communities_to_subset = [3,56,43]
learning_rate = 0.2
NUM_CLAIMS = 6000 #this is number of claims per topic per timestep
runtime = 500
agg_interval = 3
agg_steps = 3
perc_nodes_to_subset = 0.1
perc_bots = 0.1
load_data = False
update_beliefs = True
'''
First topic (row) impacts every group the same, the other topics each impact
one group significantly more than the others
'''


impactednesses = [{3: 0.5, 56: 0.5, 43: 0.5},
                  {3: 0.8, 56: 0.3, 43: 0.3},
                  {3: 0.3, 56: 0.8, 43: 0.3},
                  {3: 0.3, 56: 0.3, 43: 0.8}]

'''
For the first topic (which everyone cares about equally), belief is roughly average (50%).
For the other topics, if a community is more impacted by a topic, we assume that their
average belief is lower, indicating that they have more knowledge of the truth than the
other communities that are not impacted  by the topic.
'''
sentiments = [{3: 0.5, 56: 0.5, 43: 0.5},
              {3: 0.3, 56: 0.8, 43: 0.8},
              {3: 0.8, 56: 0.3, 43: 0.8},
              {3: 0.8, 56: 0.8, 43: 0.3}]


if load_data:
        print('Loading from file')
        with open(subset_graph_file, "rb") as f:
            sampleG = pickle.load(f)
else:
    print('simulation net file not found, creating it now')
    G = nx.read_gpickle(path)
    subG = subset_graph(G, communities=communities_to_subset)
    sampleG = create_simulation_network(G=subG,
                              perc_nodes_to_use = perc_nodes_to_subset,
                              numTopics = num_topics,
                              perc_bots = perc_bots,
                              impactednesses = impactednesses,
                              sentiments = sentiments)
    del G
    del subG

    nx.write_gpickle(sampleG, subset_graph_file)



def run(G, runtime, agg_interval=3, agg_steps=3):
    '''
    This executes a single run of the twitter_sim ABM model
    '''

    '''
    Create data structures to capture simulation output
    '''
    prestige_values = percentile(list(dict(G.in_degree()).values()))
    nodes = list(G.nodes())
    prestige = {nodes[i]: prestige_values[i] for i in range(len(nodes))}
    # Initialize objects to collect results
    all_info = {}
    # This will capture the unique-ids of each tweet read by each node.
    node_read_tweets = {node:[] for node in G.nodes()}
    community_sentiment_through_time = {com:{t:{topic: [] for topic in range(num_topics)} for t in range(runtime)} for com in communities_to_subset}
    node_read_tweets_by_time = {node:{t: [] for t in range(runtime)} for node in G.nodes()}
    topics = list(range(num_topics))
    all_claims = create_claims(num_claims = NUM_CLAIMS)
    checkworthy_data = {}

    


    bar = progressbar.ProgressBar()
    for step in bar(range(runtime)):
        # Loop over all nodes
        '''
        Users and Information interact
        '''
        rankings = calculate_sentiment_rankings(G = G, topics = topics)
        for node, data in G.nodes(data=True):
            # Check if User logs on for this Time Step
           # print(node + '\n\n\n')
            if data['wake'] == step:
                # Get new 'wake' time
                #print('\n\n\n They are awake! - ' + str(node) + '\n\n\n')

                data['wake'] = data['wake'] + \
                    np.round(1 + np.random.exponential(scale=1 / data['lambda']))

                '''
                Create tweets
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
                        claim_id = str(topic) + '-' + str(claim)
                        all_info.update({unique_id: {'topic':topic,'value':value,'claim':claim,'node-origin':node,'time-origin':step}})
                        new_tweets.append(unique_id)
                        if claim_id not in checkworthy_data.keys():
                            checkworthy_data.update({claim_id: {'topic':topic,
                            'value':value,
                            'claim':claim,
                            'num_of_origins': 1, 
                            'avg_degree_of_origins': data['degree'], 
                            'max_degree_of_origins': data['degree'],
                            'avg_centrality_of_origins': data['centrality'], 
                            'max_centrality_of_origins': data['centrality']}})
                        else:
                            checkworthy_data[claim_id]['num_of_origins'] += 1
                            if data['degree'] > checkworthy_data[claim_id]['max_degree_of_origins']:
                                checkworthy_data[claim_id]['max_degree_of_origins'] = data['degree']
                            if data['centrality'] > checkworthy_data[claim_id]['max_centrality_of_origins']:
                                checkworthy_data[claim_id]['max_centrality_of_origins'] = data['centrality']
                            checkworthy_data[claim_id]['avg_degree_of_origins'] += (data['degree'] - checkworthy_data[claim_id]['avg_degree_of_origins'])/checkworthy_data[claim_id]['num_of_origins']
                            checkworthy_data[claim_id]['avg_centrality_of_origins'] += (data['centrality'] - checkworthy_data[claim_id]['avg_centrality_of_origins'])/checkworthy_data[claim_id]['num_of_origins']
                            

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
                            read_claim = all_info[read_tweet]['claim']
                            virality = all_claims[read_claim]['virality']
                            topic_sentiment = data['sentiment'][topic]
                            creator_prestige = prestige[all_info[read_tweet]['node-origin']]

                            '''
                            update beliefs for each topic
                            '''
                            if update_beliefs and data['kind'] != 'bot':
                                data['num_read'][topic] += 1
                                data['sentiment'][topic] = update_topic_sentiment(current_sentiment=data['sentiment'][topic],
                                                                                  tweet_value = value,
                                                                                  tweet_impactedness=data['impactedness'][topic],
                                                                                  num_read = data['num_read'][topic],
                                                                                  learning_rate = learning_rate)
                            '''
                            retweet behavior
                            '''
                            perc = retweet_behavior(topic = topic,
                                                    value=value,
                                                    topic_sentiment=topic_sentiment,
                                                    creator_prestige=creator_prestige,
                                                    claim_virality=virality)

                            retweet_perc.append(perc)
                            new_retweets.append(read_tweet)
                            # updates the tweets that nodes have read
                            node_read_tweets[node].append(read_tweet)
                            node_read_tweets_by_time[node][step].append(read_tweet)
                            #update checkworthy data figures
                            time_feature = int((step - all_info[read_tweet]['time-origin'])/agg_interval)+1
                            if time_feature <= agg_steps:
                                origin_node = read_tweet.split('-')[2]
                                claim_id = read_tweet.split('-')[0] + '-' + read_tweet.split('-')[1]
                                if 'step{}_nodes_visited'.format(time_feature) not in checkworthy_data[claim_id].keys():
                                    checkworthy_data[claim_id]['step{}_nodes_visited'.format(time_feature)] = 1
                                    checkworthy_data[claim_id]['step{}_avg_degree_visited'.format(time_feature)] = data['degree']
                                    checkworthy_data[claim_id]['step{}_avg_centrality_visited'.format(time_feature)] = data['centrality']
                                    checkworthy_data[claim_id]['step{}_max_degree_visited'.format(time_feature)] = data['degree']
                                    checkworthy_data[claim_id]['step{}_max_centrality_visited'.format(time_feature)] = data['centrality']
                                    checkworthy_data[claim_id]['step{}_max_depth_from_origin'.format(time_feature)] = 1
                                else:
                                    checkworthy_data[claim_id]['step{}_nodes_visited'.format(time_feature)] += 1
                                    if data['degree'] > checkworthy_data[claim_id]['step{}_max_degree_visited'.format(time_feature)]:
                                        checkworthy_data[claim_id]['step{}_max_degree_visited'.format(time_feature)] = data['degree']
                                    if data['centrality'] > checkworthy_data[claim_id]['step{}_max_centrality_visited'.format(time_feature)]:
                                        checkworthy_data[claim_id]['step{}_max_centrality_visited'.format(time_feature)] = data['centrality']

                                    checkworthy_data[claim_id]['step{}_avg_degree_visited'.format(time_feature)] += (data['degree'] - checkworthy_data[claim_id]['step{}_avg_degree_visited'.format(time_feature)])/checkworthy_data[claim_id]['step{}_nodes_visited'.format(time_feature)]
                                    checkworthy_data[claim_id]['step{}_avg_centrality_visited'.format(time_feature)] += (data['centrality'] - checkworthy_data[claim_id]['step{}_avg_centrality_visited'.format(time_feature)])/checkworthy_data[claim_id]['step{}_nodes_visited'.format(time_feature)]
                                    
                                    
                                    distance = nx.shortest_path_length(G, node, origin_node) #worried this will take too long
                                    if distance > checkworthy_data[claim_id]['step{}_max_depth_from_origin'.format(time_feature)]:
                                        checkworthy_data[claim_id]['step{}_max_depth_from_origin'.format(time_feature)] = distance




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

                '''
                Capture sentiment across topics for node
                '''
                for topic in range(num_topics):
                    community_sentiment_through_time[data['Community']][step][topic].append(data['sentiment'][topic])


    return all_info, node_read_tweets, community_sentiment_through_time, node_read_tweets_by_time, checkworthy_data






all_info, node_read_tweets, community_sentiment_through_time, node_read_tweets_by_time, checkworthy_data = run(G = sampleG, runtime = runtime, agg_interval=agg_interval, agg_steps=agg_steps)


with open(outpath_info, 'wb') as file:
    pickle.dump(all_info, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_node_info, 'wb') as file:
    pickle.dump(node_read_tweets, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_community_sentiment, 'wb') as file:
    pickle.dump(community_sentiment_through_time, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_node_time_info, 'wb') as file:
    pickle.dump(node_read_tweets_by_time, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_checkworthy, 'wb') as file:
    pickle.dump(checkworthy_data, file, protocol=pickle.HIGHEST_PROTOCOL)


