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
from checkworthy import *
from sim_util import *
# making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))


print('running....')
#path = '/Users/tdn897/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/data/nodes_with_community.gpickle'
path = '../../data/nodes_with_community.gpickle'
outpath_info = '../output/all_info.pickle'
outpath_node_info = '../output/node_info.pickle'
outpath_community_sentiment = '../output/community_sentiment.pickle'
outpath_node_time_info = '../output/node_time_info.pickle'
outpath_checkworthy = '../output/checkworthy_data.pickle'
subset_graph_file = '../output/simulation_net.gpickle'
outpath_node_data_info = '../output/node_metadata.gpickle'
outpath_stored_model = '../output/stored_model.pickle'
outpath_all_claims = '../output/all_claims.pickle'

num_topics = 4
communities_to_subset = [3,56,43]
learning_rate = 0.2
NUM_CLAIMS = 6000 #this is number of claims per topic per timestep
runtime = 300
agg_interval = 3
agg_steps = 3
perc_nodes_to_subset = 0.1
perc_bots = 0.1
load_data = False
update_beliefs = True
depths = [2,4,6]
outcome_time = 48
MIN_DEGREE = 10

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



def run(G, runtime, agg_interval=3, agg_steps=3, outcome_time=48, impactednesses = impactednesses):
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
    check = checkworthy(agg_interval=agg_interval,
                        agg_steps = agg_steps,
                        G = G,
                        depths = depths,
                        outcome_time=outcome_time,
                        impactednesses=impactednesses)




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
                        '''
                        update checkworthy data only if original node has degree greater than some minimum
                        '''
                        if data['degree'] > MIN_DEGREE: 
                            check.intake_information(node = node, data = data, claim_id = claim_id, value = value, topic = topic, claim = claim)
                            
                            if claim_id not in check.checkworthy_data.keys():
                                check.update_keys()
                            else:
                                check.update_agg_values()


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
                            # update checkworthy data features
                            time_feature = int((step - all_info[read_tweet]['time-origin'])/agg_interval)+1
                            if time_feature <= agg_steps:
                                origin_node = read_tweet.split('-')[2]
                                claim_id = read_tweet.split('-')[0] + '-' + read_tweet.split('-')[1]
                                if data['degree'] > MIN_DEGREE: 
                                    check.intake_information(node = node, data = data, claim_id = claim_id, value = value, topic = topic, claim = read_claim)
                                    #this will count the first utterance by a node of degree > min degree as the origin
                                    if claim_id not in check.checkworthy_data.keys():
                                        check.update_keys()
                                    else:
                                        check.update_time_values(time_feature=time_feature, origin_node=origin_node)
                                
                                
                            # udpate checkworthy outcome label - average virality at t=48hrs
                            time_from_launch = step - all_info[read_tweet]['time-origin']
                            if time_from_launch <= outcome_time:
                                claim_id = read_tweet.split('-')[0] + '-' + read_tweet.split('-')[1]

                                if data['degree'] > MIN_DEGREE: 
                                    check.intake_information(node = node, data = data, claim_id = claim_id, value = value, topic = topic, claim = read_claim)
                                    #this will count the first utterance by a node of degree > min degree as the origin
                                    if claim_id not in check.checkworthy_data.keys():
                                        check.update_keys()
                                    else:
                                        check.update_virality_outcome(time_from_launch=time_from_launch)

                                


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


    return all_info, node_read_tweets, community_sentiment_through_time, node_read_tweets_by_time, check, G, all_claims

print('\n\n\n\n\n ----------- Running Simulation --------- \n\n\n\n\n')


all_info, node_read_tweets, community_sentiment_through_time, node_read_tweets_by_time, check, G, all_claims = run(G = sampleG,
                                                                                                    runtime = runtime,
                                                                                                    agg_interval=agg_interval,
                                                                                                    agg_steps=agg_steps,
                                                                                                    outcome_time=outcome_time)


print('\n\n\n\n\n ----------- Writing Data --------- \n\n\n\n\n')

with open(outpath_info, 'wb') as file:
    pickle.dump(all_info, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_all_claims, 'wb') as file:
    pickle.dump(all_claims, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_node_info, 'wb') as file:
    pickle.dump(node_read_tweets, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_community_sentiment, 'wb') as file:
    pickle.dump(community_sentiment_through_time, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_node_time_info, 'wb') as file:
    pickle.dump(node_read_tweets_by_time, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_checkworthy, 'wb') as file:
    pickle.dump(check, file, protocol=pickle.HIGHEST_PROTOCOL)
    
nx.write_gpickle(G, outpath_node_data_info)










