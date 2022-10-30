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

#making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#having trouble importing this from the other folder so just copied and pasted it here for now
def subset_graph(G, outpath, communities=None):
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
    
    nx.write_gexf(G2, outpath)

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
            data['impactedness'][topic] = np.random.normal(loc=impactednesses[topic][data['Community']], scale=0.1) #making it a gaussian for now
            data['sentiment'][topic] = np.random.normal(loc=sentiments[topic][data['Community']], scale=0.1) #making it a gaussian for now
        
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
'''
print('running....')
path = '../../data/nodes_with_community.gpickle'

impactednesses = [{3: 0.5, 56: 0.5, 43: 0.5},{3: 0.1, 56: 0.8, 43: 0.1},{3: 0.8, 56: 0.1, 43: 0.8}]
sentiments = [{3: 0.5, 56: 0.5, 43: 0.5}, {3: 0.7, 56: 0.3, 43: 0.7},{3: 0.7, 56: 0.3, 43: 0.7}]

t = time.time()
G = create_simulation_network(G=subset_graph(nx.read_gpickle(path), 
                                communities=[3,56,43], outpath='sim_subset.gexf'), 
                                perc_nodes_to_use = 0.1,
                                numTopics = 3, 
                                perc_bots = 0.01, 
                                impactednesses = impactednesses, 
                                sentiments = sentiments)

print('making network took: {} seconds'.format(time.time() - t))

for node, data in G.nodes(data=True):
    print('node: {}, Community: {}, lambda {}, wake: {}, kind: {}, impactedness: {}, sentiment: {}'.format(node, 
                                                                                                              data['Community'],
                                                                                                              data['lambda'],
                                                                                                              data['wake'],
                                                                                                              data['kind'],
                                                                                                              data['impactedness'],
                                                                                                              data['sentiment']))

'''
    



    
    

        




        
        
        
            




def create_polarized_network(size = 100, bot_initial_links = 2, perc_bots = 0.05):
    '''
    Create polarized two community scale free network.  Populate basic node data
    for normal users, bots, and stiflers.
    '''
    # Create Scale Free Graph
    F = nx.scale_free_graph(size)
    H = nx.scale_free_graph(size)
    M = {}
    for num in range(size):
        M[num] = num + size
    H = nx.relabel_nodes(H, M, copy=False)
    G = nx.compose(F,H)
    
    for node, data in G.nodes(data=True):
        data['lambda'] = np.random.uniform(0.001,0.75)
        data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
        data['inbox'] = []
        data['mentioned_by'] = []

        
        
        
        '''
        This is the key to polarized network. Compose two graphs F, H of same size into a single network, then 
        for one group, have systematically lower belief (more truth). From this setup, it looks like beacons 
        can only exist in one group (where node < size).
        '''
        if node < size:
            data['belief'] = np.random.uniform(0,0.5)
        else:
            data['belief'] = np.random.uniform(0.5,1.0)
        if data['belief'] < 0.2:
            data['kind'] = 'beacon'
        else:
            data['kind'] = 'normal'
    
    '''
    Bots are added to the periphery of the network
    '''    
    #  Add Bots
    num_bots = int(np.round(len(G.nodes)*perc_bots))
    bot_names = [len(G) + i for i in range(num_bots)]
    for bot_name in bot_names:
        initial_links = random.sample(G.nodes, bot_initial_links)
        G.add_node(bot_name)
        for link in initial_links:
            G.add_edge(bot_name,link)
    # Add Bot Data      
    for node, data in G.nodes(data=True):
        if node in bot_names:
            '''
            Bots have a much higher inter-arival time
            '''
            data['lambda'] = np.random.uniform(0.1,0.75)
            data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
            data['inbox'] = []
            data['belief'] = np.random.uniform(0.95,1.0)
            data['kind'] = 'bot'
            data['mentioned_by'] = []
        
    ## Remove self_loops and isololates
    G.remove_edges_from(list(G.selfloop_edges()))
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # Ensure every node has outdegree > 0 (otherwise similarity fails)
    A = nx.adjacency_matrix(G).astype(bool)
    b = np.squeeze(np.asarray(A.sum(axis = 1)))
    b = np.argwhere(b==0)
    for node in b:
        connected = [to for (fr, to) in G.edges(node)]
        unconnected = [n for n in G.nodes() if not n in connected] 
        new = random.sample(unconnected,1)
        G.add_edge(node[0], new[0])
        
    return(G)



'''
We need a much more scalable method for assessing similarity 
if we intend to use this. Maybe we can just calculate distance?
'''

def link_prediction(G, node,similarity):
    '''
    This function takes the graph G, a given node, and the jaccard similarity 
    matrix for the nodes, and returns recommended link based on similarity.  
    '''
    ## Potential links are drawn from those whoe follow the same accounts 
    potential = []
    successors = G.successors(node)
    predecessors = list(G.predecessors(node)) 
    for successor in successors:
        friends = G.predecessors(successor)
        for friend in friends:
            if friend != node:
                potential.append(friend)
    # If potential exists, find highest similarity, otherwise sample from predecessors
    final = []
    if len(potential) > 0:
        jaccard1 = similarity[node,potential]
        i = np.argmax(jaccard1)
        link = (node,potential[i])
        if ~G.has_edge(link[0],link[1]):
            final.append(link)
    elif len(predecessors) > 0:
        get_one = random.sample(list(predecessors),1)
        link = (node,get_one[0])
        if ~G.has_edge(link[0],link[1]):
            final.append(link)
    return(final)


def run(G, size = 100, perc_bots = 0.05, strategy = 'normal', polarized = 'normal'):
    '''
    This executes a single run of the twitter_sim ABM model
    '''
    ##################################
    influence_proportion = 0.01
    bucket1 = [0,1]
    bucket2 = [0,-1]
    probability_of_link = 0.05
    dynamic_network = True
    global_perception = 0.00000001
    retweet_perc = 0.25
    allowed_successors = 0.2
    
    # Create scale free network
    
    #Create initial simlilarity and prestige arrays
    '''
    Note - we need to find another, more scalable way to measure similarity - cannot create a 315kx315k matrix
    '''
    A = nx.adjacency_matrix(G).astype(bool) 
    similarity = 1 - pairwise_distances(A.todense(), metric = 'jaccard')
    prestige = scale(list(dict(G.degree()).values()))
    
    # Initialize objects to collect results
    total_tweets = []
    all_beliefs = {'time':[],'user':[],'beliefs':[], 'kind':[]}
    bar = progressbar.ProgressBar()
    for step in bar(range(1680)):
        # Once a week we update the similarity matrix and Global Perception and prestige
        if (step % 168) == 0:
            A = nx.adjacency_matrix(G).astype(bool)
            similarity = 1 - pairwise_distances(A.todense(), metric = 'jaccard')
            prestige = scale(list(dict(G.in_degree()).values()))
            
            ## Update Global Perception
            if len(total_tweets) > 0:
                df = pd.concat(total_tweets)
                global_perception = 0.001*df['tweets'].mean()
        # Loop over all nodes
        '''
        Users and Information interact
        '''
        for node, data in G.nodes(data=True):
            all_beliefs['time'].append(step)
            all_beliefs['user'].append(node);
            all_beliefs['beliefs'].append(data['belief']);
            all_beliefs['kind'].append(data['kind'])
            # Check if User logs on for this Time Step
            if data['wake'] < step:
                retweets = []
                # Get new 'wake' time
                data['wake'] = data['wake'] + np.round(np.random.exponential(scale = 1 / data['lambda']))
                # Read Tweets
                if len(data['inbox']) > 0:
                    number_to_read = min(random.randint(4,20),len(data['inbox']))
                    '''
                    Tweets need to have a topic associated with them.
                    Maybe we should sort by impactedness here?
                    '''
                    read_tweets = data['inbox'][-number_to_read:]
                    perc = np.mean(read_tweets)
                    # Update Belief
                    if (perc + global_perception) > 0:
                        new_belief = data['belief'] +   (perc + global_perception) * (1-data['belief'])
                    else:
                        new_belief = data['belief'] +   (perc + global_perception) * (data['belief'])
                    data['belief'] = new_belief  
                    # Get retweets from read tweets
                    '''
                    retweet_perc should be dependent on the community's relation to a particular topic.
                    Groups that find a particular topic engaging or important have a higher chance of retweeting 
                    the tweet
                    '''
                    retweets = random.sample(read_tweets, round(retweet_perc*len(read_tweets)))
                # Send Tweets for bots
                if data['kind'] == 'bot':
                    chance = 0.8
                    tweets = list(choice(bucket1, np.random.randint(0,10),p=[1-chance, chance]))
                    
                # Send Tweets for Stiflers/Beacons
                elif (data['kind'] == 'beacon') and ('read_tweets' in locals()):
                    '''
                    Little unclear here. What is num_dis? We assume they read 30 tweets?
                    -1 represents anti-misinformation, so I believe the idea here is that
                    they will write up to 30 pieces of anti-disinformation because they tend
                    to be very active users
                    '''
                    read_tweets = data['inbox'][-30:]
                    num_dis = np.sum(np.array(read_tweets) > 0)
                    tweets = [-1] * num_dis
                    
                # Send Tweets for normal users
                else:
                    '''
                    Why is this the chance? What does their influence_proportion have to do with
                    their probability of retweeting noise and misinformation? Also, we assume that 
                    they won't retweet anti-misinformation. Why?
                    '''
                    chance = data['belief'] * influence_proportion
#                    chance = 0   # Normal users only send disinformation with retweets
                    tweets = list(choice(bucket1, np.random.randint(0,10),p=[1-chance, chance]))
                tweets.extend(retweets)
                total_tweets.append(pd.DataFrame({'tweets': tweets, 'time' :[step] * len(tweets)}))
                predecessors = G.predecessors(node)
                
                
                '''
                Pass information on to followers
                '''
                for follower in predecessors:
                    homophily = similarity[node,follower]
                    importance =  prestige[follower]
                    tweets = [homophily * importance * i for i in tweets]
                    G.nodes[follower]['inbox'].extend(tweets)
                    
                # Send Mentions
                neighbors = list(G.neighbors(node))
                '''
                What is going on with the mention?
                '''
                mention = random.sample(neighbors,1)[0]
                G.nodes[mention]['mentioned_by'].append(node)
                    
                
                '''
                Adding new links
                
                Perhaps instead of link_prediction() which computes a matrix, we could look at the set 
                of all nodes within a certain distance and randomly connect them using something like:
                
                radius = 3 # Degrees of separation
                new_graph = nx.generators.ego_graph(graph, node, radius=radius)
                random.sample(set(new_graph.nodes) - set(successors), 1)

                '''
                # Make sure doesn't have too many successors already
                successors = list(G.successors(node)) + [node]
                if len(successors) < allowed_successors * len(G.nodes) and (dynamic_network):
                    # If probabliliy right, add link for non-bot users
                    if (np.random.uniform(0,1) < probability_of_link) and (data['kind'] != 'bot'):
                        new_link = link_prediction(G,node,similarity)
                        if len(new_link) > 0:
                            G.add_edges_from(new_link) 

                    # If probabliliy right, add link to a mention
                    if (np.random.uniform(0,1) < probability_of_link) and (len(data['mentioned_by']) > 0):
                        new_link = random.sample(data['mentioned_by'],1)
                        if len(new_link) > 0:
                            G.add_edge(node, new_link[0]) 
                    # Bots try to add link every time
                    if (data['kind'] == 'bot'):
                        potential = list(set(G.nodes) - set(successors))
                        if len(potential) > 0:
                            if strategy == 'targeted':
                                degree = dict(G.in_degree(potential))
                                new_link = max(degree.items(), key=operator.itemgetter(1))[0]
                            else:
                                new_link = random.sample(list(potential),1)[0]
                            G.add_edge(node,new_link)
    return(pd.DataFrame(all_beliefs),pd.concat(total_tweets), G )