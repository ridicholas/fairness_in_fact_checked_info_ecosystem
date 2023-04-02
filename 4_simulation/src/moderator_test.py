#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:06:31 2023

@author: tdn897
"""

import networkx as nx
import numpy as np
import pandas as pd
import random
import os
from plotnine import *

os.chdir('/Users/tdn897/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/4_simulation/src')
communities = [3, 49, 72]
oversample_amt = 0.15
perc_nodes_to_use = 0.25


def calc_connections_between_communities(G, from_community, to_community):
    community = nx.get_node_attributes(G, 'Community')
    connection_count = 0
    for node in community:
        if community[node] == from_community:
            eg = nx.ego_graph(G, node)
            for n, d in eg.nodes(data=True):
                if d['Community'] == to_community:
                    connection_count += 1
    return connection_count
                    


G = nx.read_gexf('../../data/community_subset_network.gexf')

majority_community = communities[0]
minority_community = communities[1]
knowledgeable_community = communities[2]

reps = 20

moderators_list = ['none',
                   'increase-density', 
                   'reduce-density', 
                   'increase-connections-majority', 
                   'reduce-connections-majority', 
                   'increase-connections-knowledgeable', 
                   'reduce-connections-knowledgeable']

final_result = pd.DataFrame()

for rep in range(reps):
    
    print('\n\n\n ------- Rep: ' + str(rep) + ' ------- \n\n\n\n')
    
    for moderators in moderators_list:
        
        H = G.copy()
        
        print('\n\n\n ------- Rep: ' + str(rep) + ' Moderators: ' + moderators + ' ------- \n\n\n\n')

        if moderators != 'none':
                    
            community = nx.get_node_attributes(H, 'Community')
            to_remove = []
            minority_community_size = 0
                    
            # oversample nodes for minority community
            for node in community:
                if community[node] == minority_community:
                    minority_community_size += 1
                    if np.random.uniform() > perc_nodes_to_use + oversample_amt:
                        to_remove.append(node) 
                else:
                    if np.random.uniform() > perc_nodes_to_use:
                        to_remove.append(node)
                
            amt_to_remove = int(minority_community_size*oversample_amt)
            H.remove_nodes_from(to_remove)
            H.remove_edges_from(list(nx.selfloop_edges(H, data=True)))
            H.remove_nodes_from(list(nx.isolates(H)))

        
            
            if 'density' in moderators:
                
                cluster = nx.clustering(H)
                community = nx.get_node_attributes(H, 'Community')
                for node in community:
                    if community[node] != minority_community:
                        cluster.pop(node)
                        
                cluster_df = pd.DataFrame.from_dict(cluster, orient='index', columns=['clustering'])
                cluster_df['clustering'] = cluster_df['clustering'] + 0.001
                
                if moderators == 'reduce-density':
                    cluster_df['pct'] = (cluster_df['clustering'])/sum(cluster_df['clustering'])
                
                elif moderators == 'increase-density':
                    cluster_df['difference_clustering'] = max(cluster_df['clustering']) - cluster_df['clustering']
                    cluster_df['pct'] = (cluster_df['difference_clustering'])/sum(cluster_df['difference_clustering'])
                
                more_to_remove = np.random.choice(cluster_df.index.to_list(), 
                                                  size=amt_to_remove, 
                                                  p=cluster_df.pct.to_list(), 
                                                  replace=False).tolist()
                
                to_remove = list(to_remove) + list(more_to_remove)
                H.remove_nodes_from(more_to_remove)
                H.remove_edges_from(list(nx.selfloop_edges(H, data=True)))
                H.remove_nodes_from(list(nx.isolates(H)))
                to_keep = list(H.nodes)               
                
            elif 'connections-majority' in moderators:
                
                connections_dict = {}
                
                for node, data in H.nodes(data=True):
                    # -- 1. collect average per topic belief of nodes followed
                    # -- 2. collect number of bots followed
                    eg = nx.ego_graph(H, node)
                    num_in_majority_followed = 0
                    for n, d in eg.nodes(data=True):
                        if d['Community'] == majority_community and data['Community'] == minority_community:
                            num_in_majority_followed += 1
                    if data['Community'] == minority_community:
                        connections_dict.update({node: num_in_majority_followed})
                            
                connection_df = pd.DataFrame.from_dict(connections_dict, orient = 'index', columns = ['connections'])
                connection_df['connections'] += 1
                
                if moderators == 'reduce-connections-majority':
                    connection_df['pct'] = (connection_df['connections'])/sum(connection_df['connections'])
                    
                elif moderators == 'increase-connections-majority':
                    connection_df['difference_connections'] = max(connection_df['connections']) - connection_df['connections']
                    connection_df['pct'] = (connection_df['difference_connections'])/sum(connection_df['difference_connections'])
                
                more_to_remove = np.random.choice(connection_df.index.to_list(), 
                                                  size=amt_to_remove, 
                                                  p=connection_df.pct.to_list(), 
                                                  replace=False).tolist()
                
                to_remove = list(to_remove) + list(more_to_remove)
                H.remove_nodes_from(more_to_remove)
                H.remove_edges_from(list(nx.selfloop_edges(H, data=True)))
                H.remove_nodes_from(list(nx.isolates(H)))
                to_keep = list(H.nodes)  
        
        
            elif 'connections-knowledgeable' in moderators:
                
                connections_dict = {}
                
                for node, data in H.nodes(data=True):
                    # -- 1. collect average per topic belief of nodes followed
                    # -- 2. collect number of bots followed
                    eg = nx.ego_graph(H, node)
                    num_in_knowledgeable_community_followed = 0
                    for n, d in eg.nodes(data=True):
                        if d['Community'] == knowledgeable_community and data['Community'] == minority_community:
                            num_in_knowledgeable_community_followed += 1
                    if data['Community'] == minority_community:
                        connections_dict.update({node: num_in_knowledgeable_community_followed})
                            
                connection_df = pd.DataFrame.from_dict(connections_dict, orient = 'index', columns = ['connections'])
                connection_df['connections'] += 1
                
                if moderators == 'reduce-connections-knowledgeable':
                    connection_df['pct'] = (connection_df['connections'])/sum(connection_df['connections'])
                    
                elif moderators == 'increase-connections-knowledgeable':
                    connection_df['difference_connections'] = max(connection_df['connections']) - connection_df['connections']
                    connection_df['pct'] = (connection_df['difference_connections'])/sum(connection_df['difference_connections'])
                
                more_to_remove = np.random.choice(connection_df.index.to_list(), 
                                                  size=amt_to_remove, 
                                                  p=connection_df.pct.to_list(), 
                                                  replace=False).tolist()
                
                to_remove = list(to_remove) + list(more_to_remove)
                H.remove_nodes_from(more_to_remove)
                H.remove_edges_from(list(nx.selfloop_edges(H, data=True)))
                H.remove_nodes_from(list(nx.isolates(H)))
                to_keep = list(H.nodes)
        
        elif moderators == 'none':
            to_remove = random.sample(list(H.nodes), int(len(H.nodes)*(1-perc_nodes_to_use)))
            to_keep = list(set(list(H.nodes)) - set(to_remove))
            H.remove_nodes_from(to_remove)
            H.remove_edges_from(list(nx.selfloop_edges(H, data=True)))
            H.remove_nodes_from(list(nx.isolates(H)))

            
        #######################
        #### -- Gather summary stats
        #######################
        
        community = nx.get_node_attributes(H, 'Community')
        in_degree = dict(H.in_degree())
        out_degree = dict(H.out_degree())
        betweenness = nx.betweenness_centrality(H, k=int(0.1*len(list(H.nodes()))))
        clustering = nx.clustering(H)
        
        results = []
        for node in H:
            results.append([community[node], in_degree[node], out_degree[node], betweenness[node], clustering[node]])
            
        results_frame = pd.DataFrame(results, columns = ['community', 'in_degree', 'out_degree', 'betweenness', 'clustering'])\
            .groupby(['community'])\
                .mean()
                
        # attach connections between communities

        conn1 = {}
        conn2 = {}
        for c in communities:
            conn1.update({c: calc_connections_between_communities(H, from_community=c, to_community=majority_community)})
            conn2.update({c: calc_connections_between_communities(H, from_community=c, to_community=knowledgeable_community)})
            
        results_frame = results_frame.join(pd.DataFrame.from_dict(conn1, orient='index', columns=['connection_to_majority']))
        results_frame = results_frame.join(pd.DataFrame.from_dict(conn2, orient='index', columns=['connection_to_knowledgeable']))
        results_frame['rep'] = rep
        results_frame['moderator'] = moderators
        final_result = pd.concat([results_frame, final_result])
        
   
        
final_result.to_csv('../output/moderators_experiment/moderators_experiment_sampling.csv')


######################################
################## Plots
######################################

final_result = pd.read_csv('../output/moderators_experiment/v2-oversample=0.15/moderators_experiment_sampling.csv')
final_result['community'] = np.where(final_result['community']==minority_community,'Minority Community',
                                     np.where(final_result['community']==majority_community, 'Majority Community', 'Knowledgeable Community'))


final_result_long = pd.melt(final_result, id_vars = ['community', 'rep', 'moderator'])
final_result_long['variable'] = np.where(final_result_long['variable'].isin(['in_degree', 'out_degree', 'betweenness']), 'CV: ' + final_result_long['variable'],
                                         'IV: '+ final_result_long['variable'])

g = (ggplot(final_result_long)
     + geom_boxplot(aes(x='moderator', y='value', fill = 'moderator'))
     + facet_wrap('~ community + variable', scales='free_y', nrow=len(communities), ncol=len(moderators_list)-1)
     + theme(axis_text_x=element_blank(), 
             axis_text_y=element_blank(), 
             text=element_text(size=14),
             legend_position='bottom'))

g.save(filename='../output/moderators_experiment/v2-oversample=0.15/moderators_experiment_combined.png', width = 24, height = 8, dpi = 300)



# g1 = (ggplot(final_result)
#      + geom_boxplot(aes(x='moderator', y='in_degree', fill='moderator'))
#      + facet_wrap('~ community')
#      + theme(axis_text_x=element_blank()))

# g1.save(filename='../output/moderators_experiment/moderators_experiment_in_degree.png', width=10,height=6, dpi=800)



# g2 = (ggplot(final_result)
#      + geom_boxplot(aes(x='moderator', y='out_degree', fill='moderator'))
#      + facet_wrap('~ community')
#      + theme(axis_text_x=element_blank()))

# g2.save(filename='../output/moderators_experiment/moderators_experiment_out_degree.png', width=10,height=6, dpi=800)


# g3 = (ggplot(final_result)
#      + geom_boxplot(aes(x='moderator', y='betweenness', fill='moderator'))
#      + facet_wrap('~ community')
#      + theme(axis_text_x=element_blank()))

# g3.save(filename='../output/moderators_experiment/moderators_experiment_betweenness.png', width=10,height=6, dpi=800)


# g4 = (ggplot(final_result)
#      + geom_boxplot(aes(x='moderator', y='clustering', fill='moderator'))
#      + facet_wrap('~ community')
#      + theme(axis_text_x=element_blank()))

# g4.save(filename='../output/moderators_experiment/moderators_experiment_clustering.png', width=10,height=6, dpi=800)


# g5 = (ggplot(final_result)
#      + geom_boxplot(aes(x='moderator', y='connection_to_majority', fill='moderator'))
#      + facet_wrap('~ community', scales = 'free_y')
#      + theme(axis_text_x=element_blank()))

# g5.save(filename='../output/moderators_experiment/moderators_experiment_connection_to_majority.png', width=12,height=6, dpi=800)

# g6 = (ggplot(final_result)
#      + geom_boxplot(aes(x='moderator', y='connection_to_knowledgeable', fill='moderator'))
#      + facet_wrap('~ community', scales = 'free_y')
#      + theme(axis_text_x=element_blank()))

# g6.save(filename='../output/moderators_experiment/moderators_experiment_connection_to_knowledgeable.png', width=12,height=6, dpi=800)

#community_counts = pd.DataFrame.from_dict(nx.get_node_attributes(H, 'Community'), orient='index', columns = ['Community']).groupby('Community').value_counts()






