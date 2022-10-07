#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 01:46:45 2022

@author: tdn897
"""

import pandas as pd
import networkx as nx
import os
import numpy as np
import community as community_louvain

#making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__))) 

## Function takes a graph partitioned by communities and returns number of edges between communities

def calc_edges_between_communities(G, outpath):
    communities = nx.get_node_attributes(G, 'Community')
    ig = community_louvain.induced_graph(communities, G, weight = 'WEIGHT')
    c1 = []
    c2 = []
    weight = []
    for u,v,a in ig.edges(data=True):
        if u != v: 
           c1.append(u)
           c2.append(v)
           weight.append(a['WEIGHT'])
    edges_between_communities = pd.DataFrame()
    edges_between_communities['community1'] = c1
    edges_between_communities['community2'] = c2
    edges_between_communities['edges'] = weight
    edges_between_communities.to_csv(outpath)
    print('\n\nEdges Between Communities Calculated...\n\n')
    return ig


## Function takes a graph partitioned by communities and returns stats such as 
## max, mean, median degree and community size

def calc_stats_for_communities(G, community_network, outpath):
    communities = pd.DataFrame.from_dict(nx.get_node_attributes(G, 'Community'), orient='index')\
        .rename(columns={0:'com'})
    
    community_size = communities.groupby(['com']).size().sort_values(ascending=False)
    print('\n\nCommunity Size Calculated...\n\n')
    t = list(G.degree)
    degrees = [item[1] for item in t]
    communities['degree'] = degrees
    degree_stat = communities.groupby('com').agg(max_degree=('degree', 'max'),
                                                mean_degree=('degree', 'mean'),
                                                median_degree=('degree', 'median'))
    stats = degree_stat.merge(community_size.rename('nodes'), left_index=True, right_index=True)

    btwn_c = nx.betweenness_centrality(community_network, weight='WEIGHT')
    stats = stats.merge(pd.DataFrame.from_dict(btwn_c, orient='index'), left_index=True, right_index=True).rename(columns={0:'betweenness_centrality'})
    stats.to_csv(outpath, index=True)
    print('\n\nDegree Stats Calculated ... \n\n\n')
    return None

def calc_stats_for_network(G, outpath):
    stats = pd.DataFrame(columns=['Avg. Degree', 'Density', 'Diameter', 'Clustering Coefficient', 'Avg. Shortest Path Length'])
    stats.loc[0, 'Avg. Degree'] = np.array([*nx.average_degree_connectivity(G).values()]).mean()
    stats.loc[0, 'Density'] = nx.density(G)
    try:
        stats.loc[0, 'Diameter'] = nx.diameter(G)
    except Exception:
        stats.loc[0, 'Diameter'] = str(Exception)
    stats.loc[0, 'Clustering Coefficient'] = nx.average_clustering(G)
    try:
        stats.loc[0, 'Avg. Shortest Path Length'] = nx.average_shortest_path_length(G)
    except Exception:
        stats.loc[0, 'Avg. Shortest Path Length'] = str(Exception)
    
    print('Here are yo network stats:')
    print(stats)
    stats.to_csv(outpath, index=False)
    return None

    

path = '../../data/nodes_with_community.gpickle'
stats_outpath = '../output/community_stats.csv'
edges_outpath = '../output/edges_between_communities.csv'
network_stats_outpath = '../output/network_stats.csv'

G = nx.read_gpickle(path)
ig = calc_edges_between_communities(G, outpath=edges_outpath)
calc_stats_for_communities(G, ig, outpath=stats_outpath)

#calc_stats_for_network(G, outpath=network_stats_outpath)

