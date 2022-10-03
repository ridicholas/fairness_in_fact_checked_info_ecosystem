import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community
import pickle
import os.path



def create_network(path):
    print('reading in network...')
    G = nx.read_edgelist(path)
    #identify communities
    print('checking if communities file exists...')
    if os.path.isfile('communities.pkl'):
        print('found communities file, loading communities')
        with open("communities.pkl", "rb") as f:
            communities = pickle.load(f)
    else:
        print('communities file not found, using lovain to generate communities...')
        communities = community.louvain_communities(G)
        with open("communities.pkl", "wb") as f:
            pickle.dump(np.array(communities), f)
    
    #assign communities to node attributes
    community_label = 1
    for community in communities:
        for node in community:
            G.nodes[node]['Community'] = community_label
        
        community_label += 1

    return G

