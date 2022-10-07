import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community as comm
import pickle
import os.path


#making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__))) 

def create_network(path):
    print('reading in network...')
    G = nx.read_edgelist(path, create_using=nx.DiGraph)
    #identify communities
    print('checking if communities file exists...')
    if os.path.isfile('../../data/communities.pkl'):
        print('found communities file, loading communities')
        with open("../../data/communities.pkl", "rb") as f:
            communities = pickle.load(f)
    else:
        print('communities file not found, using lovain to generate communities...')
        communities = comm.louvain_communities(G)
        with open("../../data/communities.pkl", "wb") as f:
            pickle.dump(np.array(communities), f)
    
    #assign communities to node attributes
    community_label = 1
    for community in communities:
        for node in community:
            G.nodes[node]['Community'] = community_label
        
        community_label += 1

    nx.write_gpickle(G, "../../data/nodes_with_community.gpickle")
    nx.write_gexf(G, "../../data/nodes_with_community.gexf")
    return None

