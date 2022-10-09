import pandas as pd
import networkx as nx
import os
import numpy as np
import community as community_louvain
import matplotlib.pyplot as plt

#making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

    #get log degree distribution
    degrees = list(list(zip(*G2.degree))[1])

    #log scale pdf
    plt.clf()
    hist, bins = np.histogram(degrees, bins=10, normed=1)
    bin_centers = (bins[1:]+bins[:-1])*0.5

    #plt.hist(np.log(data['Degree']), bins=10, density=1, edgecolor='black')
    plt.plot(np.log(bin_centers), np.log(hist), color='red')
    plt.title('Twitter Log Scale PDF')
    plt.xlabel('Log(Degree)')
    plt.ylabel('log(Probability)')
    plt.show()

    #log-log rank-frequency
    plt.clf()
    unique, counts = np.unique(degrees, return_counts=True)
    rank_freq = pd.DataFrame({'degree': unique, 'frequency': counts})
    rank_freq.sort_values(by='frequency', ascending=False)
    rank_freq['rank'] = range(1, len(unique)+1)
    plt.scatter(np.log(rank_freq['rank']), np.log(rank_freq['frequency']))
    plt.xlabel('log(rank)')
    plt.ylabel('log(frequency')
    plt.title('Twitter Degree Log-Log Rank-Frequency Plot')
    plt.show()



    return None


def export_community_net(G, outpath):

    communities = nx.get_node_attributes(G, 'Community')
    ig = community_louvain.induced_graph(communities, G, weight = 'WEIGHT')
    communities = pd.DataFrame.from_dict(nx.get_node_attributes(G, 'Community'), orient='index')\
        .rename(columns={0:'com'})
    community_size = communities.groupby(['com']).size().sort_values(ascending=False).to_dict()
    nx.set_node_attributes(ig, community_size, 'SIZE')
    #remove self edges
    #ig.remove_edges_from(nx.selfloop_edges(ig))

    nx.write_gexf(ig, outpath)
    return None



    
    



path = '../../data/nodes_with_community.gpickle'
net_outpath = '../output/subset_net.gexf'
com_outpath = '../output/community_net.gexf'

G = nx.read_gpickle(path)
subset_graph(G, net_outpath, communities=[3, 56, 43])
export_community_net(G, com_outpath)
