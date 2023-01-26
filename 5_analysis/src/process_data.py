

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import progressbar
import time
import community as community_louvain
import gc
# making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def make_comm_string(communities):
        comm_string = ''
        for comm in communities:
            comm_string += '_' + str(comm)

        return comm_string

def induce_graph(G):
    communities = nx.get_node_attributes(G, 'Community')
    ig = community_louvain.induced_graph(communities, G, weight = 'WEIGHT')
    inv_weights = {}
    
    for u,v,a in ig.edges(data=True):
        inv_weights[(u,v)] = 1/a['WEIGHT']
    nx.set_edge_attributes(ig, values=inv_weights, name='InvWeight')
    
    
    return ig

def get_network_structure_stats(start_network_path):
    G = nx.read_gpickle(start_network_path)
    ig = induce_graph(G)
    communities = list(set(nx.get_node_attributes(G, 'Community').values()))
    nodes_in_community = {comm: [] for comm in communities}
    
    
    for node, comm in nx.get_node_attributes(G, 'Community').items():
        nodes_in_community[comm].append(node)

    stats = {comm: {} for comm in communities}
    unique_counts = np.unique(list(nx.get_node_attributes(G, 'Community').values()), return_counts=True)
    largest = 0
    total = len(G.nodes())

    for i in range(len(unique_counts[0])):
        comm = unique_counts[0][i]
        size = unique_counts[1][i]
        stats[comm]['size'] = size
        if size > largest:
            largest = size
            largest_community = comm
        stats[comm]['perc_of_network'] =  size/total
        stats[comm]['cluster_coef'] = round(nx.average_clustering(G, nodes=nodes_in_community[comm]),4)
        stats[comm]['avg_degree_graph'] = np.array(list(nx.get_node_attributes(G.subgraph(nodes_in_community[comm]), 'degree').values())).mean() 
        stats[comm]['avg_degree_within_community'] = np.array(list(dict(G.subgraph(nodes_in_community[comm]).in_degree()).values())).mean()
        stats[comm]['density'] = nx.density(G.subgraph(nodes_in_community[comm]))
        stats[comm]['average_centrality_of_nodes'] = np.array(list(nx.get_node_attributes(G.subgraph(nodes_in_community[comm]), 'centrality').values())).mean() 
        stats[comm]['comm_centrality'] = nx.betweenness_centrality(ig, weight='InvWeight')[comm] #this is the incuded subgraph estimated version
        #stats[comm]['comm_centrality'] = nx.group_betweenness_centrality(G, nodes_in_community[comm]) #this is the very slow version
    
    #relative measures
    weights = nx.get_edge_attributes(ig, 'WEIGHT')
    for comm in communities:
        stats[comm]['perc_of_largest'] = stats[comm]['size']/largest
        
        try:
            stats[comm]['ratio_connections_self_to_largest_self'] = weights[(comm, largest_community)]/weights[(comm,comm)]
            stats[comm]['ratio_connections_self_to_largest_largest'] = weights[(comm, largest_community)]/weights[(largest_community,largest_community)]
        except:
            try:
                stats[comm]['ratio_connections_self_to_largest_self'] = weights[(largest_community, comm)]/weights[(comm,comm)]
                stats[comm]['ratio_connections_self_to_largest_largest'] = weights[(largest_community, comm)]/weights[(largest_community,largest_community)]
            except:
                stats[comm]['ratio_connections_self_to_largest_self'] = 0
                stats[comm]['ratio_connections_self_to_largest_largest'] = 0
           
        
    return stats, communities
    
    

def make_results_community_level(infile, reps, paths, communities, comm_list, community_features=False):


    import TopicSim
    import checkworthy
    import gc

    
    regression = pd.DataFrame(columns=['run', 'mod', 'community', 'topic', 'size', 'perc_of_net', 'avg_degree_within_graph', 'avg_degree_within_community', 
                                       'density', 'cluster_coef', 'average_centrality_of_nodes', 'comm_centrality', 'impactedness', 'start_belief', 
                                       'perc_of_largest', 'ratio_connections_self_to_largest_self', 'ratio_connections_self_to_largest_largest', 
                                       'misinfo_read', 'anti_misinfo_read', 'noise_read', 'misinfo_checked', 'anti_misinfo_checked', 
                                       'noise_checked', 'change_in_belief'])
    gc.enable()
    i=0

    reads = {}
    checks = {}
    beliefs = {}


    for rep in range(reps):

        start_path = '../../4_simulation/output/simulation_net_run{}_communities{}.gpickle'.format(rep, communities)
        if community_features:
            stats, comm_list = get_network_structure_stats(start_path)

        print('\n\n\n\n\n Processing Reads Data - Repetition #' + str(rep) + '------ \n\n\n\n')

        reads[rep] = {}
        checks[rep] = {}
        beliefs[rep] = {}

        for path in paths:

            mod = path.replace('../../4_simulation/output/simulation_', '')

            reads[rep][mod] = {}
            checks[rep][mod] = {}
            beliefs[rep][mod] = {}



            
            with open(path + 'run' + str(rep) + '_communities' + comms + '.pickle', 'rb') as file:
                sim = pickle.load(file)
            
            

            community_read_over_time = sim.community_read_tweets_by_type
            community_sentiment_through_time = sim.community_sentiment_through_time
            community_checked_over_time = sim.community_checked_tweets_by_type

            #rearrange dictionaries
            for comm in comm_list:
                keys = list(range(sim.num_topics))
                reads[rep][mod][comm] = {key:{k:community_read_over_time[comm][k][key] for k in community_read_over_time[comm] if key in community_read_over_time[comm][k]} for key in keys}
                for topic in keys:
                    new_keys = ['misinfo','noise','anti-misinfo']
                    reads[rep][mod][comm][topic] = {key:{k:reads[rep][mod][comm][topic][k][key] for k in reads[rep][mod][comm][topic] if key in reads[rep][mod][comm][topic][k]} for key in new_keys}
            
            
            for comm in comm_list:
                keys = list(range(sim.num_topics))
                checks[rep][mod][comm] = {key:{k:community_checked_over_time[comm][k][key] for k in community_checked_over_time[comm] if key in community_checked_over_time[comm][k]} for key in keys}
                for topic in keys:
                    new_keys = ['misinfo','noise','anti-misinfo']
                    checks[rep][mod][comm][topic] = {key:{k:checks[rep][mod][comm][topic][k][key] for k in checks[rep][mod][comm][topic] if key in checks[rep][mod][comm][topic][k]} for key in new_keys}
                
            
            for comm in comm_list:
                keys = list(range(sim.num_topics))
                beliefs[rep][mod][comm] = {key:{k:community_sentiment_through_time[comm][k][key] for k in community_sentiment_through_time[comm] if key in community_sentiment_through_time[comm][k]} for key in keys}
                for topic in keys:
                    for step in beliefs[rep][mod][comm][topic].keys():
                        beliefs[rep][mod][comm][topic][step] = np.array(beliefs[rep][mod][comm][topic][step]).mean()
            
            

            
            topics = list(range(sim.num_topics))
            impactedness = sim.impactedness
            start_beliefs = sim.beliefs

            del sim
            gc.collect()
            if community_features:
                for community in comm_list:
                    for topic in topics:
                        regression.loc[i, ['run', 
                                           'mod', 
                                           'community', 
                                           'topic', 
                                           'impactedness', 
                                           'start_belief',
                                           'change_in_belief',
                                           'misinfo_read',
                                           'anti_misinfo_read',
                                           'noise_read',
                                           'misinfo_checked',
                                           'anti_misinfo_checked',
                                           'noise_checked', 
                                           'size', 
                                           'perc_of_net', 
                                           'avg_degree_within_graph', 
                                           'avg_degree_within_community', 
                                           'density', 
                                           'cluster_coef', 
                                           'average_centrality_of_nodes', 
                                           'comm_centrality', 
                                           'perc_of_largest', 
                                           'ratio_connections_self_to_largest_self', 
                                           'ratio_connections_self_to_largest_largest']] = [rep,
                                                               mod, 
                                                               community, 
                                                               topic, 
                                                               impactedness[topic][community], 
                                                               start_beliefs[topic][community],
                                                               beliefs[rep][mod][community][topic][299] - beliefs[rep][mod][community][topic][100],
                                                               sum(reads[rep][mod][community][topic]['misinfo'].values()),
                                                               sum(reads[rep][mod][community][topic]['anti-misinfo'].values()),
                                                               sum(reads[rep][mod][community][topic]['noise'].values()),
                                                               sum(checks[rep][mod][community][topic]['misinfo'].values()),
                                                               sum(checks[rep][mod][community][topic]['anti-misinfo'].values()),
                                                               sum(checks[rep][mod][community][topic]['noise'].values()), 
                                                               stats[community]['size'], 
                                                               stats[community]['perc_of_network'],
                                                               stats[community]['avg_degree_graph'],
                                                               stats[community]['avg_degree_within_community'],
                                                               stats[community]['density'],
                                                               stats[community]['cluster_coef'],
                                                               stats[community]['average_centrality_of_nodes'],
                                                               stats[community]['comm_centrality'],
                                                               stats[community]['perc_of_largest'],
                                                               stats[community]['ratio_connections_self_to_largest_self'],
                                                               stats[community]['ratio_connections_self_to_largest_largest']]
                        
                        
                                                                              
    
                        i+=1
            
            

    
    beliefs_frame = pd.DataFrame.from_dict({(i,j,k,y,z): beliefs[i][j][k][y][z]
                           for i in beliefs.keys() 
                           for j in beliefs[i].keys() for k in beliefs[i][j].keys() for y in beliefs[i][j][k].keys() for z in beliefs[i][j][k][y].keys()},
                       orient='index')
    beliefs_frame[['Rep', 'Mod', 'Community', 'Topic', 'Time']] = beliefs_frame.index.to_list()
    beliefs_frame.rename(columns={0: 'belief'}, inplace=True)

    reads_frame = pd.DataFrame.from_dict({(i,j,k,y,z,x): reads[i][j][k][y][z][x]
                           for i in reads.keys() 
                           for j in reads[i].keys() for k in reads[i][j].keys() for y in reads[i][j][k].keys() for z in reads[i][j][k][y].keys() for x in reads[i][j][k][y][z].keys()},
                       orient='index')
    reads_frame[['Rep', 'Mod', 'Community', 'Topic', 'Info_Type', 'Time']] = reads_frame.index.to_list()
    reads_frame.rename(columns={0:'reads'}, inplace=True)
    
    checks_frame = pd.DataFrame.from_dict({(i,j,k,y,z,x): checks[i][j][k][y][z][x]
                           for i in checks.keys() 
                           for j in checks[i].keys() for k in reads[i][j].keys() for y in checks[i][j][k].keys() for z in checks[i][j][k][y].keys() for x in checks[i][j][k][y][z].keys()},
                       orient='index')
    checks_frame[['Rep', 'Mod', 'Community', 'Topic', 'Info_Type', 'Time']] = checks_frame.index.to_list()
    checks_frame.rename(columns={0:'checks'}, inplace=True)


    

    if community_features:
        return reads_frame, beliefs_frame, checks_frame, reads, beliefs, checks, regression
    else:
        return reads_frame, beliefs_frame, checks_frame, reads, beliefs, checks






def process_individual_level_data(reps, modules, labels, sampling, comm_string):
   

    import gc
    gc.enable()
    
    regression_data = []

    for rep in range(reps):
        
        print('\n\n\n Rep ' + str(rep) + '\n\n\n\n')
        # Midpoint data
        
        pre_path = '../../4_simulation/output/simulation_pre_period_run{}_communities{}.pickle'.format(rep, comm_string)
        
        with open(pre_path, 'rb') as file:
            sim = pickle.load(file)
            
        G = sim.G
        
        print('Calculating Clustering Coefficient for whole graph')
        cluster = nx.clustering(G)
        print('Finished Calculating Clustering coefficient!')
        
        node_dict = {node: {'topic_' + str(topic): {'midpoint_belief':data['sentiment'][topic]} for topic in list(data['sentiment'].keys())} for node, data in G.nodes(data=True)}
        
        for node, data in G.nodes(data=True):
            node_dict[node].update({'betweenness_centrality':data['centrality'], 
                                    'community':data['Community'], 
                                    'rep':rep, 
                                    'kind':data['kind'],
                                    'out_degree':G.out_degree[node],
                                    'clustering':cluster[node]})
            
            # -- 1. collect average per topic belief of nodes followed
            # -- 2. collect number of bots followed
            eg = nx.ego_graph(G, node)
            num_bots_followed = 0
            num_in_other_communities_followed = 0
            for n, d in eg.nodes(data=True):
                if d['kind'] == 'bot':
                    num_bots_followed += 1
                if d['Community'] != data['Community']:
                    num_in_other_communities_followed += 1
                for topic in list(data['sentiment'].keys()):
                    avg_belief = []
                    if n == node:
                        avg_belief.append(0.5)
                    else:
                        avg_belief.append(d['sentiment'][topic]) 
                    node_dict[node]['topic_' + str(topic)].update({'average_external_belief': np.mean(avg_belief)})
            node_dict[node]['number_bots_followed'] = num_bots_followed
            node_dict[node]['number_followed_in_other_comms'] = num_in_other_communities_followed


                        
        del G
        del sim
        gc.collect()
        
        # Post data - No intervention
        print('No Intervention')
        
        p = '../../4_simulation/output/simulation_final_no_intervention_run{}_communities{}.pickle'.format(rep, comm_string)
        with open(p, 'rb') as file:
            sim = pickle.load(file)
        
        G = sim.G
        for node, data in G.nodes(data=True):
            for topic in list(data['sentiment'].keys()):
                node_dict[node]['topic_' + str(topic)].update({'impactedness': data['impactedness'][topic]})
                node_dict[node]['topic_' + str(topic)]['intervention'] = {}
                node_dict[node]['topic_' + str(topic)]['intervention'].update({'no_intervention_change_in_belief': data['sentiment'][topic] - node_dict[node]['topic_'+str(topic)]['midpoint_belief']})
        del G
        del sim
        gc.collect()
 
        
        # Post data - All interventions
        
        for mod in modules:
            for l in labels:
                for s in sampling:
                    p = '../../4_simulation/output/simulation_' + mod + '_' + l + '_' + s + '_' + 'run' + str(rep) + '_communities' + comm_string + '.pickle'
                    if os.path.isfile(p):
                        print('Intervention: ' + mod + '_' + l + '_' + s)
                        with open(p, 'rb') as file:
                            sim = pickle.load(file)
                        G = sim.G
                        for node, data in G.nodes(data=True):
                            for topic in list(data['sentiment'].keys()):
                                node_dict[node]['topic_' + str(topic)]['intervention'].update({mod + '_' + l + '_' + s:data['sentiment'][topic] - node_dict[node]['topic_'+str(topic)]['midpoint_belief']})
                        del G
                        del sim
                        gc.collect()
                        

        for node in list(node_dict.keys()):
            for topic in list(node_dict[node].keys()):
                if 'topic' in topic:
                    for intervention in list(node_dict[node][topic]['intervention'].keys()):
                        regression_data.append([rep, 
                                                node, 
                                                topic,
                                                intervention,
                                                node_dict[node]['community'],
                                                node_dict[node]['kind'],
                                                node_dict[node]['out_degree'],
                                                node_dict[node]['number_bots_followed'],
                                                node_dict[node]['betweenness_centrality'],
                                                node_dict[node]['clustering'],
                                                node_dict[node]['number_followed_in_other_comms'],
                                                node_dict[node][topic]['impactedness'],
                                                node_dict[node][topic]['average_external_belief'],
                                                node_dict[node][topic]['midpoint_belief'],
                                                node_dict[node][topic]['intervention'][intervention]
                                                ])
                        
        
    results = pd.DataFrame(regression_data,
                           columns = ['Rep', 
                                        'Node', 
                                        'Topic', 
                                        'Intervention', 
                                        'Community', 
                                        'Kind',
                                        'Out Degree',
                                        'Number of Bots Followed',
                                        'Betweenness Centrality',
                                        'Clustering',
                                        'Number Followed in Other Comms',
                                        'Impactedness',
                                        'Average External Belief', 
                                        'Midpoint Belief', 
                                        'Change in Belief'])
    
    results = results.loc[(results['Kind'] != 'bot')&(results['Out Degree'] > 0)]
    results.drop(columns = ['Kind', 'Out Degree'], inplace=True)   
    return results
               
                        
                        
                                    

                        
modules = ['TopPredicted', 'TopPredictedByTopic']
labels = ['random', 'knowledgable_community', 'stratified']
sampling = ['nodes_visited', 'stratified_nodes_visited']
reps = 10
comms = [43, 56, 127]
comm_string = make_comm_string(comms)


     
results = process_individual_level_data(reps = reps, 
                                        modules = modules, 
                                        labels = labels, 
                                        sampling = sampling, 
                                        comm_string = comm_string)


            
                    
results.to_csv('../output/individual_level_regression_data.csv', index = False)
                    
        
        
            
            
            
        
 

        






print('\n\n\n ------- Processing Community Sentiment over Time ------- \n\n\n')


modules = ['TopPredicted', 'TopPredictedByTopic']
label_methods = ['random', 'stratified', 'knowledgable_community']
sample_methods = ['nodes_visited', 'stratified_nodes_visited']
infile = '../../4_simulation/output/simulation_'
comms = '_43_56_127' #manually input this for now, we can write an automated finder later
comm_list = [43, 56, 127]
start_network_path = '../../4_simulation/output/simulation_net_communities_' + comms + '.gpickle'
regression_outfile = '../output/regression' +  comms
readFrame_outfile = '../output/misinfo_reads' + comms
beliefFrame_outfile = '../output/belief_results' + comms
reps = 10

paths = []

paths.append('../../4_simulation/output/simulation_final_no_intervention_')
for mod in modules:
    for l in label_methods:
        for s in sample_methods:
            p = '../../4_simulation/output/simulation_' + mod + '_' + l + '_' + s + '_' 
            paths.append(p)



reads_frame, beliefs_frame, checks_frame, reads, beliefs, checks = make_results_community_level(infile=infile, 
                                                                                                reps = reps,
                                                                                                paths = paths,
                                                                                                communities=comms,
                                                                                                comm_list=comm_list)

reads_frame.to_csv(readFrame_outfile + '.csv', index=False)
beliefs_frame.to_csv(beliefFrame_outfile + '.csv', index=False)

