import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import progressbar
import pandas as pd
# making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#read in files from 4_simulation/output

G = nx.read_gpickle("../../4_simulation/output/simulation_net.gpickle")

inpath_info = '../../4_simulation/output/all_info.pickle'
inpath_node_info = '../../4_simulation/output/node_info.pickle'
inpath_node_time_info = '../../4_simulation/output/node_time_info.pickle'
inpath_sentiment = '../../4_simulation/output/community_sentiment.pickle'
runtime = 500

with open(inpath_info, 'rb') as file:
    all_info = pickle.load(file)

with open(inpath_node_info, 'rb') as file:
    node_read_tweets = pickle.load(file)

with open(inpath_node_time_info, 'rb') as file:
    node_read_tweets_by_time = pickle.load(file)

with open(inpath_sentiment, 'rb') as file:
    community_sentiment = pickle.load(file)

def make_node_by_time_frame(G, all_info, node_read_tweets_by_time, steps, topic_filter=None):
    '''
    Create 3 dataframes (misinfo, noise, anti). For each: index is node, first col is 'community' and all following columns are time steps
    Each node/timestep value for each frame is count of claims read by the node at that timestep.
    For example: misinfo_frame[1, 10] would return the number of misinfo claims read by node 1 at timestep 10.
    '''

    #misinfo/anti/noise read over time

    misinfo_frame = pd.DataFrame(0, index=list(node_read_tweets_by_time.keys()), columns=['Community'] + list(range(steps)))
    anti_frame = misinfo_frame.copy()
    noise_frame = misinfo_frame.copy()

    bar = progressbar.ProgressBar()
    for node in bar(node_read_tweets_by_time.keys()):
        #add node community to the frame
        misinfo_frame.loc[node, 'Community'] = G.nodes(data=True)[node]['Community']
        anti_frame.loc[node, 'Community'] = G.nodes(data=True)[node]['Community']
        noise_frame.loc[node, 'Community'] = G.nodes(data=True)[node]['Community']

        for step in range(steps):
            for claim in node_read_tweets_by_time[node][step]:
                #look up claim properties in all_info and add it to appropriate count
                topic = claim[0]
                if topic==topic_filter:
                    value = all_info[claim]['value']
                    if value == -1:
                        anti_frame.loc[node, step] += 1
                    elif value == 0:
                        noise_frame.loc[node, step] += 1
                    else:
                        misinfo_frame.loc[node, step] += 1




    return anti_frame, noise_frame, misinfo_frame

def make_claim_by_time_frame(G, all_info, node_read_tweets_by_time, steps, community):
    '''
    Create dataframe for provided community in simulation. For each: index is claim, all following columns are time steps
    Each claim/timestep value for each frame is count of nodes that read the claim at that timestep.
    For example: community3_frame[1, 10] would return the number of nodes in community 3 that read claim 1 at timestep 10.
    '''

    #make a frame for each community (the 3 frames can be summed up to make the entire )


    frame = pd.DataFrame(0, index = list(all_info.keys()), columns = list(range(steps)))



    bar = progressbar.ProgressBar()
    for node in bar(node_read_tweets_by_time.keys()):

        if G.nodes(data=True)[node]['Community'] == community:
            for step in range(steps):
                for claim in node_read_tweets_by_time[node][step]:
                    #look up community and put it in appropriate table

                    frame.loc[claim, step] += 1




    return frame


def process_community_sentiment(community_sentiment):
    
    result = pd.DataFrame(columns = ['Community','Topic','Time','Mean Sentiment'])
    for comm in list(community_sentiment.keys()):
        for t in list(community_sentiment[comm].keys()):
            for topic in list(community_sentiment[comm][t].keys()):
                mean_sentiment = np.mean(community_sentiment[comm][t][topic])
                result.loc[len(result)] = [comm,topic,t,mean_sentiment]
                
    result = result.sort_values(by=['Community','Topic','Time'])
    return result
        
def mean_value_of_info_produced(all_info):      
        
    raw = pd.DataFrame.from_dict(all_info, orient='index')
    result = raw.groupby(['time-origin','topic']).agg(mean_value=pd.NamedAgg(column='value',aggfunc='mean')).sort_values(by=['topic','time-origin'])
    return result





topics = ['0', '1', '2', '3']
for topic in topics:
    node_by_time_anti, node_by_time_noise, node_by_time_misinfo = make_node_by_time_frame(G, all_info, node_read_tweets_by_time, steps=runtime, topic_filter=topic)
    node_by_time_misinfo.to_pickle('../output/topic{}_node_by_time_misinfo.pickle'.format(topic))
    node_by_time_noise.to_pickle('../output/topic{}_node_by_time_noise.pickle'.format(topic))
    node_by_time_anti.to_pickle('../output/topic{}_node_by_time_anti.pickle'.format(topic))
    del node_by_time_anti
    del node_by_time_noise
    del node_by_time_misinfo


clean_community_sentiment = process_community_sentiment(community_sentiment=community_sentiment)
clean_community_sentiment.to_csv('../output/community_sentiment_clean.csv', index = False)

node_by_time_anti, node_by_time_noise, node_by_time_misinfo = make_node_by_time_frame(G, all_info, node_read_tweets_by_time, steps =runtime)
node_by_time_misinfo.to_pickle('../output/node_by_time_misinfo.pickle'.format(topic))
node_by_time_noise.to_pickle('../output/node_by_time_noise.pickle'.format(topic))
node_by_time_anti.to_pickle('../output/node_by_time_anti.pickle'.format(topic))

#claim_by_time_frame3 = make_claim_by_time_frame(G, all_info, node_read_tweets_by_time, 1000, community=3)
#claim_by_time_frame3.to_pickle('../output/claim_by_time_community3.pickle')
#del claim_by_time_frame3


#claim_by_time_frame56 = make_claim_by_time_frame(G, all_info, node_read_tweets_by_time, 1000, community=56)
#claim_by_time_frame56.to_pickle('../output/claim_by_time_community56.pickle')

#del claim_by_time_frame56

#claim_by_time_frame43 = make_claim_by_time_frame(G, all_info, node_read_tweets_by_time, 1000, community=43)
#claim_by_time_frame43.to_pickle('../output/claim_by_time_community43.pickle')
