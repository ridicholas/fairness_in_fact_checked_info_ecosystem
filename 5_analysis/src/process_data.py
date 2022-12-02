import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import progressbar
import pandas as pd
import checkworthy

# making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#read in files from 4_simulation/output


mitigation_method = 'stop_reading_misinfo'
label_to_use = 'average_truth_perception_random'
sample_method = 'top_avg_origin_degree'

if mitigation_method == 'None':
    run_outfile = 'mitigation-none'
    print('\n\n\n ---  Running with no mitigation --- \n\n\n')
else:
    run_outfile = 'mitigation-' + mitigation_method + '-labelmethod-' + label_to_use + '-sample_method-' + sample_method
    print('\n\n\nRunning with\nmitigation method = ' + mitigation_method + ' &\nlabel method = ' + label_to_use + ' &\nsample method = ' + sample_method)

if not os.path.isdir('../output/' + run_outfile):
    os.makedirs('../output/' + run_outfile)


inpath_community_read_over_time_mit = '../../4_simulation/output/{}/community_read_tweets_by_type.pickle'.format(run_outfile)
inpath_sentiment_mit = '../../4_simulation/output/{}/community_sentiment.pickle'.format(run_outfile)


with open(inpath_sentiment_mit, 'rb') as file:
    community_sentiment = pickle.load(file)

with open(inpath_community_read_over_time_mit, 'rb') as file:
    community_read_over_time = pickle.load(file)





def make_reads_by_time_frame(community_read_over_time):
    
    cols = ['Community', 'Step', 'Topic', 'Type', 'Reads']
    anti_frame = pd.DataFrame(columns = cols)
    noise_frame = pd.DataFrame(columns = cols)
    misinfo_frame = pd.DataFrame(columns = cols)
    
    bar = progressbar.ProgressBar()
    for com in bar(list(community_read_over_time.keys())):
        for step in list(community_read_over_time[com].keys()):
            for topic in list(community_read_over_time[com][step].keys()):
                count_anti = community_read_over_time[com][step][topic]['anti-misinfo']
                count_noise = community_read_over_time[com][step][topic]['noise']
                count_misinfo = community_read_over_time[com][step][topic]['misinfo']
                # append rows
                anti_frame.loc[len(anti_frame)] = [com, step, topic, 'anti-misinfo', count_anti]
                noise_frame.loc[len(noise_frame)] = [com, step, topic, 'noise', count_noise]
                misinfo_frame.loc[len(misinfo_frame)] = [com, step, topic, 'misinfo', count_misinfo]
    return anti_frame, noise_frame, misinfo_frame

                
                
    

def process_community_sentiment(community_sentiment):

    result = pd.DataFrame(columns = ['Community','Topic','Time','Mean Sentiment'])
    for comm in list(community_sentiment.keys()):
        for t in list(community_sentiment[comm].keys()):
            for topic in list(community_sentiment[comm][t].keys()):
                mean_sentiment = np.mean(community_sentiment[comm][t][topic])
                result.loc[len(result)] = [comm,topic,t,mean_sentiment]

    result = result.sort_values(by=['Community','Topic','Time'])
    return result


if not os.path.isdir('../../4_simulation/output/mitigation-none'):
    print('Need to first run with no mitigation!!')


print('\n\n\n ------- Processing Community Sentiment over Time ------- \n\n\n')

clean_community_sentiment = process_community_sentiment(community_sentiment=community_sentiment)
clean_community_sentiment.to_csv('../output/{}/community_sentiment_clean.csv'.format(run_outfile), index = False)


print('\n\n\n ------- Processing Information Read Over Time ------- \n\n\n')

node_by_time_anti, node_by_time_noise, node_by_time_misinfo = make_reads_by_time_frame(community_read_over_time = community_read_over_time)
node_by_time_misinfo.to_pickle('../output/{}/node_by_time_misinfo.pickle'.format(run_outfile))
node_by_time_noise.to_pickle('../output/{}/node_by_time_noise.pickle'.format(run_outfile))
node_by_time_anti.to_pickle('../output/{}/node_by_time_anti.pickle'.format(run_outfile))

