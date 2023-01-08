import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from plotnine import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

runtime = 200
impactednesses = [{3: 0.5, 56: 0.5, 43: 0.5},
                  {3: 0.8, 56: 0.3, 43: 0.3},
                  {3: 0.3, 56: 0.8, 43: 0.3},
                  {3: 0.3, 56: 0.3, 43: 0.8}]

misinfo_none = pd.read_pickle('../output/mitigation-none/node_by_time_misinfo.pickle')
anti_none = pd.read_pickle('../output/mitigation-none/node_by_time_anti.pickle')
noise_none = pd.read_pickle('../output/mitigation-none/node_by_time_noise.pickle')
sentiment_none = pd.read_csv('../output/mitigation-none/community_sentiment_clean.csv')
misinfo_none['Mitigation'] = 'None'
anti_none['Mitigation'] = 'None'
noise_none['Mitigation'] = 'None'

reads = {'random' : {}, 'stratified': {}, 'knowledgable_community': {}}
checks = {'random' : {}, 'stratified': {}, 'knowledgable_community': {}}
for strat in reads.keys():

    file = 'mitigation-stop_reading_misinfo-labelmethod-average_truth_perception_{}-sample_method-top_avg_origin_degree'.format(strat)
    reads[strat]['misinfo'] = pd.read_pickle('../output/{}/node_by_time_misinfo.pickle'.format(file))
    reads[strat]['anti'] = pd.read_pickle('../output/{}/node_by_time_anti.pickle'.format(file))
    reads[strat]['noise'] = pd.read_pickle('../output/{}/node_by_time_noise.pickle'.format(file))
    reads[strat]['sentiment'] = pd.read_csv('../output/{}/community_sentiment_clean.csv'.format(file))
    reads[strat]['misinfo']['Mitigation'] = 'Stop Reading'
    reads[strat]['anti']['Mitigation'] = 'Stop Reading'
    reads[strat]['noise']['Mitigation'] = 'Stop Reading'
    checks[strat]['misinfo'] = pd.read_pickle('../output/{}/checked_by_time_misinfo.pickle'.format(file))
    checks[strat]['anti'] = pd.read_pickle('../output/{}/checked_by_time_anti.pickle'.format(file))
    checks[strat]['noise'] = pd.read_pickle('../output/{}/checked_by_time_noise.pickle'.format(file))



    






comms = [3, 43, 56]
topics = [0,1,2,3]
misinfo_read = pd.DataFrame(index=comms, columns=reads.keys())
misinfo_caught_percent = pd.DataFrame(index=comms, columns=reads.keys())
tpfcr = pd.DataFrame(index=comms, columns = reads.keys())
belief_change_weighted_percent = pd.DataFrame(index=comms, columns = reads.keys())
sum_misinfo_checked = pd.DataFrame(index=comms, columns = reads.keys())

sum_misinfo_none = misinfo_none[misinfo_none['Step'] >= runtime].groupby(by=['Community']).agg({'Reads': 'sum'})
sum_misinfos = {'random' : {}, 'stratified': {}, 'knowledgable_community': {}}
sum_checks = {'random' : {}, 'stratified': {}, 'knowledgable_community': {}}

for strat in reads.keys():

    #inefficiently weight beliefs
    for i in range(len(reads[strat]['sentiment'])):
        


    sum_misinfos[strat] = reads[strat]['misinfo'][reads[strat]['misinfo']['Step'] >= runtime].groupby(by=['Community']).agg({'Reads': 'sum'})
    sum_checks[strat] = checks[strat]['misinfo'].groupby(by=['Community']).agg({'Reads': 'sum'}) + checks[strat]['anti'].groupby(by='Community').agg({'Reads': 'sum'}) + checks[strat]['noise'].groupby(by='Community').agg({'Reads': 'sum'})

    misinfo_read['{}'.format(strat)] = (sum_misinfos[strat])
    sum_misinfo_checked['{}'.format(strat)] = checks[strat]['misinfo'].groupby(by=['Community']).agg({'Reads': 'sum'})
    misinfo_caught_percent['{}'.format(strat)] = (sum_misinfos[strat] - sum_misinfo_none) / sum_misinfo_none
    tpfcr['{}'.format(strat)] = checks[strat]['misinfo'].groupby(by=['Community']).agg({'Reads': 'sum'}) / sum_checks[strat]
    belief_change_weighted_percent['{}'.format(strat)] = ((reads[strat]['sentiment'][reads[strat]['sentiment']['Time'] == 399].groupby(['Community', 'Topic']).agg({'Mean Sentiment': 'mean'})) - sentiment_none[sentiment_none['Time'] == 399].groupby('Community').agg({'Mean Sentiment': 'mean'})) / sentiment_none[sentiment_none['Time'] == 399].groupby('Community').agg({'Mean Sentiment': 'mean'})
    

#print('Misinformation Read: \n' , misinfo_read, '\n\n')
print('Percent Diff Misinformation Read w/ Mitigation vs. Without Mitigation: \n' ,misinfo_caught_percent, '\n\n')
#print('Sum Misinformation Checked: \n' , sum_misinfo_checked, '\n\n')
print('True Positive Fact Check Rate: \n' ,tpfcr, '\n\n')
print('Weighted Percent Change in Belief: \n' ,belief_change_weighted_percent, '\n\n')

