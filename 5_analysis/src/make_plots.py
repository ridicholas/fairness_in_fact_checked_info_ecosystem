import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import progressbar
import fastparquet

from plotnine import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

communities_to_subset = [3, 56, 43]
runtime = 500
inpath_community_sentiment = '../output/community_sentiment_clean.csv'


def plot_total_reads_over_time(by_community=False,specific_topic=None):
    
    colors = {'anti-misinfo':'blue', 'noise':'black', 'misinfo':'darkred'}  

    
    if specific_topic != None:
        misinfo = pd.read_pickle('../output/topic{}_node_by_time_misinfo.pickle'.format(specific_topic))
        anti = pd.read_pickle('../output/topic{}_node_by_time_anti.pickle'.format(specific_topic))
        noise = pd.read_pickle('../output/topic{}_node_by_time_noise.pickle'.format(specific_topic))
    else:
        misinfo = pd.read_pickle('../output/node_by_time_misinfo.pickle')
        anti = pd.read_pickle('../output/node_by_time_anti.pickle')
        noise = pd.read_pickle('../output/node_by_time_noise.pickle')
        
    time = list(range(runtime)) + list(range(runtime)) + list(range(runtime))
    labels = ['misinfo' for i in range(runtime)] + ['anti-misinfo' for i in range(runtime)] + ['noise' for i in range(runtime)]
    misinfo_reads = misinfo.iloc[:, 1:].sum(axis=0)
    anti_misinfo_reads = anti.iloc[:, 1:].sum(axis=0)
    noise_reads = noise.iloc[:, 1:].sum(axis=0)
    plot_frame = pd.concat([misinfo_reads,anti_misinfo_reads,noise_reads]).to_frame()
    plot_frame['time'] = time
    plot_frame['label'] = labels
    plot_frame = plot_frame.rename(columns={0:'reads'})
    
    
    if specific_topic:
        title = 'Total Topic {} Information Read over Time'.format(specific_topic)

    else:
        title = 'Total Information Read over Time'

    
    plt=(ggplot(plot_frame,
                aes(x='time',y='reads', color = 'label'))
         + geom_line()
         + ggtitle(title)
         + scale_color_manual(values=colors))

    plt.save(filename='../output/total_topic_{}.png'.format(specific_topic), width=8,height=6)
    
    
    if by_community:
        community_plot_frame = pd.DataFrame()
        for community in communities_to_subset:
            time = list(range(runtime)) + list(range(runtime)) + list(range(runtime))
            labels = ['misinfo' for i in range(runtime)] + ['anti-misinfo' for i in range(runtime)] + ['noise' for i in range(runtime)]
            misinfo_reads = misinfo[misinfo['Community']==community].iloc[:, 1:].sum(axis=0)
            anti_misinfo_reads = anti[anti['Community']==community].iloc[:, 1:].sum(axis=0)
            noise_reads = noise[noise['Community']==community].iloc[:, 1:].sum(axis=0)
            plot_frame = pd.concat([misinfo_reads,anti_misinfo_reads,noise_reads]).to_frame()
            plot_frame['time'] = time
            plot_frame['label'] = labels
            plot_frame['community'] = 'Community ' + str(community)
            plot_frame = plot_frame.rename(columns={0:'reads'})
            community_plot_frame = pd.concat([plot_frame, community_plot_frame])

        if specific_topic:
            title = 'Topic {} Information Read over Time by Community'.format(specific_topic)
        else:
            title = 'Total Information Read over Time'
            
        plt=(ggplot(community_plot_frame, aes(x='time',y='reads', color = 'label'))
             + geom_line()
             + facet_wrap('community')
             + ggtitle(title)
             + scale_color_manual(values=colors))
        
        output = '../output/topic_' + str(specific_topic) + '_by_community.png'
        plt.save(filename=output, width=16,height=6)
    
    return 'Finished Making Plots!'



plot_total_reads_over_time(by_community=True, specific_topic='0')
plot_total_reads_over_time(by_community=True, specific_topic='1')
plot_total_reads_over_time(by_community=True, specific_topic='2')
plot_total_reads_over_time(by_community=True, specific_topic='3')


def make_community_sentiment_plot(inpath):
    community_sentiment = pd.read_csv(inpath)
    community_sentiment['Topic'] = 'Topic ' + community_sentiment['Topic'].astype(int).apply(str)
    community_sentiment['Community'] = 'Community ' + community_sentiment['Community'].astype(int).apply(str)
    
    plt=(ggplot(community_sentiment,
                aes(x='Time',y='Mean Sentiment', color = 'Topic'))
         + geom_line()
         + facet_wrap('Community')
         + ggtitle('Mean Belief for Communties by Topic')
         + ylab('Mean Belief'))

    plt.save(filename='../output/mean_sentiment_by_community.png', width = 10, height = 7)

make_community_sentiment_plot(inpath =  inpath_community_sentiment)
