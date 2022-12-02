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



def plot_total_reads_over_time(midpoint, mitigation, file, by_community=False,specific_topic=None, smooth_interval = 25):
    
    colors = {'anti-misinfo':'blue', 'noise':'black', 'misinfo':'darkred'}  

    misinfo_none = pd.read_pickle('../output/mitigation-none/node_by_time_misinfo.pickle')
    anti_none = pd.read_pickle('../output/mitigation-none/node_by_time_anti.pickle')
    noise_none = pd.read_pickle('../output/mitigation-none/node_by_time_noise.pickle')
    misinfo_none['Mitigation'] = 'None'
    anti_none['Mitigation'] = 'None'
    noise_none['Mitigation'] = 'None'
    
    misinfo_mit = pd.read_pickle('../output/{}/node_by_time_misinfo.pickle'.format(file))
    anti_mit = pd.read_pickle('../output/{}/node_by_time_anti.pickle'.format(file))
    noise_mit = pd.read_pickle('../output/{}/node_by_time_noise.pickle'.format(file))
    misinfo_mit['Mitigation'] = mitigation
    anti_mit['Mitigation'] = mitigation
    noise_mit['Mitigation'] = mitigation
    
    combined = pd.concat([misinfo_none, anti_none, noise_none,misinfo_mit,anti_mit,noise_mit])
    combined = combined.sort_values(by=['Community', 'Topic', 'Type', 'Mitigation', 'Step'])

    
    if specific_topic:
        title = 'Total Topic {} Information Read over Time'.format(specific_topic)
        
    else:
        title = 'Total Information Read over Time'


    topic_frame = combined[combined['Topic']==specific_topic]
    
    topic_frame_total = topic_frame.groupby(['Topic', 'Type', 'Mitigation', 'Step']).agg({'Reads': 'sum'})
    topic_frame_total['Reads (moving average)'] = topic_frame_total.groupby(['Type','Mitigation'])['Reads'].transform(lambda x: x.rolling(smooth_interval, 1).mean())
    topic_frame_total = topic_frame_total.reset_index()
    
    plt=(ggplot(topic_frame_total,
                aes(x='Step',y='Reads (moving average)', color = 'Type', linetype = 'Mitigation'))
         + geom_line()
         + ggtitle(title)
         + scale_color_manual(values=colors)
         + geom_vline(aes(xintercept = midpoint), color = 'green', linetype = 'dashed'))

    plt.save(filename='../output/' + file + '/total_topic_{}.png'.format(specific_topic), width=8,height=6)
    
    
    if by_community:

        if specific_topic:
            title = 'Topic {} Information Read over Time by Community'.format(specific_topic)
        else:
            title = 'Total Information Read over Time'
            
        topic_frame_com = topic_frame
        topic_frame_com['Reads (moving average)'] = topic_frame_com.groupby(['Community', 'Type', 'Mitigation'])['Reads'].transform(lambda x: x.rolling(smooth_interval, 1).mean())
        topic_frame_com= topic_frame_com.reset_index()

        plt=(ggplot(topic_frame_com, aes(x='Step',y='Reads (moving average)', color = 'Type', linetype = 'Mitigation'))
             + geom_line()
             + facet_wrap('Community')
             + ggtitle(title)
             + scale_color_manual(values=colors)
             + geom_vline(aes(xintercept = midpoint), color = 'green', linetype = 'dashed'))
        
        output = '../output/' + file + '/topic_' + str(specific_topic) + '_by_community.png'
        plt.save(filename=output, width=16,height=6)
    
    return 'Finished Making Plots!'

def make_community_sentiment_plot(inpath_none, inpath_mit, mitigation, midpoint, file, smooth_interval):
    community_sentiment_none = pd.read_csv(inpath_none)
    community_sentiment_none['Mitigation'] = 'None'
    community_sentiment_mit = pd.read_csv(inpath_mit)
    community_sentiment_mit['Mitigation'] = str(mitigation)
    
    community_sentiment = pd.concat([community_sentiment_none, community_sentiment_mit])
    community_sentiment = community_sentiment.sort_values(by=['Mitigation', 'Community', 'Topic', 'Time'])
    community_sentiment['Mean Sentiment (smooth)'] = community_sentiment.groupby(['Community','Topic','Mitigation'])['Mean Sentiment'].transform(lambda x: x.rolling(smooth_interval, 1).mean())
    community_sentiment = community_sentiment.reset_index()
    community_sentiment['Topic'] = 'Topic ' + community_sentiment['Topic'].astype(int).apply(str)
    community_sentiment['Community'] = 'Community ' + community_sentiment['Community'].astype(int).apply(str)
    
    
    
    
    plt=(ggplot(community_sentiment,
                aes(x='Time',y='Mean Sentiment (smooth)', color = 'Topic', linetype = 'Mitigation'))
         + geom_line(alpha = 0.8)
         + facet_wrap('Community')
         + ggtitle('Mean Belief for Communties by Topic')
         + ylab('Mean Belief')
         + geom_vline(aes(xintercept = midpoint), colour = 'green', linetype= 'dashed'))

    plt.save(filename='../output/{}/mean_sentiment_by_community.png'.format(file), width = 10, height = 7)

if not os.path.isdir('../../4_simulation/output/mitigation-none'):
    print('Need to first run with no mitigation!!')
else:
    midpoint = 200
    file = 'mitigation-stop_reading_misinfo-labelmethod-average_truth_perception_random-sample_method-top_avg_origin_degree'
    inpath_community_sentiment_none = '../output/mitigation-none/community_sentiment_clean.csv'
    inpath_community_sentiment_mit = '../output/{}/community_sentiment_clean.csv'.format(file)
    mitigation = 'Stop Reading'
    plot_total_reads_over_time(by_community=True, specific_topic=0, file = file, mitigation = mitigation, midpoint = midpoint)
    plot_total_reads_over_time(by_community=True, specific_topic=1, file = file, mitigation = mitigation, midpoint = midpoint)
    plot_total_reads_over_time(by_community=True, specific_topic=2, file = file, mitigation = mitigation, midpoint = midpoint)
    plot_total_reads_over_time(by_community=True, specific_topic=3, file = file, mitigation = mitigation, midpoint = midpoint)
    make_community_sentiment_plot(inpath_none =  inpath_community_sentiment_none, 
                                  inpath_mit = inpath_community_sentiment_mit,
                                  midpoint = midpoint,
                                  mitigation = mitigation,
                                  file=file,
                                  smooth_interval=25)


