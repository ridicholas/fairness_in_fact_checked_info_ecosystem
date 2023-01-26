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


'''
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

'''

def plot_total_misinfo_reads_over_time(file, midpoint, mods_to_include, smooth_interval = 25):
    
    
    
    colors = {'No Intervention':'black', '(Baseline) Random Label - Top View Sampling - Standard Mitigation':'darkred', 'KC Label - Strat. Top View Sampling - Topic Mitigation':'blue'}  

    reads_over_time = pd.read_csv(file)
    reads_over_time = reads_over_time.loc[(reads_over_time['Mod'].isin(mods_to_include))]
    reads_over_time = reads_over_time.sort_values(by=['Community', 'Topic', 'Info_Type', 'Mod', 'Time'])
    reads_over_time['Mod'] = np.where(reads_over_time['Mod'] == 'final_no_intervention_', 'No Intervention',
                                               np.where(reads_over_time['Mod']=='TopPredictedByTopic_knowledgable_community_stratified_nodes_visited_', 'KC Label - Strat. Top View Sampling - Topic Mitigation',
                                                        np.where(reads_over_time['Mod']=='TopPredicted_random_nodes_visited_', '(Baseline) Random Label - Top View Sampling - Standard Mitigation', 'Community Stratified Label')))
    
    
    reads_over_time['Topic'] = 'Topic ' + reads_over_time['Topic'].astype(int).apply(str)
    reads_over_time['Community'] = 'Community ' + reads_over_time['Community'].astype(int).apply(str)

    
    reads_over_time_mean = reads_over_time.groupby(by=['Community', 'Topic', 'Info_Type', 'Mod', 'Time']).\
        agg(mean_reads=pd.NamedAgg(column='reads', aggfunc='mean')).reset_index()
    
    reads_over_time_mean['Reads (Cumulative)'] = reads_over_time_mean.groupby(['Info_Type','Community','Mod', 'Topic'])['mean_reads'].transform(pd.Series.cumsum)



    
    # 1. Community + Topic Breakdown of Misinfo read across trials.
    reads_over_time_misinfo = reads_over_time_mean\
        .loc[(reads_over_time_mean['Info_Type']=='misinfo')]


    plt=(ggplot(reads_over_time_misinfo, aes(x='Time',y='Reads (Cumulative)', color = 'Mod'))
                 + geom_line()
                 + facet_wrap('~ Community + Topic', scales = 'free_y')
                 + ggtitle('Misinformation Read By Community')
                 + scale_color_manual(values=colors)
                 + geom_vline(aes(xintercept = midpoint), color = 'green', linetype = 'dashed'))

    output = '../output/misinfo_reads_results.png'
    plt.save(filename=output, width=16,height=6, dpi=1000)
    
    # 2. Topic Breakdown of Misinfo read across trials.

    reads_over_time_topic = reads_over_time_mean\
        .loc[(reads_over_time_mean['Info_Type']=='misinfo')]\
            .groupby(['Mod', 'Topic', 'Time'])\
                .agg(Reads=pd.NamedAgg(column='Reads (Cumulative)', aggfunc='sum')).reset_index()
    
    plt=(ggplot(reads_over_time_topic, aes(x='Time',y='Reads', color = 'Mod'))
                 + geom_line()
                 + facet_wrap('Topic')
                 + ggtitle('Misinformation Read By Topic')
                 + ylab('Reads (Cumulative)')
                 + scale_color_manual(values=colors)
                 + geom_vline(aes(xintercept = midpoint), color = 'green', linetype = 'dashed'))

    output = '../output/misinfo_reads_topic_results.png'
    plt.save(filename=output, width=10,height=6, dpi=1000)


    #3. Community Breakdown of Misinfo read across trials
    reads_over_time_comm = reads_over_time_mean\
        .loc[(reads_over_time_mean['Info_Type']=='misinfo')]\
            .groupby(['Mod', 'Community', 'Time'])\
                .agg(Reads=pd.NamedAgg(column='Reads (Cumulative)', aggfunc='sum')).reset_index()
    
    plt=(ggplot(reads_over_time_comm, aes(x='Time',y='Reads', color = 'Mod'))
                 + geom_line()
                 + facet_wrap('Community')
                 + ggtitle('Misinformation Read By Community')
                 + ylab('Reads (Cumulative)')
                 + scale_color_manual(values=colors)
                 + geom_vline(aes(xintercept = midpoint), color = 'green', linetype = 'dashed'))

    output = '../output/misinfo_reads_comm_results.png'
    plt.save(filename=output, width=16,height=6, dpi=1000)
    
    #4. Total breakdown of Misinfo Read Across the Network
    reads_over_time_total = reads_over_time_mean\
        .loc[(reads_over_time_mean['Info_Type']=='misinfo')]\
            .groupby(['Mod', 'Time'])\
                .agg(Reads=pd.NamedAgg(column='Reads (Cumulative)', aggfunc='sum')).reset_index()

    plt=(ggplot(reads_over_time_total, aes(x='Time',y='Reads', color = 'Mod'))
                 + geom_line()
                 + ggtitle('Misinformation Read Across Network')
                 + scale_color_manual(values=colors)
                 + geom_vline(aes(xintercept = midpoint), color = 'green', linetype = 'dashed'))

    output = '../output/misinfo_reads_total_results.png'
    plt.save(filename=output, width=10,height=6, dpi=1000)





def make_community_sentiment_plot(midpoint, smooth_interval = 25):
    
    
    colors = {'No Intervention':'black', 'Random Label - Top View Sampling - Standard Mitigation':'darkred', 'KC Label - Strat. Top View Sampling - Topic Mitigation':'orange'}  


    community_sentiment = pd.read_csv('../output/exp_results_community_belief.csv')
    community_sentiment['Intervention'] = np.where(community_sentiment['Intervention'] == 'no_intervention', 'No Intervention',
                                               np.where(community_sentiment['Intervention']=='intervention_kc_label_stratified_sample_topic_mitigation', 'KC Label - Strat. Top View Sampling - Topic Mitigation',
                                                        np.where(community_sentiment['Intervention']=='intervention_random_label_random_sample_standard_mitigation', 'Random Label - Top View Sampling - Standard Mitigation', 'Community Stratified Label')))
    community_sentiment['Topic'] = 'Topic ' + community_sentiment['Topic'].astype(int).apply(str)
    community_sentiment['Community'] = 'Community ' + community_sentiment['Community'].astype(int).apply(str)
    
    community_sentiment_mean = community_sentiment.groupby(by=['Community', 'Topic', 'Intervention', 'Time']).\
        agg(mean_belief=pd.NamedAgg(column='Mean Belief', aggfunc='mean')).reset_index()
    
    community_sentiment_mean['Mean Belief (moving average)'] = community_sentiment_mean.groupby(['Community','Intervention', 'Topic'])['mean_belief'].transform(lambda x: x.rolling(smooth_interval, 1).mean())

    
    
    
    # 1. By Topic and Community
    plt=(ggplot(community_sentiment_mean,
                aes(x='Time',y='Mean Belief (moving average)', color = 'Intervention'))
         + geom_line(alpha = 0.8)
         + facet_wrap('~ Community + Topic', scales = 'free_y')
         + ggtitle('Mean Belief by Communties and Topic')
         + ylab('Mean Belief')
         + scale_color_manual(values = colors)
         + geom_vline(aes(xintercept = midpoint), colour = 'green', linetype= 'dashed'))

    plt.save(filename='../output/mean_belief_results.png', width = 16, height = 6)


    #2. By Community
    
    
    community_sentiment_mean_comm = community_sentiment_mean\
            .groupby(['Intervention', 'Community', 'Time'])\
                .agg(Mean_Belief=pd.NamedAgg(column='Mean Belief (moving average)', aggfunc='mean')).reset_index()

    
    
    plt=(ggplot(community_sentiment_mean_comm,
                aes(x='Time',y='Mean_Belief', color = 'Intervention'))
         + geom_line(alpha = 0.8)
         + facet_wrap('~ Community', scales = 'free_y')
         + ggtitle('Mean Belief Across All Topics by Community')
         + ylab('Mean Belief')
         + scale_color_manual(values = colors)
         + geom_vline(aes(xintercept = midpoint), colour = 'green', linetype= 'dashed'))

    plt.save(filename='../output/mean_belief_results_by_community.png', width = 16, height = 6)

    #4. Community + Topic for one intervention

    community_sentiment_raw = community_sentiment[community_sentiment['Intervention']=='No Intervention']

    community_sentiment_mean = community_sentiment_raw.groupby(by=['Community', 'Topic', 'Intervention', 'Time']).\
            agg(mean_belief=pd.NamedAgg(column='Mean Belief', aggfunc='mean')).reset_index()

    community_sentiment_mean['Mean Belief (moving average)'] = community_sentiment_mean.groupby(['Community','Intervention', 'Topic'])['mean_belief'].transform(lambda x: x.rolling(smooth_interval, 1).mean())

    plt=(ggplot(community_sentiment_mean, aes(x='Time',y='Mean Belief (moving average)', color = 'Topic'))
             + geom_line()
             + facet_wrap('Community')
             + ggtitle('Change in Belief w/ No Intervention\nBy Community + Topic')
             + geom_vline(aes(xintercept = midpoint), color = 'green', linetype = 'dashed'))

    plt.save(filename='../output/mean_belief_no_intervention.png', width = 12, height = 6)


plot_total_misinfo_reads_over_time(file = '../output/misinfo_reads_43_56_127.csv', 
                                   mods_to_include = ['final_no_intervention_', 'TopPredictedByTopic_knowledgable_community_stratified_nodes_visited_', 'TopPredicted_random_nodes_visited_'],
                                   midpoint = 150)
make_community_sentiment_plot(midpoint = 100)