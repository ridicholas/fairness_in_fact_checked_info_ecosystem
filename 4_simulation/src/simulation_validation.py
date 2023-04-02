#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:29:27 2022

@author: tdn897
"""


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import progressbar
import networkx as nx
import gc
os.chdir('/Users/tdn897/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/4_simulation/src')

from checkworthy import Checkworthy
from TopicSim import TopicSim
from plotnine import *
# making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def calc_cascade_stats(utterance):
    claim = utterance['claim']
    value = utterance['value']
    topic = utterance['topic']
    size = len(utterance['re-tweets']) + 1
    max_depth = 1
    breadth = 1
    breadth_at_depth = {}
    max_breadth_at_depth = 1
    if len(utterance['re-tweets'])>0:
        for i in range(len(utterance['re-tweets'])):
            depth = nx.shortest_path_length(sim.G, utterance['re-tweets'][i], utterance['node-origin'])
            if depth == max_depth:
                breadth += 1
                breadth_at_depth.update({depth: breadth})
            if depth > max_depth:
                breadth = 1
                max_depth = depth
                breadth_at_depth.update({depth:breadth})
                
        max_breadth_at_depth = max(breadth_at_depth.values())
                
    return [topic, str(topic) + '-' + str(claim), value, size, max_depth, max_breadth_at_depth]



def calculate_ccdf(results_frame):
    
    metrics = ['Size of Cascade (Nodes)', 'Max Depth of Cascade', 'Max Breadth  of Cascade']
    info_types = ['False', 'Truth']
    
    ccdf_frame = pd.DataFrame()
    
    
    for metric in metrics:
        for info_type in info_types:

            if info_type == 'False':
                sum_frame = results_frame.loc[(results_frame['value']==1)]\
                    .groupby([metric])['claim'].count().reset_index()
            elif info_type == 'Truth':
                sum_frame = results_frame.loc[(results_frame['value']==-1)]\
                    .groupby([metric])['claim'].count().reset_index()
        
            mapping = {metric: 'value'}
            sum_frame = sum_frame.rename(columns=mapping)
            sum_frame['CCDF (%)'] = 0
            
            for i in range(len(sum_frame)):
                sum_frame.loc[i,"CCDF (%)"] = np.sum(sum_frame.claim.values[i:len(sum_frame)])/np.sum(sum_frame.claim.values)*100
                
            sum_frame['type'] = metric
            sum_frame['Info Veracity'] = info_type
            
            ccdf_frame = ccdf_frame.append(sum_frame)
    
    
    return ccdf_frame

#0, 5!, 6, 9

infile = '../output/simulation_pre_period_run9_communities_3_34_72.pickle'
with open(infile, 'rb') as file:
    sim = pickle.load(file)

all_info = sim.all_info

result = []
for key in list(all_info.keys()):
    result.append(calc_cascade_stats(utterance=sim.all_info[key]))
    
results_frame = pd.DataFrame(result, columns = ['topic', 'claim', 'value', 'Size of Cascade (Nodes)', 'Max Depth of Cascade', 'Max Breadth  of Cascade'])
ccdf_frame = calculate_ccdf(results_frame = results_frame)


measures = ['Size of Cascade (Nodes)', 'Max Breadth  of Cascade', 'Max Depth of Cascade']
for i in range(len(measures)):
    
    ccdf_measure = ccdf_frame.loc[ccdf_frame['type']==measures[i]]
    
    
    if i == 0:
        
        g = (ggplot(ccdf_measure)
         + geom_line(aes(x='value', y = 'CCDF (%)', color = 'Info Veracity'), size = 1.5)
         + theme_light()
         + theme(legend_position=(0.7, 0.75), text=element_text(size=16))
         + scale_y_log10()
         + scale_x_log10()
         + xlab(measures[i]))
        
        g.save('../output/simulation_validation' + measures[i] + '.png', width=6, height =6, dpi=600)
    
    else:
        
        g = (ggplot(ccdf_measure)
         + geom_line(aes(x='value', y = 'CCDF (%)', color = 'Info Veracity'), size = 1.5)
         + theme_light()
         + theme(legend_position='none', text=element_text(size=16))
         + scale_y_log10()
         + scale_x_log10()
         + xlab(measures[i]))
        
        g.save('../output/simulation_validation' + measures[i] + '.png', width=6, height =6, dpi=600)

        
        





type_ccdf = results_frame\
    .groupby(['value', 'claim']).size().reset_index(name='draws')\
        .groupby(['draws', "value"])['claim'].count().reset_index()\
            .sort_values(by=['value', 'draws']).rename(columns={'claim': 'Number of Cascades'})

types = [-1, 1]

type_ccdf['CCDF (%)'] = 0
final_ccdf = pd.DataFrame()
for info in types:
    tmp = type_ccdf.loc[(type_ccdf['value']==info)].reset_index().sort_values('draws')
    for i in range(len(tmp)):
        tmp.loc[tmp.index.values[i], 'CCDF (%)'] = np.sum(tmp['Number of Cascades'].values[i:len(tmp)])/np.sum(tmp['Number of Cascades'].values)*100
    final_ccdf = final_ccdf.append(tmp)

final_ccdf['Info Veracity'] = np.where(final_ccdf['value']==-1, 'Truth', 'False')

g1 = (ggplot(final_ccdf)
 + geom_line(aes(x='draws', y = 'CCDF (%)', color = 'Info Veracity'), size = 1.5)
 + theme_light()
 + theme(legend_position='none', text=element_text(size=16))
 + scale_y_log10()
 + scale_x_log10()
 + xlab('Number of Utterances (Cascades) per Claim'))


g1.save('../output/simulation_validation_information_distribution.png', width=6, height =6, dpi = 600)
