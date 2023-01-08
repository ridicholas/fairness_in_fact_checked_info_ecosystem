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
            sum_frame['info_type'] = info_type
            
            ccdf_frame = ccdf_frame.append(sum_frame)
    
    
    return ccdf_frame


infile = '../output/simulation_pre_period0.pickle'
with open(infile, 'rb') as file:
    sim = pickle.load(file)

n_sample = 4000000
random_keys = np.random.choice(list(sim.all_info), size = n_sample, replace = False)



result = []
for key in random_keys:
    result.append(calc_cascade_stats(utterance=sim.all_info[key]))
    
results_frame = pd.DataFrame(result, columns = ['topic', 'claim', 'value', 'Size of Cascade (Nodes)', 'Max Depth of Cascade', 'Max Breadth  of Cascade'])
ccdf_frame = calculate_ccdf(results_frame = results_frame)


g = (ggplot(ccdf_frame)
 + geom_line(aes(x='value', y = 'CCDF (%)', color = 'info_type'), size = 1.5)
 + facet_wrap('~ type',
              scales = 'free_x')
 + theme(panel_grid_major = element_blank(), panel_grid_minor = element_blank())
 + scale_y_log10()
 + scale_x_log10())



g.save('../output/simulation_validation.png', width=14, height =6)

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

final_ccdf['Info Type'] = np.where(final_ccdf['value']==-1, 'Truth', 'False')

g1 = (ggplot(final_ccdf)
 + geom_line(aes(x='draws', y = 'CCDF (%)', color = 'Info Type'), size = 1.5)
 + theme(panel_grid_major = element_blank(), panel_grid_minor = element_blank())
 + scale_y_log10()
 + scale_x_log10()
 + xlab('Number of Cascades for Specific Rumor'))


g1.save('../output/simulation_validation_information_distribution.png', width=7, height =6)
