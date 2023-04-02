#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 11:05:50 2023

@author: tdn897
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import progressbar
import time
import gc
from plotnine import *

os.chdir('/Users/tdn897/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/5_analysis/src')

communities = [3, 34, 72]
reps = 10
interventions = ['TopPredictedByTopic_knowledgable_community_stratified_nodes_visited',
                 'TopPredictedByTopic_knowledgable_community_nodes_visited',
                 'TopPredictedByTopic_random_nodes_visited',
                 'TopPredictedByTopic_random_stratified_nodes_visited',
                 'TopPredictedByTopic_stratified_nodes_visited',
                 'TopPredictedByTopic_stratified_stratified_nodes_visited',
                 'TopPredicted_knowledgable_community_nodes_visited',
                 'TopPredicted_knowledgable_community_stratified_nodes_visited',
                 'TopPredicted_random_nodes_visited',
                 'TopPredicted_random_stratified_nodes_visited',
                 'TopPredicted_stratified_nodes_visited',
                 'TopPredicted_stratified_stratified_nodes_visited']


individual_regression_data = pd.read_csv('../output/world_state_2/individual_level_regression_data.csv')
individual_regression_data = individual_regression_data.rename(columns={'Change in Belief':'Change'})

no_intervention = individual_regression_data.loc[(individual_regression_data['Intervention']=='no_intervention_change_in_belief')]
all_interventions = individual_regression_data.loc[(individual_regression_data['Intervention']!='no_intervention_change_in_belief')]


results = []

for rep in range(reps):
    
    print('\n\n\n\n Rep #' + str(rep) + '\n\n\n\n')
    
    
    for community in communities:
        no_intervention_comm = no_intervention.loc[(no_intervention['Community']==community) & (no_intervention['Rep']==rep)].copy()
        no_intervention_comm.loc[:,'ImpactednessSum'] = no_intervention_comm.groupby('Node')['Impactedness'].transform('sum')
        no_intervention_comm.loc[:,'ImpactednessPct'] = no_intervention_comm['Impactedness']/no_intervention_comm['ImpactednessSum']
        no_intervention_comm_all_topics = pd.Series(no_intervention_comm.groupby(['Node']).apply(lambda x: np.average(x.Change, weights=x.ImpactednessPct)), name='no_intervention')
        
        for intervention in interventions:
            intervention_comm = all_interventions[(all_interventions['Community']==community) & \
                                                  (all_interventions['Rep']==rep) & \
                                                  (all_interventions['Intervention'] == intervention)].copy()
                
            intervention_comm.loc[:,'ImpactednessSum'] = intervention_comm['Impactedness'].groupby(intervention_comm['Node']).transform('sum')
            intervention_comm.loc[:,'ImpactednessPct'] = intervention_comm['Impactedness']/intervention_comm['ImpactednessSum']
            intervention_comm_all_topics = pd.Series(intervention_comm.groupby(['Node']).apply(lambda x: np.average(x.Change, weights=x.ImpactednessPct)), name='intervention')

            ate = pd.merge(no_intervention_comm_all_topics, intervention_comm_all_topics, right_index=True, left_index=True)
            ate['Difference'] = ate['intervention'] - ate['no_intervention']
            ate_est = np.mean(ate.Difference)
            mean_treated = np.mean(ate.intervention)
            mean_untreated = np.mean(ate.no_intervention)
            var_treated = (1/(len(ate) - 1))*np.sum((ate.intervention - mean_treated)**2)
            var_untreated = (1/(len(ate) - 1))*np.sum((ate.no_intervention - mean_untreated)**2)
            ate_var = np.sqrt((var_treated)/len(ate) + var_untreated/len(ate))
            results.append([rep, community, intervention, ate_est, ate_var])
                
            
results_frame = pd.DataFrame(results, columns = ['Rep', 'Community', 'Intervention', 'ATE_est', 'ATE_var'])\
    .sort_values(by=['Community','Intervention', 'Rep'])
    
results_frame_grouped = results_frame\
    .groupby(['Community', 'Intervention'])['ATE_est', 'ATE_var'].apply(lambda x: np.mean(x)).reset_index()
        
results_frame.to_csv('../output/ATE_Data_raw.csv', index=False)
results_frame_grouped.to_csv('../output/ATE_data.csv', index=False)
