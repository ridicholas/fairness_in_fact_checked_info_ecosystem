#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:52:27 2022

@author: tdn897
"""


import os
import xgboost as xgb
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
import numpy as np
from plotnine import *


os.chdir('/Users/tdn897/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/4_simulation/src')

inpath_checkworthy = '../output/mitigation-stop_reading_misinfo-labelmethod-average_truth_perception_knowledgable_community-sample_method-top_avg_origin_degree/checkworthy_data.pickle'
topics = [0,1,2,3]
labels_to_use = ['average_truth_perception_random', 'average_truth_perception_knowledgable_community']
pct_checkworthy = 0.25
reps = 100

with open(inpath_checkworthy, 'rb') as file:
    check = pickle.load(file)
    
fn_rate = []

for label in labels_to_use:
    print('\n\n\n\n --- Label: ' + label)
    for rep in range(reps):
        check_df = pd.DataFrame.from_dict(check.sampled_checkworthy_data).T.fillna(0)
        check_df['target'] = check_df[label]
        
        
        train, test = train_test_split(check_df, test_size=0.3)
        train_x = train[[i for i in train.columns if ('truth' not in i) and ('claim' not in i) and ('target' not in i) and ('outcome' not in i) and ('value' not in i)]]
        train_y = train[['target']]
        train_ground_truth = train.value.values
        train_ground_truth[train_ground_truth == -1] = 0
        
        test_x = test[[i for i in test.columns if ('truth' not in i) and ('claim' not in i) and ('target' not in i) and ('outcome' not in i) and ('value' not in i)]]
        test_y = test[['target']]
        test_ground_truth = test.value.values
        test_ground_truth[test_ground_truth == -1] = 0
        
        
        
        clf = xgb.XGBRegressor().fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        cols_when_model_builds = clf.get_booster().feature_names
        print('\n\n --- XGBoost Test Set Results (MSE)): ' + str(mean_squared_error(test_y, pred_y)) + '----- \n\n')
        print('\n\n --- XGBoost Test Set Compared to Ground Truth (MSE): ' + str(mean_squared_error(test_ground_truth, pred_y)) + '----\n\n\n')
        
        
        
        
        test['preds'] = pred_y
        # top 25% are checkworthy
        test['checkworthy'] = np.where(test['preds'] >= test.preds.quantile(0.75), 1, 0)
        
        
        if 'knowledgable' in label:
            clean_label = 'Label = Knowledgable Community'
        elif 'random' in label:
            clean_label = 'Label = Random Sample'
        
        print('-- Rep: ' + str(rep) + ' --\n')
        for topic in topics:
            test_topic = test[test['topic']==topic]
            test_topic_ground_truth = test_topic.value.values
            test_topic_checkworthy = test_topic.checkworthy.values
            confusion_matrix(test_topic_ground_truth, test_topic_checkworthy)
            test_topic_ground_truth[test_topic_ground_truth == -1] = 0
            cm = confusion_matrix(test_topic_ground_truth, test_topic_checkworthy)
            fn_rate.append(['Topic ' + str(topic), cm[1,0]/np.sum(cm), rep, clean_label])
        
plot_frame = pd.DataFrame(fn_rate, columns = ['Topic', 'False Neg Rate', 'Rep', 'Label Method'])

plt = (ggplot(plot_frame)
 + geom_boxplot(aes(x='Topic', y = 'False Neg Rate', color = 'Topic'))
 + facet_wrap('Label Method'))
    
plt.save(filename='../output/model_evaluation.png'.format(file), width = 10, height = 7)
