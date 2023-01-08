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



inpath_pre = '../../4_simulation/output/simulation_pre_period5.pickle'

topics = [0,1,2,3]
labels_to_use = ['average_truth_perception_random',
                 'average_truth_perception_knowledgable_community',
                 'average_truth_perception_stratified']

label_strategy = ['random',
                  'knowledgable_community',
                  'stratified']

pct_checkworthy = 0.25
reps = 100

with open(inpath_pre, 'rb') as file:
    sim = pickle.load(file)

check = sim.return_check()
check.sample_claims(num_to_sample=15000, sample_method='top_avg_origin_degree')
check.sample_labels_for_claims(labels_per_claim = 100, sample_method = label_strategy[0])
check.sample_labels_for_claims(labels_per_claim = 100, sample_method = label_strategy[1])
check.sample_labels_for_claims(labels_per_claim = 100, sample_method = label_strategy[2])


fn_rate = []
fidelity = []
accuracy = []

for i in range(len(labels_to_use)):
    print('\n\n\n\n --- Label: ' + labels_to_use[i] + ' \n\n\n\n')
    for rep in range(reps):

        if 'knowledgable' in labels_to_use[i]:
            clean_label = 'Label = Knowledgable Community'
        elif 'random' in labels_to_use[i]:
            clean_label = 'Label = Random Sample'
        elif 'stratified' in labels_to_use[i]:
            clean_label = 'Label = Stratified Sample'


        check_df = pd.DataFrame.from_dict(check.sampled_checkworthy_data).T.fillna(0)


        check_df['target'] = check_df[labels_to_use[i]]


        train, test = train_test_split(check_df, test_size=0.8)
        train_x = train[[i for i in train.columns if ('truth' not in i) and ('claim' not in i) and ('num_of_origin' not in i) and ('target' not in i) and ('outcome' not in i) and ('value' not in i)]]
        train_y = train[['target']]
        train_ground_truth = train.value.values
        train_ground_truth[train_ground_truth == -1] = 0

        test_x = test[[i for i in test.columns if ('truth' not in i) and ('claim' not in i) and ('num_of_origin' not in i) and ('target' not in i) and ('outcome' not in i) and ('value' not in i)]]
        test_y = test[['target']]
        test_ground_truth = test.value.values
        test_ground_truth[test_ground_truth == -1] = 0



        clf = xgb.XGBRegressor().fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        
        label_mse = mean_squared_error(test_y, pred_y)
        true_mse = mean_squared_error(test_ground_truth, pred_y)
        cols_when_model_builds = clf.get_booster().feature_names
        print('\n\n --- XGBoost Test Set Results (MSE)): ' + str(label_mse) + '----- \n\n')
        print('\n\n --- XGBoost Test Set Compared to Ground Truth (MSE): ' + str(true_mse) + '----\n\n\n')
        fidelity.append([label_mse, rep, clean_label])
        accuracy.append([true_mse, rep, clean_label])



        test['preds'] = pred_y
        # top 25% are checkworthy
        test['checkworthy'] = np.where(test['preds'] >= test.preds.quantile(0.75), 1, 0)

        

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
 + geom_boxplot(aes(x='Topic', y = 'False Neg Rate', fill = 'Topic'))
 + facet_wrap('Label Method'))

plt.save(filename='../output/checkworthy_model_evaluation_false_negative.png', width = 12, height = 4)

fidelity_frame = pd.DataFrame(fidelity, columns = ['MSE', 'Rep', 'Label'])
fidelity_frame['Metric'] = 'Label Fidelity MSE'
accuracy_frame = pd.DataFrame(accuracy, columns = ['MSE', 'Rep', 'Label'])
accuracy_frame['Metric'] = 'Ground Truth MSE'

result = pd.concat([fidelity_frame, accuracy_frame])

plt = (ggplot(result)
 + geom_boxplot(aes(x='Label', y = 'MSE', fill = 'Label'))
 + facet_wrap('Metric', scales = 'free_y')
 + theme(axis_text_x=element_blank()))

plt.save(filename='../output/checkworthy_model_evaluation_fidelity.png', width = 12, height = 4)
