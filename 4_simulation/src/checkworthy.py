#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:26:27 2022

@author: tdn897
"""
import pandas as pd

class Checkworthy():


    def __init__(self, agg_interval, agg_steps, depths, impactednesses):
        self.checkworthy_data = {}
        self.sampled_checkworthy_data = {}
        self.agg_interval = agg_interval
        self.agg_steps = agg_steps
        self.impactednesses = impactednesses
        self.depths = depths
        self.node = 'empty'
        self.data = 'empty'
        self.claim_id = 'empty'
        self.value = 'empty'
        self.topic = 'empty'
        self.claim = 'empty'

        from networkx import shortest_path_length, get_node_attributes
        self.shortest_path_length = shortest_path_length
        self.get_node_attributes = get_node_attributes


    def set_network(self, G, communities):
        self.G = G
        self.communities = communities



    def intake_information(self, node, data, claim_id, value, topic, claim):
        self.node = node
        self.data = data
        self.claim_id = claim_id
        self.value = value
        self.topic = topic
        self.claim = claim
        #if self.claim_id not in self.checkworthy_data.keys():
        #    self.update_keys()


    def update_keys(self):
        self.checkworthy_data.update({self.claim_id:
                                 {'topic':self.topic,
                                  'value':self.value,
                                  'claim':self.claim,
                                  'num_of_origins': 1,
                                  'avg_degree_of_origins': self.data['degree'],
                                  'max_degree_of_origins': self.data['degree'],
                                  'avg_centrality_of_origins': self.data['centrality'],
                                  'max_centrality_of_origins': self.data['centrality']}})
        
        for comm in self.communities:
            if comm == self.data['Community']:
                self.checkworthy_data[self.claim_id]['nodes_visited_in_community{}_count'.format(comm)] = 1
                self.checkworthy_data[self.claim_id]['nodes_visited_in_community{}_avg'.format(comm)] = 1
            else:
                self.checkworthy_data[self.claim_id]['nodes_visited_in_community{}_count'.format(comm)] = 0
                self.checkworthy_data[self.claim_id]['nodes_visited_in_community{}_avg'.format(comm)] = 0



    def update_agg_values(self):
        claim_id = self.claim_id
        data = self.data

        self.checkworthy_data[claim_id]['num_of_origins'] += 1
        if data['degree'] > self.checkworthy_data[claim_id]['max_degree_of_origins']:
            self.checkworthy_data[claim_id]['max_degree_of_origins'] = data['degree']
        if data['centrality'] > self.checkworthy_data[claim_id]['max_centrality_of_origins']:
            self.checkworthy_data[claim_id]['max_centrality_of_origins'] = data['centrality']
        self.checkworthy_data[claim_id]['avg_degree_of_origins'] += (data['degree'] - self.checkworthy_data[claim_id]['avg_degree_of_origins'])/self.checkworthy_data[claim_id]['num_of_origins']
        self.checkworthy_data[claim_id]['avg_centrality_of_origins'] += (data['centrality'] - self.checkworthy_data[claim_id]['avg_centrality_of_origins'])/self.checkworthy_data[claim_id]['num_of_origins']

    def update_time_values(self, time_feature, origin_node):

        claim_id = self.claim_id
        node = self.node
        data = self.data
        depths = self.depths

        if 'step{}_nodes_visited_count'.format(time_feature) not in self.checkworthy_data[claim_id].keys():
            self.checkworthy_data[claim_id]['step{}_nodes_visited_count'.format(time_feature)] = 1
            self.checkworthy_data[claim_id]['step{}_nodes_visited_avg'.format(time_feature)] = 1

            self.checkworthy_data[claim_id]['step{}_avg_degree_visited'.format(time_feature)] = data['degree']
            self.checkworthy_data[claim_id]['step{}_avg_centrality_visited'.format(time_feature)] = data['centrality']
            self.checkworthy_data[claim_id]['step{}_max_degree_visited'.format(time_feature)] = data['degree']
            self.checkworthy_data[claim_id]['step{}_max_centrality_visited'.format(time_feature)] = data['centrality']
            self.checkworthy_data[claim_id]['step{}_max_depth_from_origin'.format(time_feature)] = 1
            for depth in depths:
                self.checkworthy_data[claim_id]['step{0}_nodes_at_depth{1}_count'.format(time_feature, depth)] = 0
                self.checkworthy_data[claim_id]['step{0}_nodes_at_depth{1}_avg'.format(time_feature, depth)] = 0

        else:
            self.checkworthy_data[claim_id]['step{}_nodes_visited_count'.format(time_feature)] += 1
            self.checkworthy_data[claim_id]['step{}_nodes_visited_avg'.format(time_feature)] = self.checkworthy_data[claim_id]['step{}_nodes_visited_count'.format(time_feature)] / self.checkworthy_data[claim_id]['num_of_origins']

            if data['degree'] > self.checkworthy_data[claim_id]['step{}_max_degree_visited'.format(time_feature)]:
                self.checkworthy_data[claim_id]['step{}_max_degree_visited'.format(time_feature)] = data['degree']
            if data['centrality'] > self.checkworthy_data[claim_id]['step{}_max_centrality_visited'.format(time_feature)]:
                self.checkworthy_data[claim_id]['step{}_max_centrality_visited'.format(time_feature)] = data['centrality']

            self.checkworthy_data[claim_id]['step{}_avg_degree_visited'.format(time_feature)] += (data['degree'] - self.checkworthy_data[claim_id]['step{}_avg_degree_visited'.format(time_feature)])/self.checkworthy_data[claim_id]['step{}_nodes_visited_count'.format(time_feature)]
            self.checkworthy_data[claim_id]['step{}_avg_centrality_visited'.format(time_feature)] += (data['centrality'] - self.checkworthy_data[claim_id]['step{}_avg_centrality_visited'.format(time_feature)])/self.checkworthy_data[claim_id]['step{}_nodes_visited_count'.format(time_feature)]

            distance = self.shortest_path_length(self.G, node, origin_node)

            for depth in depths:
                if distance == depth:
                    self.checkworthy_data[claim_id]['step{0}_nodes_at_depth{1}_count'.format(time_feature, depth)] += 1
                    self.checkworthy_data[claim_id]['step{0}_nodes_at_depth{1}_avg'.format(time_feature, depth)] = self.checkworthy_data[claim_id]['step{0}_nodes_at_depth{1}_count'.format(time_feature, depth)] / self.checkworthy_data[claim_id]['num_of_origins']

            if distance > self.checkworthy_data[claim_id]['step{}_max_depth_from_origin'.format(time_feature)]:
                self.checkworthy_data[claim_id]['step{}_max_depth_from_origin'.format(time_feature)] = distance
            
            self.checkworthy_data[claim_id]['nodes_visited_in_community{}_count'.format(data['Community'])] += 1
            self.checkworthy_data[claim_id]['nodes_visited_in_community{}_avg'.format(data['Community'])] = self.checkworthy_data[claim_id]['nodes_visited_in_community{}_count'.format(data['Community'])] / self.checkworthy_data[claim_id]['num_of_origins']


    def sample_claims(self, num_to_sample=2000, sample_method='random'):
        import random

        checkworthy_data = pd.DataFrame(self.checkworthy_data).T.fillna(0).copy()
        checkworthy_data['total_nodes_visited'] = 0
        visit_cols = []
        for comm in self.communities:
            visit_cols.append('nodes_visited_in_community{}_avg'.format(comm))
        for col in visit_cols:
            checkworthy_data['total_nodes_visited'] += checkworthy_data[col]

        numSamples = min(num_to_sample, checkworthy_data.shape[0])
        if sample_method == 'random': #fully random sampling of claims
            checkworthy_data = checkworthy_data.loc[random.sample(list(self.checkworthy_data.index), numSamples), :]
        
        if sample_method == 'top_num_origins': #probably not a great idea
            checkworthy_data = checkworthy_data.sort_values(by='num_of_origins', ascending=False).iloc[0:numSamples, :]

        if sample_method == 'top_max_origin_degree':
            checkworthy_data = checkworthy_data.sort_values(by='max_degree_of_origins', ascending=False).iloc[0:numSamples, :]

        if sample_method == 'top_avg_origin_degree':
            checkworthy_data = checkworthy_data.sort_values(by='avg_degree_of_origins', ascending=False).iloc[0:numSamples, :]
        
        if sample_method == 'nodes_visited':
            checkworthy_data = checkworthy_data.sort_values(by='total_nodes_visited', ascending=False).iloc[0:numSamples, :]

        if sample_method == 'stratified_nodes_visited':
            indexes_to_keep = []
            numSamples = int(numSamples/len(self.communities))
            check_copy = checkworthy_data.copy(deep=True)
            for comm in self.communities:
                indexes = list(check_copy.sort_values(by='nodes_visited_in_community{}_avg'.format(comm), ascending=False).iloc[0:numSamples, :].index)
                check_copy.drop(index=indexes, inplace=True)
                indexes_to_keep += indexes
            checkworthy_data.drop(index=list(set(checkworthy_data.index) - set(indexes_to_keep)), inplace=True)
            
        self.sampled_checkworthy_data = self.checkworthy_data.copy()
        for key in list(self.checkworthy_data.keys()):
            if key not in checkworthy_data.index:
                self.sampled_checkworthy_data.pop(key)
        
        

    def sample_labels_for_claims(self, labels_per_claim = 6, sample_method = 'random'):

        import pandas as pd
        import random
        import numpy as np

        G = self.G
        impactedness = self.impactednesses
        communities = pd.DataFrame.from_dict(data = self.get_node_attributes(G, 'Community'), orient='index').rename(columns={0:'Community'})
        belief = self.get_node_attributes(G, 'sentiment')
        all_nodes = list(G.nodes())

        for claim_id in self.sampled_checkworthy_data:

            topic = self.checkworthy_data[claim_id]['topic']
            value = self.checkworthy_data[claim_id]['value']

            if sample_method == 'random':
                nodes_to_survey = random.sample(all_nodes, k=labels_per_claim)
            elif sample_method == 'stratified':
                nodes_to_survey = communities.groupby('Community', group_keys=False).apply(lambda x: x.sample(int(labels_per_claim/3))).index.to_list()
            elif sample_method == 'knowledgable_community':
                max_keys = [key for key, value in impactedness[topic].items() if value == max(impactedness[topic].values())]
                nodes_to_survey = communities[communities['Community'].isin(max_keys)].apply(lambda x: x.sample(labels_per_claim, replace = True)).index.to_list()

            survey_results = []
            for node in nodes_to_survey:
                chance = np.max([0,belief[node][topic]])
                chance = np.min([1,chance])
                if value == -1:
                    survey_results.append(np.random.choice([0,1], size=1, p=[1-chance,chance])[0])
                elif value == 1:
                    survey_results.append(np.random.choice([0,1], size=1, p=[chance,1-chance])[0])
                elif value == 0:
                    survey_results.append(np.random.choice([0,1], size=1, p=[0.95, 0.05])[0])


            mean_survey = np.mean(survey_results)
            self.sampled_checkworthy_data[claim_id]['average_truth_perception_{}'.format(sample_method)] = mean_survey
    
    def train_model(self, label_to_use):
        import pandas as pd
        import xgboost as xg
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error

        print('Training Model on Survey Labels')
        check_df = pd.DataFrame.from_dict(self.sampled_checkworthy_data).T.fillna(0)
        check_df['target'] = check_df[label_to_use]
        train, test = train_test_split(check_df, test_size=0.2)
        train_x = train[[i for i in train.columns if ('num_of_origins' not in i) and ('count' not in i) and ('truth' not in i) and ('claim' not in i) and ('target' not in i) and ('outcome' not in i) and ('value' not in i)]]
        train_y = train[['target']]
        test_x = test[[i for i in test.columns if ('num_of_origins' not in i) and ('count' not in i) and ('truth' not in i) and ('claim' not in i) and ('target' not in i) and ('outcome' not in i) and ('value' not in i)]]
        test_y = test[['target']]
        clf = xg.XGBRegressor(booster='gbtree').fit(train_x, train_y)
        self.cols_when_model_builds = clf.get_booster().feature_names
        print('\n\n --- XGBoost Test Set Results -- survey label --- (MSE)): ' + str(mean_squared_error(test_y, clf.predict(test_x))) + '----- \n\n')
        self.clf = clf
        
        print('Training Model on Nodes Visited')
        check_df = pd.DataFrame.from_dict(self.sampled_checkworthy_data).T.fillna(0)
        check_df['total_nodes_visited'] = 0
        visit_cols = []
        for comm in self.communities:
            visit_cols.append('nodes_visited_in_community{}_avg'.format(comm))
        for col in visit_cols:
            check_df['total_nodes_visited'] += check_df[col]
        train, test = train_test_split(check_df, test_size=0.2)
        train_x = train[[i for i in train.columns if ('num_of_origins' not in i) and ('count' not in i) and ('total_nodes' not in i) and ('truth' not in i) and ('claim' not in i) and ('target' not in i) and ('outcome' not in i) and ('value' not in i)]]
        train_y = train[['total_nodes_visited']]
        test_x = test[[i for i in test.columns if ('num_of_origins' not in i) and ('total_nodes' not in i) and ('count' not in i) and ('truth' not in i) and ('claim' not in i) and ('target' not in i) and ('outcome' not in i) and ('value' not in i)]]
        test_y = test[['total_nodes_visited']]
        reg = xg.XGBRegressor(booster='gbtree').fit(train_x, train_y)
        print('\n\n --- XGBoost Test Set Results --- total nodes visited -- (MSE)): ' + str(mean_squared_error(test_y, reg.predict(test_x))) + '----- \n\n')
        self.reg = reg

        

