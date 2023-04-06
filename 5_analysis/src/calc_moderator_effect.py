#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:16:10 2023

@author: tdn897
"""

import networkx as nx
import numpy as np
import pandas as pd
import random
import os
from plotnine import *

os.chdir('/Users/tdn897/Desktop/NetworkFairness/fairness_in_fact_checked_info_ecosystem/5_analysis/src')

data_dir_none = 'moderator_experiment_none'
data_dir_mod = 'moderator_experiment_density'


individual_regression_none = pd.read_csv('../output/' + data_dir_none + '/individual_level_regression_data.csv')
individual_regression_density = pd.read_csv('../output/' + data_dir_mod + '/individual_level_regression_data.csv')

ate_raw_none = pd.read_csv('../output/' + data_dir_none + '/ATE_Data_raw.csv')
ate_raw_density = pd.read_csv('../output/' + data_dir_mod + '/ATE_Data_raw.csv')

ate_none = ate_raw_none.groupby(['Community', 'Intervention'], as_index=False).\
    agg(ATE_est_mean_none = ('ATE_est', 'mean'), ATE_est_sd_none = ('ATE_est', 'std'))
ate_none['version'] = 'No Moderators'


ate_density = ate_raw_density.groupby(['Community', 'Intervention'], as_index=False).\
    agg(ATE_est_mean_density = ('ATE_est', 'mean'), ATE_est_sd_density = ('ATE_est', 'std'))
ate_density['version'] = 'Reduce Density'


majority = 3
minority = 49


minority_none = ate_none.loc[ate_none['Community'] == minority]
majority_none = ate_none.loc[ate_none['Community'] == majority]
none_stat = pd.merge(minority_none, majority_none, on=['Intervention'])
none_stat['Difference in Benefit'] = none_stat['ATE_est_mean_none_y'] - none_stat['ATE_est_mean_none_x']
none_stat['Joint_SE'] = (none_stat['ATE_est_sd_none_x'] + none_stat['ATE_est_sd_none_y'])/2
none_stat['Moderator'] = 'No Moderator'

minority_density = ate_density.loc[ate_density['Community'] == minority]
majority_density = ate_density.loc[ate_density['Community'] == majority]
density_stat = pd.merge(minority_density, majority_density, on=['Intervention'])
density_stat['Difference in Benefit'] = density_stat['ATE_est_mean_density_y'] - density_stat['ATE_est_mean_density_x']
density_stat['Joint_SE'] = (density_stat['ATE_est_sd_density_x'] + density_stat['ATE_est_sd_density_y'])/2
density_stat['Moderator'] = 'Reduce Density'

stat = pd.concat([none_stat, density_stat])[['Intervention', 'Moderator', 'Difference in Benefit', 'Joint_SE']]




ate = pd.merge(ate_none, ate_density, on=['Community', 'Intervention'])
ate['Difference (Majority Group']


ate = pd.concat([ate_none, ate_density])
ate['Community'] = np.where(ate['Community']==3, 'Majority Community',
                            np.where(ate['Community']==49, 'Minority Community',
                                     'Knowledgeable Community'))


