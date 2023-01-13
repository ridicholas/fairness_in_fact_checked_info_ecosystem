import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import progressbar
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf

# making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

comms = '12_28_56_43'
regression_infile = '../output/regression' +  comms

data = pd.read_pickle(regression_infile + '.pickle')
#convert to numerics
cols = data.columns.drop('mod')

data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
    

#basic regression
print('pause')

results = smf.ols('change_in_belief ~ C(mod, Treatment(reference="no_intervention_"))', data=data).fit()

results = smf.ols('change_in_belief ~ size + perc_of_largest*C(mod, Treatment(reference="no_intervention_")) +  \
       avg_degree_within_graph + avg_degree_within_community + density + \
       cluster_coef + average_centrality_of_nodes + comm_centrality + \
       impactedness + start_belief + perc_of_net + \
       ratio_connections_self_to_largest_self + \
       ratio_connections_self_to_largest_largest', data=data).fit()



