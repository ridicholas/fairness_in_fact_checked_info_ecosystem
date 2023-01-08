#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:49:53 2023

@author: tdn897
"""
import numpy as np
import pandas as pd
from plotnine import *
from scipy.special import softmax
import itertools
import math

a1 = 2.2
a2 = 1
a3 = 1
num_claims = 180000
def create_claims(num_claims, a1, a2, a3):

    def type_func(x):
        if x < int(num_claims/3):
            return 'anti-misinfo'
        elif x >= int(num_claims/3) and x < int((2*num_claims)/3):
            return 'noise'
        else:
            return 'misinfo'

    def virality_func(x):
        if x == 'anti-misinfo':
            return 1 + np.random.beta(a=3, b=7, size=1)[0]
        elif x == 'noise':
            return 1 + np.random.beta(a=1, b=9, size=1)[0]
        else:
            return 1.25 + np.random.beta(a=7,b=3,size=1)[0]
        
    def choice_prob_func(x, info, a1, a2, a3):
        # This control probability of starting a cascade ("utterance") about a claim.
        # anti-misinformation cascades distribution has fatter tails than misinformation, thus a1 > a3
        if info == 'anti-misinfo':
            return softmax(x**a1)
        elif info == 'noise':
            return softmax(x**a2)
        else:
            return softmax(x**a3)

    claims = pd.DataFrame(data=[i for i in range(num_claims)], columns = ['claim_id'])
    claims['type'] = claims['claim_id'].apply(type_func)
    claims['virality'] = claims['type'].apply(virality_func)

    utterance_virality = {}
    types = ['anti-misinfo', 'noise', 'misinfo']
    keys = [-1, 0, 1]
    for i in range(len(types)):
       tmp = claims.loc[(claims['type']==types[i])]
       tmp['utterance_virality'] = choice_prob_func(x=tmp['virality'],info=types[i],a1=a1,a2=a2,a3=a3)
       probs = tmp['utterance_virality'].values
       utterance_virality.update({keys[i]:probs})
    
    c = claims.set_index('claim_id')
    c_dict = c.to_dict('index')
    return c_dict, utterance_virality


def choose_claim(value, num_claims, utterance_virality):
    '''
    Within topics, there is a high-dimensional array of "potential claims". This (topic, claim) pair
    is the main feature we will use to train the fact-checking algorithm. Claims are partitioned by the quality of information
    so that we don't have agents posting {-1,0,1} all relative to the same claim.'
    Parameters
    ----------
    value : quality of informaiton {-1, 0, 1} if anti-misinformation, noise, misinformation
    Returns
    -------
    claim number : (0-33) if anti-misinfo, (34-66) if noise, (66-100) if misinfo.
    '''

    if value == -1:
        claim = np.random.choice(list(range(0,int(num_claims/3))), p=utterance_virality[value])
    elif value == 0:
        claim = np.random.choice(list(range(int(num_claims/3),int(num_claims/3)*2)), p=utterance_virality[value])
    elif value == 1:
        claim = np.random.choice(list(range(int(num_claims/3)*2,num_claims)), p=utterance_virality[value])
    return claim




claims, utterance_virality = create_claims(num_claims, a1, a2, a3)


### Plot raw distribution (just p's)
utterance_dist = pd.DataFrame()
for key in list(utterance_virality.keys()):
    if key == -1:
        info = 'anti-misinfo'
    elif key == 0:
        info = 'noise'
    else:
        info = 'misinfo'
    tmp = pd.DataFrame(utterance_virality[key],columns = ['probs'])
    tmp['Info Type'] = info
    utterance_dist = pd.concat([utterance_dist, tmp])
    



g = (ggplot(utterance_dist)
     + geom_histogram(aes(x='probs',y=after_stat('density'),fill='Info Type'),bins=1000)
     )




### Plot realized distribution



number_samples = 1000000
samples = []

for i in range(number_samples):
    info_type = np.random.choice([-1, 0, 1])
    probs = utterance_virality[info_type]
    claim = choose_claim(value=info_type, num_claims=num_claims, utterance_virality=utterance_virality)
    samples.append([info_type, claim])
    
type_ccdf = pd.DataFrame(samples, columns=['value','claim'])\
    .groupby(['value', 'claim']).size().reset_index(name='draws')\
        .groupby(['draws', "value"])['claim'].count().reset_index()\
            .sort_values(by=['value', 'draws']).rename(columns={'claim': 'Number of Occurences'})

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
 + scale_x_log10())



softmax(np.random.gamma(shape=1, scale=1, size=3000))
dist = 1 + np.random.beta(a=5, b=15, size=3000)

# tests
# for false, q99/q90=10, q90/q1 = 100, q99.9/q99=4
# for true, q99/q90=100, q90/q1 = 10, q99.9/q99=8
def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


targets = [10, 50, 2]


a_s = [i for i in range(15)]
b_s = [i for i in range(15)]
scale1 = [i for i in range(10)]
scale2 = [i for i in range(5)]
intercept = [i for i in range(10)]

d = {'a':a_s,
     'b':b_s,
     'int': intercept,
     'exp':scale1,
     'mult': scale2
     }

reps = 3
grid = pd.DataFrame(itertools.product(*d.values()),columns=d.keys())
grid = grid.loc[(grid['a']>0) & (grid['b'] > 0) & (grid['exp'] > 0) & (grid['mult'] > 0) & (grid['int'] > 0)].reset_index()
grid['Distance'] = 0
grid['q1'] = 0
grid['q2'] = 0
grid['q3'] = 0

for row in range(len(grid)):
    q1s = []
    q2s = []
    q3s = []
    distances = []
    for rep in range(reps):
        distribution = grid.loc[row, 'int'] + np.random.beta(a=grid.loc[row,'a'], b=grid.loc[row,'b'], size=5000)
        probs = softmax(grid.loc[row,'mult']*distribution**grid.loc[row,'exp'])
        q1 = np.quantile(probs, q=0.99)/np.quantile(probs,q=0.10)
        q2 = np.quantile(probs, q=0.90)/np.quantile(probs, q=0.01)
        q3 = np.quantile(probs, q=0.999)/np.quantile(probs,q=0.99)
        q1s.append(q1)
        q2s.append(q2)
        q3s.append(q3)
        distances.append(mape(targets, [q1, q2, q3]))
    grid.loc[row, 'Distance'] = np.mean(distances)
    grid.loc[row,'q1']=np.mean(q1s)
    grid.loc[row,'q2']=np.mean(q2s)
    grid.loc[row,'q3']=np.mean(q3s)
    
    

distribution = 1 + np.random.beta(a=1, b=18, size=5000)
probs = softmax(5*distribution**2)
q1 = np.quantile(probs, q=0.99)/np.quantile(probs,q=0.10)
q2 = np.quantile(probs, q=0.90)/np.quantile(probs, q=0.01)
q3 = np.quantile(probs, q=0.999)/np.quantile(probs,q=0.99)




# misinfo: a=2,b=17,exp=2,mult=3
# truth: a=1,b=17,exp=2,mult=9



