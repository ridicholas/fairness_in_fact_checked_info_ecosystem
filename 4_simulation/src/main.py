#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:54:39 2022

@author: tdn897
"""

from checkworthy import Checkworthy
from TopicSim import TopicSim
import yaml
import pickle

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)



'''
First topic (row) impacts every group the same, the other topics each impact
one group significantly more than the others
'''


impactednesses = [{3: 0.5, 56: 0.5, 43: 0.5},
                  {3: 0.8, 56: 0.3, 43: 0.3},
                  {3: 0.3, 56: 0.8, 43: 0.3},
                  {3: 0.3, 56: 0.3, 43: 0.8}]

'''
For the first topic (which everyone cares about equally), belief is roughly average (50%).
For the other topics, if a community is more impacted by a topic, we assume that their
average belief is lower, indicating that they have more knowledge of the truth than the
other communities that are not impacted  by the topic.
'''
beliefs = [{3: 0.5, 56: 0.5, 43: 0.5},
              {3: 0.3, 56: 0.8, 43: 0.8},
              {3: 0.8, 56: 0.3, 43: 0.8},
              {3: 0.8, 56: 0.8, 43: 0.3}]





check = Checkworthy(
    agg_interval = config['agg_interval'],
    agg_steps = config['agg_steps'],
    depths = config['depths'],
    impactednesses = impactednesses
)


sim = TopicSim(
    impactedness = impactednesses,
    beliefs = beliefs,
    num_topics = config['num_topics'],
    runtime = config['runtime'],
    communities = config['communities'],
    num_claims = config['num_claims']
)

'''
Import or create networks.
If create, nodes in network must already have 'Community' attribtue
'''

if config['load_data']:
    sim.load_simulation_network(ready_network_path = config['ready_network_path'])
else:
    sim.create_simulation_network(
        raw_network_path = config['raw_network_path'],
        perc_nodes_to_subset = config['perc_nodes_to_subset'],
        perc_bots = config['perc_bots']
    )


check.set_network(G = sim.return_network())
sim.set_check(check = check)


'''
Run 1: pre-period 
'''

sim.run(
    period = 'pre', 
    learning_rate = config['learning_rate'], 
    fact_checks_per_step = config['fact_checks_per_step'], 
    mitigation_type = 'None'
)


with open(config['output_sim_midpoint'], 'wb') as file:
    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)
 

del sim
del check


'''
Run 2: Post-period, no intervention
'''

with open(config['output_sim_midpoint'], 'rb') as file:
    sim = pickle.load(file)

sim.set_post_duration(config['post_duration'])

sim.run(
    period = 'post', 
    learning_rate = config['learning_rate'], 
    fact_checks_per_step = config['fact_checks_per_step'], 
    mitigation_type = 'None'
)

with open(config['output_sim_final_no_intervention'], 'wb') as file:
    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)

del sim


'''
Run 3: Post-period, checkworthy intervention, label = random
'''

with open(config['output_sim_midpoint'], 'rb') as file:
    sim = pickle.load(file)

sim.set_post_duration(config['post_duration'])

check_pre = sim.return_check()

print('\n\n\n\n\n ----------- Sampling Claims for Checkworthy Dataset --------- \n\n\n\n\n')
check_pre.sample_claims(num_to_sample=config['claims_to_sample'], sample_method='top_avg_origin_degree')

print('\n\n\n\n\n ----------- Random Sampling of Labels for Checkworthy Dataset --------- \n\n\n\n\n')
check_pre.sample_labels_for_claims(labels_per_claim = config['nodes_to_sample'], sample_method = 'random')

print('\n\n\n\n\n ----------- Training Model with Label = average_truth_perception_random --------- \n\n\n\n\n')
check_pre.train_model(label_to_use='average_truth_perception_random')

sim.set_check(check=check_pre)

sim.run(
    period = 'post', 
    learning_rate = config['learning_rate'], 
    fact_checks_per_step = config['fact_checks_per_step'], 
    mitigation_type = 'stop_reading_misinfo'
)

with open(config['output_sim_final_intervention_random'], 'wb') as file:
    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)

del sim

'''
Run 4: Post-period, checkworthy intervention, label = knowledgable community
'''

with open(config['output_sim_midpoint'], 'rb') as file:
    sim = pickle.load(file)

sim.set_post_duration(config['post_duration'])

check_pre = sim.return_check()

print('\n\n\n\n\n ----------- Sampling Claims for Checkworthy Dataset --------- \n\n\n\n\n')
check_pre.sample_claims(num_to_sample=config['claims_to_sample'], sample_method='top_avg_origin_degree')

print('\n\n\n\n\n ----------- Random Sampling of Labels for Checkworthy Dataset --------- \n\n\n\n\n')
check_pre.sample_labels_for_claims(labels_per_claim = config['nodes_to_sample'], sample_method = 'knowledgable_community')

print('\n\n\n\n\n ----------- Training Model with Label = average_truth_perception_random --------- \n\n\n\n\n')
check_pre.train_model(label_to_use='average_truth_perception_knowledgable_community')

sim.set_check(check=check_pre)

sim.run(
    period = 'post', 
    learning_rate = config['learning_rate'], 
    fact_checks_per_step = config['fact_checks_per_step'], 
    mitigation_type = 'stop_reading_misinfo'
)

with open(config['output_sim_final_intervention_kc'], 'wb') as file:
    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)

del sim

