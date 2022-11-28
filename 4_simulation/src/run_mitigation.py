import pandas as pd
import networkx as nx
import random
import numpy as np
from numpy.random import choice
import progressbar
import os
from scipy.stats import beta, rankdata
import pickle
from checkworthy import *
from sim_util import *
# making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import xgboost as xg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error



path = '../../data/nodes_with_community.gpickle'
inpath_info = '../output/all_info.pickle'
inpath_node_info = '../output/node_info.pickle'
inpath_community_sentiment = '../output/community_sentiment.pickle'
inpath_node_time_info = '../output/node_time_info.pickle'
inpath_checkworthy = '../output/checkworthy_data.pickle'
subset_graph_file = '../output/simulation_net.gpickle'
inpath_node_data_info = '../output/node_metadata.gpickle'
inpath_stored_model = '../output/stored_model.pickle'
inpath_all_claims = '../output/all_claims.pickle'


outpath_info = '../output/mitigation/all_info.pickle'
outpath_node_info = '../output/mitigation/node_info.pickle'
outpath_community_sentiment = '../output/mitigation/community_sentiment.pickle'
outpath_node_time_info = '../output/mitigation/node_time_info.pickle'
outpath_checkworthy = '../output/mitigation/checkworthy_data.pickle'
subset_graph_file = '../output/mitigation/simulation_net.gpickle'
outpath_node_data_info = '../output/mitigation/node_metadata.gpickle'
outpath_stored_model = '../output/mitigation/stored_model.pickle'
outpath_all_claims = '../output/mitigation/all_claims.pickle'

nodes_to_sample = 100

num_topics = 4
communities_to_subset = [3,56,43]
learning_rate = 0.2
NUM_CLAIMS = 6000 #this is number of claims per topic per timestep
runtime = 300
agg_interval = 3
agg_steps = 3
perc_nodes_to_subset = 0.1
perc_bots = 0.1
load_data = False
update_beliefs = True
depths = [2,4,6]
outcome_time = 48
fact_checks_per_step = 5
MIN_DEGREE = 10


print('\n\n\n\n\n ----------- Load State After First Half of Simulation --------- \n\n\n\n\n')

with open(inpath_info, 'rb') as file:
    all_info = pickle.load(file)

with open(inpath_all_claims, 'rb') as file:
    all_claims = pickle.load(file)

with open(inpath_node_info, 'rb') as file:
    node_read_tweets = pickle.load(file)

with open(inpath_community_sentiment, 'rb') as file:
    community_sentiment_through_time = pickle.load(file)

with open(inpath_node_time_info, 'rb') as file:
    node_read_tweets_by_time = pickle.load(file)

with open(inpath_checkworthy, 'rb') as file:
    check = pickle.load(file)
    
G = nx.read_gpickle(inpath_node_data_info)



print('\n\n\n\n\n ----------- Sampling Claims for Checkworthy Dataset --------- \n\n\n\n\n')
check.sample_claims(num_to_sample=2000, sample_method='top_avg_origin_degree')


print('\n\n\n\n\n ----------- Sampling Labels for Checkworthy Dataset --------- \n\n\n\n\n')


check.sample_labels_for_claims(labels_per_claim =nodes_to_sample, sample_method = 'random')
check.sample_labels_for_claims(labels_per_claim =nodes_to_sample, sample_method = 'stratified')
check.sample_labels_for_claims(labels_per_claim =nodes_to_sample, sample_method = 'knowledgable_community')




print('\n\n\n\n\n ----------- Training Checkworthy Model --------- \n\n\n\n\n')


check_df = pd.DataFrame.from_dict(check.sampled_checkworthy_data).T.fillna(0)
check_df['target'] = check_df['average_truth_perception_stratified']


train, test = train_test_split(check_df, test_size=0.2)
train_x = train[[i for i in train.columns if ('truth' not in i) and ('claim' not in i) and ('target' not in i) and ('outcome' not in i) and ('value' not in i)]]
train_y = train[['target']]

test_x = test[[i for i in test.columns if ('truth' not in i) and ('claim' not in i) and ('target' not in i) and ('outcome' not in i) and ('value' not in i)]]
test_y = test[['target']]

clf = xg.XGBRegressor().fit(train_x, train_y)
print(mean_squared_error(test_y, clf.predict(test_x)))

#print(confusion_matrix(test_y, clf.predict(test_x)))
#plt.scatter(test_y, clf.predict(test_x))
#plt.xlabel('test_y')
#plt.ylabel('pred_y')
#plt.show()



print('\n\n\n\n\n ----------- Running second half of simulation --------- \n\n\n\n\n')

def run_second_half(G, runtime, agg_interval=3, agg_steps=3, outcome_time=48, mitigation_type = "delete_from_inbox"):
    '''
    mitigation_types: "delete_from_inbox", "stop_reading_misinfo", "None", "change_belief_updates"
    '''

    prestige_values = percentile(list(dict(G.in_degree()).values()))
    nodes = list(G.nodes())
    prestige = {nodes[i]: prestige_values[i] for i in range(len(nodes))}
    topics = list(range(num_topics))
    for node in nodes:
        node_read_tweets_by_time[node].update({t: [] for t in range(runtime, runtime*2)})
    for com in communities_to_subset:
        community_sentiment_through_time[com].update({t:{topic: [] for topic in range(num_topics)} for t in range(runtime, runtime*2)})

    
   

    def new_tweets_func():
        all_info.update({unique_id: {'topic':topic,'value':value,'claim':claim,'node-origin':node,'time-origin':step}})
        new_tweets.append(unique_id)
        '''
        update checkworthy data
        '''
        check.intake_information(node = node, data = data, claim_id = claim_id, value = value, topic = topic, claim = claim)
        if claim_id not in check.checkworthy_data.keys():
            check.update_keys()
        else:
            check.update_agg_values()

    def read_tweets_func():
        virality = all_claims[read_claim]['virality']
        topic_sentiment = data['sentiment'][topic]
        creator_prestige = prestige[all_info[read_tweet]['node-origin']]

        '''
        update beliefs for each topic
        '''
        if update_beliefs and data['kind'] != 'bot':
            data['num_read'][topic] += 1
            data['sentiment'][topic] = update_topic_sentiment(current_sentiment=data['sentiment'][topic],
                                                            tweet_value = update_value,
                                                            tweet_impactedness=data['impactedness'][topic],
                                                            num_read = data['num_read'][topic],
                                                            learning_rate = learning_rate)
        '''
        retweet behavior
        '''
        perc = retweet_behavior(topic = topic,
                                value=value,
                                topic_sentiment=topic_sentiment,
                                creator_prestige=creator_prestige,
                                claim_virality=virality)

        retweet_perc.append(perc)
        new_retweets.append(read_tweet)
        node_read_tweets[node].append(read_tweet)
        node_read_tweets_by_time[node][step].append(read_tweet)
        time_feature = int((step - all_info[read_tweet]['time-origin'])/agg_interval)+1
        if time_feature <= agg_steps:
            origin_node = read_tweet.split('-')[2]
            claim_id = read_tweet.split('-')[0] + '-' + read_tweet.split('-')[1]
            check.intake_information(node = node, data = data, claim_id = claim_id, value = value, topic = topic, claim = read_claim)
            check.update_time_values(time_feature=time_feature, origin_node=origin_node)
        time_from_launch = step - all_info[read_tweet]['time-origin']
        if time_from_launch <= outcome_time:
            claim_id = read_tweet.split('-')[0] + '-' + read_tweet.split('-')[1]
            check.intake_information(node = node, data = data, claim_id = claim_id, value = value, topic = topic, claim = read_claim)
            check.update_virality_outcome(time_from_launch=time_from_launch)


    bar = progressbar.ProgressBar()
    fact_checked = []
    for step in bar(range(runtime, runtime*2)):
        # Loop over all nodes
        '''
        Users and Information interact
        '''
        
        ##for each time step, determine which claims to fact check using classifier
        check_df = pd.DataFrame.from_dict(check.checkworthy_data).T.fillna(0)
        x = check_df[[i for i in check_df.columns if ('truth' not in i) and ('target' not in i) and ('claim' not in i) and ('outcome' not in i) and ('value' not in i)]]
        preds = pd.Series(clf.predict(x), index=check_df.index)
        preds = preds.drop(fact_checked)
        preds.sort_values(ascending=False, inplace=True)

        

        fact_checked = fact_checked + list(preds.index[0:fact_checks_per_step])
        fact_checked = [*set(fact_checked)]
        
        
        rankings = calculate_sentiment_rankings(G = G, topics = topics)
        for node, data in G.nodes(data=True):
            if mitigation_type == 'delete_from_inbox':
                #delete misinfo from inbox
                data['inbox'] = [i for i in data['inbox'] if ( '-'.join(i.split('-')[0:2]) not in fact_checked) and (all_info[i]['value'] != 1)]
                #add antimisinfo to beginning of inbox
                for fc in fact_checked:
                    random_tweet_of_topicclaim = random.choice([i for i, v in all_info.items() if v['topic'] == fc.split('-')[0] and v['claim'] == fc.split('-')[1]])
                    data['inbox'].insert(0, random_tweet_of_topicclaim)
            
            if data['wake'] == step:
                data['wake'] = data['wake'] + \
                    np.round(1 + np.random.exponential(scale=1 / data['lambda']))

                '''
                Create tweets
                '''

                if data['kind'] == 'bot':
                    chance = 1
                elif data['kind'] != 'bot':
                    chance = prestige[node]

                new_tweets = []
                if chance > np.random.uniform():
                    num_tweets = np.random.randint(1,10)
                    for i in range(num_tweets):
                        topic = choose_topic(data = data)
                        value = choose_info_quality(node = node, rankings = rankings, topic = topic, agent_type = data['kind'])
                        claim = choose_claim(value = value, num_claims=NUM_CLAIMS)
                        unique_id = str(topic) + '-' + str(claim) + '-' + str(node) + '-' + str(step)
                        claim_id = str(topic) + '-' + str(claim)
                        
                        if mitigation_type == 'stop_reading_misinfo':
                            #if this claim has been fact checked as misinformation, everyone stops reading/tweeting/believing them
                            if not ((str(topic) + '-' + str(claim) in fact_checked) and (value == 1)):
                                new_tweets_func()
                        else:
                            new_tweets_func()
                        


                '''

                Read tweets, update beliefs, and re-tweet

                '''
                retweets = []
                if len(data['inbox']) > 0:
                    number_to_read = min(random.randint(4, 20), len(data['inbox'])) #should this be fully random?
                    read_tweets = data['inbox'][-number_to_read:]
                    retweet_perc = []
                    new_retweets = []
                    for read_tweet in read_tweets:
                        if read_tweet not in node_read_tweets[node]:
                            topic = all_info[read_tweet]['topic']
                            value = all_info[read_tweet]['value']
                            update_value = value
                            read_claim = all_info[read_tweet]['claim']
                            #if this claim has been fact checked as misinformation, everyone stops reading/tweeting/believing them
                            if mitigation_type == "stop_reading_misinfo":
                                if not ((str(topic) + '-' + str(read_claim) in fact_checked) and (value == 1)):
                                    read_tweets_func()
                            elif mitigation_type == 'change_belief_updates':
                                if value == 1:
                                    update_value = 5
                                if value == -1:
                                    update_value = -5
                                read_tweets_func()
                            else:
                                read_tweets_func()

                    for i in range(len(new_retweets)):
                        if retweet_perc[i] > np.random.uniform():
                            retweets.append(new_retweets[i])
                    data['inbox'] = []

                '''
                Pass information on to followers
                '''
                new_tweets.extend(retweets)
                if len(new_tweets) > 0:
                    predecessors = G.predecessors(node)
                    for follower in predecessors:
                        G.nodes[follower]['inbox'].extend(new_tweets)

                '''
                Capture sentiment across topics for node
                '''
                for topic in range(num_topics):
                    community_sentiment_through_time[data['Community']][step][topic].append(data['sentiment'][topic])


    return all_info, node_read_tweets, community_sentiment_through_time, node_read_tweets_by_time, check, G, all_claims




all_info, node_read_tweets, community_sentiment_through_time, node_read_tweets_by_time, check, G, all_claims = run_second_half(G = G,
                                                                                                    runtime = runtime,
                                                                                                    agg_interval=agg_interval,
                                                                                                    agg_steps=agg_steps,
                                                                                                    outcome_time=outcome_time,
                                                                                                    mitigation_type = "change_belief_updates")


print('\n\n\n\n\n ----------- Writing Data --------- \n\n\n\n\n')

with open(outpath_info, 'wb') as file:
    pickle.dump(all_info, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_all_claims, 'wb') as file:
    pickle.dump(all_claims, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_node_info, 'wb') as file:
    pickle.dump(node_read_tweets, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_community_sentiment, 'wb') as file:
    pickle.dump(community_sentiment_through_time, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_node_time_info, 'wb') as file:
    pickle.dump(node_read_tweets_by_time, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath_checkworthy, 'wb') as file:
    pickle.dump(check, file, protocol=pickle.HIGHEST_PROTOCOL)
    
nx.write_gpickle(G, outpath_node_data_info)