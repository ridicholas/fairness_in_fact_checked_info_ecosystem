#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:57:48 2022

@author: tdn897
"""
import progressbar
import numpy as np
import pandas as pd
import random
from scipy.stats import beta
from random import choice

class TopicSim():

    def __init__(self, impactedness, beliefs, num_topics, runtime, communities, num_claims):

        self.impactedness = impactedness
        self.beliefs = beliefs
        self.num_topics = num_topics
        self.runtime = runtime
        self.communities = communities
        self.num_claims = num_claims
        self.all_info = {}
        self.node_read_tweets = {}
        self.community_sentiment_through_time = {}
        self.community_read_tweets_by_type = {}
        self.community_checked_tweets_by_type = {}
        self.node_read_tweets_by_time = {}
        self.set_comm_string()
        self.start_network_path = ''
        

    def set_impactedness(self, impactedness):
        self.impactedness = impactedness

    def set_beliefs(self, beliefs):
        self.beliefs = beliefs

    def set_network(self, G):
        self.G = G

    def return_network(self):
        return self.G

    def set_check(self, check):
        self.check = check

    def return_check(self):
        return self.check
    
    def return_communities(self):
        return self.communities

    def set_post_duration(self, post_duration):
        self.post_duration = post_duration
    
    def set_start_network_path(self, path):
        self.start_network_path = path

    def set_comm_string(self):
        comm_string = ''
        for comm in self.communities:
            comm_string += '_' + str(comm)

        self.comm_string = comm_string


    def load_simulation_network(self, ready_network_path):
        import pickle
        with open(ready_network_path, "rb") as f:
            self.G = pickle.load(f)
        
        self.set_start_network_path(ready_network_path)

    # This assumes a "raw" networkx gpickle where each node has one attribtue: "Community".
    # We ran Louvain community detection algorithm to create this attribute
    def create_simulation_network(self, raw_network_path, perc_nodes_to_subset, perc_bots):
        import networkx as nx
        G = nx.read_gpickle(raw_network_path)
        subG = self.subset_graph(G, communities=self.communities)
        sampleG = self.set_node_attributes(G=subG,
                                           perc_nodes_to_use = perc_nodes_to_subset,
                                           numTopics = self.num_topics,
                                           perc_bots = perc_bots,
                                           impactednesses = self.impactedness,
                                           sentiments = self.beliefs)
        self.G = sampleG
        

        self.start_network_path = '../output/simulation_net_communities{}.gpickle'.format(self.comm_string)
        nx.write_gpickle(self.G, '../output/simulation_net_communities{}.gpickle'.format(self.comm_string))
        




    def run(self, period, learning_rate, min_degree, fact_checks_per_step, mitigation_type):


        G = self.G
        check = self.check
        in_degree = list(dict(G.in_degree()).values())
        prestige_values = self.percentile(in_degree)
        nodes = list(G.nodes())
        prestige = {nodes[i]: prestige_values[i] for i in range(len(nodes))}
        topics = list(range(self.num_topics))



        if period == 'pre':
            time = range(self.runtime)

            # Initialize objects to collect results
            all_info = {}
            # This will capture the unique-ids of each tweet read by each node.
            node_read_tweets = {node:[] for node in nodes}
            community_sentiment_through_time = {com:{t:{topic: [] for topic in range(self.num_topics)} for t in range(self.runtime)} for com in self.communities}
            community_read_tweets_by_type = {com:{t:{topic:{'misinfo': 0, 'noise': 0, 'anti-misinfo': 0} for topic in range(self.num_topics)} for t in range(self.runtime)} for com in self.communities}
            node_read_tweets_by_time = {node:{t: [] for t in range(self.runtime)} for node in nodes}
            all_claims = self.create_claims(num_claims = self.num_claims)
            community_checked_tweets_by_type = {}
            

        elif period == 'post':

            all_info = self.all_info
            node_read_tweets = self.node_read_tweets
            community_sentiment_through_time = self.community_sentiment_through_time
            node_read_tweets_by_time = self.node_read_tweets_by_time
            all_claims = self.all_claims
            community_read_tweets_by_type = self.community_read_tweets_by_type
            community_checked_tweets_by_type = {com:{t:{topic:{'misinfo': 0, 'noise': 0, 'anti-misinfo': 0} for topic in range(self.num_topics)} for t in range(self.runtime, self.runtime*self.post_duration)} for com in self.communities}



            time = range(self.runtime, self.runtime*self.post_duration)
            for node in nodes:
                node_read_tweets_by_time[node].update({t: [] for t in range(self.runtime, self.runtime*self.post_duration)})
            for com in self.communities:
                community_sentiment_through_time[com].update({t:{topic: [] for topic in range(self.num_topics)} for t in range(self.runtime, self.runtime*self.post_duration)})
                community_read_tweets_by_type[com].update({t:{topic:{'misinfo': 0, 'noise': 0, 'anti-misinfo': 0} for topic in range(self.num_topics)} for t in range(self.runtime, self.runtime*self.post_duration)})
                community_checked_tweets_by_type[com].update({t:{topic:{'misinfo': 0, 'noise': 0, 'anti-misinfo': 0} for topic in range(self.num_topics)} for t in range(self.runtime, self.runtime*self.post_duration)})



        bar = progressbar.ProgressBar()
        fact_checked = []
        for step in bar(time):
                # Loop over all nodes
            '''
            Users and Information interact
            '''
            if period == 'post' and mitigation_type != 'None':
            ##for each time step, determine which claims to fact check using classifier
                check_df = pd.DataFrame.from_dict(self.check.checkworthy_data).T.fillna(0)
                x = check_df[[i for i in check_df.columns if ('truth' not in i) and ('target' not in i) and ('claim' not in i) and ('outcome' not in i) and ('value' not in i)]]
                x = x[check.cols_when_model_builds]
                preds = pd.Series(check.clf.predict(x), index=check_df.index)
                preds = preds.drop(fact_checked)
                preds.sort_values(ascending=False, inplace=True)
                fact_checked = fact_checked + list(preds.index[0:fact_checks_per_step])
                fact_checked = [*set(fact_checked)]


            rankings = self.calculate_sentiment_rankings(G = G, topics = topics)

            for node, data in G.nodes(data=True):

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
                            topic = self.choose_topic(data = data)
                            value = self.choose_info_quality(node = node, rankings = rankings, topic = topic, agent_type = data['kind'])
                            claim = self.choose_claim(value = value, num_claims=self.num_claims)
                            unique_id = str(topic) + '-' + str(claim) + '-' + str(node) + '-' + str(step)
                            claim_id = str(topic) + '-' + str(claim)

                            if mitigation_type == 'stop_reading_misinfo':
                                #if this claim has been fact checked as misinformation, everyone stops reading/tweeting/believing them
                                if not ((str(topic) + '-' + str(claim) in fact_checked) and (value == 1)):
                                    all_info.update({unique_id: {'topic':topic,'value':value,'claim':claim,'node-origin':node,'time-origin':step}})
                                    new_tweets.append(unique_id)
                                    '''
                                    update checkworthy data
                                    '''
                                    check.intake_information(node = node, data = data, claim_id = claim_id, value = value, topic = topic, claim = claim)
                                    if data['degree'] >= min_degree:
                                        if claim_id not in check.checkworthy_data.keys():
                                            check.update_keys()
                                        else:
                                            check.update_agg_values()

                            else:
                                all_info.update({unique_id: {'topic':topic,'value':value,'claim':claim,'node-origin':node,'time-origin':step}})
                                new_tweets.append(unique_id)
                                '''
                                update checkworthy data
                                '''
                                check.intake_information(node = node, data = data, claim_id = claim_id, value = value, topic = topic, claim = claim)
                                if data['degree'] >= min_degree:
                                    if claim_id not in check.checkworthy_data.keys():
                                        check.update_keys()
                                    else:
                                        check.update_agg_values()


                    '''
                    Read tweets, update beliefs, and re-tweet
                    '''
                    retweets = []
                    if len(data['inbox']) > 0:
                        number_to_read = min(np.random.randint(4, 20), len(data['inbox'])) #should this be fully random?
                        read_tweets = data['inbox'][-number_to_read:]
                        retweet_perc = []
                        new_retweets = []
                        for read_tweet in read_tweets:
                            if read_tweet not in node_read_tweets[node]:
                                topic = all_info[read_tweet]['topic']
                                value = all_info[read_tweet]['value']
                                read_claim = all_info[read_tweet]['claim']

                                #if claim is fact checked, add it to checked by type tracker
                                if (str(topic) + '-' + str(read_claim)) in fact_checked:
                                    community_checked_tweets_by_type = self.update_read_counts(community_read_tweets_by_type = community_checked_tweets_by_type,
                                                                                                topic = topic,
                                                                                                info_type = value,
                                                                                                com = data['Community'],
                                                                                                step = step)

                                #if this claim has been fact checked as misinformation, everyone stops reading/tweeting/believing them
                                if mitigation_type == "stop_reading_misinfo":
                                    if not ((str(topic) + '-' + str(read_claim) in fact_checked) and (value == 1)):
                                        virality = all_claims[read_claim]['virality']
                                        topic_sentiment = data['sentiment'][topic]
                                        creator_prestige = prestige[all_info[read_tweet]['node-origin']]

                                        '''
                                        update beliefs for each topic
                                        '''
                                        if data['kind'] != 'bot':
                                            data['num_read'][topic] += 1
                                            data['sentiment'][topic] = self.update_topic_sentiment(current_sentiment=data['sentiment'][topic],
                                                                                                   tweet_value = value,
                                                                                                   tweet_impactedness=data['impactedness'][topic],
                                                                                                   num_read = data['num_read'][topic],
                                                                                                   learning_rate = learning_rate)
                                        '''
                                        retweet behavior
                                        '''
                                        perc = self.retweet_behavior(topic = topic,
                                                                     value=value,
                                                                     topic_sentiment=topic_sentiment,
                                                                     creator_prestige=creator_prestige,
                                                                     claim_virality=virality)

                                        retweet_perc.append(perc)
                                        new_retweets.append(read_tweet)
                                        node_read_tweets[node].append(read_tweet)
                                        node_read_tweets_by_time[node][step].append(read_tweet)
                                        time_feature = int((step - all_info[read_tweet]['time-origin'])/check.agg_interval)+1
                                        if time_feature <= check.agg_steps:
                                            origin_node = read_tweet.split('-')[2]
                                            claim_id = read_tweet.split('-')[0] + '-' + read_tweet.split('-')[1]
                                            check.intake_information(node = node, data = data, claim_id = claim_id, value = value, topic = topic, claim = read_claim)
                                            if data['degree'] >= min_degree:

                                                if claim_id not in check.checkworthy_data.keys():
                                                    check.update_keys()
                                                else:
                                                    check.update_time_values(time_feature=time_feature, origin_node=origin_node)

                                        community_read_tweets_by_type = self.update_read_counts(community_read_tweets_by_type = community_read_tweets_by_type,
                                                                                                topic = topic,
                                                                                                info_type = value,
                                                                                                com = data['Community'],
                                                                                                step = step)
                                else:
                                    virality = all_claims[read_claim]['virality']
                                    topic_sentiment = data['sentiment'][topic]
                                    creator_prestige = prestige[all_info[read_tweet]['node-origin']]

                                    '''
                                    update beliefs for each topic
                                    '''
                                    if data['kind'] != 'bot':
                                        data['num_read'][topic] += 1
                                        data['sentiment'][topic] = self.update_topic_sentiment(current_sentiment=data['sentiment'][topic],
                                                                                               tweet_value = value,
                                                                                               tweet_impactedness=data['impactedness'][topic],
                                                                                               num_read = data['num_read'][topic],
                                                                                               learning_rate = learning_rate)
                                    '''
                                    retweet behavior
                                    '''
                                    perc = self.retweet_behavior(topic = topic,
                                                                 value=value,
                                                                 topic_sentiment=topic_sentiment,
                                                                 creator_prestige=creator_prestige,
                                                                 claim_virality=virality)

                                    retweet_perc.append(perc)
                                    new_retweets.append(read_tweet)
                                    node_read_tweets[node].append(read_tweet)
                                    node_read_tweets_by_time[node][step].append(read_tweet)
                                    time_feature = int((step - all_info[read_tweet]['time-origin'])/check.agg_interval)+1
                                    if time_feature <= check.agg_steps:
                                        origin_node = read_tweet.split('-')[2]
                                        claim_id = read_tweet.split('-')[0] + '-' + read_tweet.split('-')[1]
                                        check.intake_information(node = node, data = data, claim_id = claim_id, value = value, topic = topic, claim = read_claim)
                                        if data['degree'] >= min_degree:
                                            if claim_id not in check.checkworthy_data.keys():
                                                check.update_keys()
                                            else:
                                                check.update_time_values(time_feature=time_feature, origin_node=origin_node)

                                    community_read_tweets_by_type = self.update_read_counts(community_read_tweets_by_type = community_read_tweets_by_type,
                                                                                            topic = topic,
                                                                                            info_type = value,
                                                                                            com = data['Community'],
                                                                                            step = step)

                            # update read counts by type of info



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
                    for topic in range(self.num_topics):
                        community_sentiment_through_time[data['Community']][step][topic].append(data['sentiment'][topic])

        
        if period == 'pre':
            self.all_info = all_info
            self.node_read_tweets_by_time = node_read_tweets_by_time
            self.node_read_tweets = node_read_tweets
            self.all_claims = all_claims
        else:
            self.all_info = 'Removed for light storage'
            self.node_read_tweets = 'Removed for light storage'
            self.node_read_tweets_by_time = 'Removed  for light storage'
            self.all_claims = 'Removed for light storage'
        
        # These objects are used in process_data.py
        check.set_network(G=G, communities=self.communities)
        self.G = G
        self.check = check
        self.community_sentiment_through_time = community_sentiment_through_time
        self.community_read_tweets_by_type = community_read_tweets_by_type
        self.community_checked_tweets_by_type = community_checked_tweets_by_type
        

        

    def choose_topic(self, data):
        topic_probs = [i / sum(data['impactedness'].values()) for i in data['impactedness'].values()]
        topic = np.random.choice(np.arange(0, len(topic_probs)), p=topic_probs)
        return topic

    def choose_info_quality(self, node, rankings, topic, agent_type):

        '''
        For each (non-bot) type, we draw from a beta distribution with beta(5 - a, 5 + a), and we shift the parameters according
        to their percentile rankings of sentiment deviation from the mean, such that those with low sentiment
        produce more anti-misinformation, and those with high sentiment produce more misinformation, but noise is always the most common
        info type produced.

        Because B(a,b) is bounded by (0, 1), we can just use thirds as cut points to effectively give different
        probabilistic weight to information quality in {-1, 0, 1}.

        Bots produce misinformation 80% of the time
        '''
        if agent_type != 'bot':
            deviation_rank = rankings[str(node)][topic]
            raw = beta.rvs(5 + deviation_rank, 5 - deviation_rank, size=1)
            value = np.where(raw < 0.3333, -1, np.where(raw >= 0.3333 and raw < 0.666, 0, 1))[0]
        else:
            value = np.where(np.random.uniform(size=1) > 0.2, 1, 0)[0]
        return value

    def create_claims(self, num_claims):

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
                return 1 + np.random.beta(a=6,b=4,size=1)[0]

        claims = pd.DataFrame(data=[i for i in range(num_claims)], columns = ['claim_id'])
        claims['type'] = claims['claim_id'].apply(type_func)
        claims['virality'] = claims['type'].apply(virality_func)

        c = claims.set_index('claim_id')
        c_dict = c.to_dict('index')
        return c_dict

    def calculate_sentiment_rankings(self, G, topics):

        import networkx as nx
        '''
        This function returns a pandas DataFrame with all nodes' percentile rankings of deviation from mean sentiment across all topics.
        This ranking is multiplied by -1 if they have a negative deviation and by +1 if they have a positive deviation,
        creating a range of possible values [-1,1].

        This pandas dataframe is used as an input to modify the distribution from which agents draw their quality of information when tweeting.
        A higher rank value in the dataframe results in a higher probability of creating misinformation.
        This should be intuitive... if someone's sentiment is already high, they are
        more likely to create misinformation. If someone's sentiment is low, they are more likely to produce anti-misinformation.

        One potential issue here is if sentiment is tightly clustered for all agents, this will sort of artificially make some agents produce more/less misinformation in that case.
        '''
        all_node_sentiments = nx.get_node_attributes(G, 'sentiment')
        node_sentiments = [[key, all_node_sentiments[key][topic], topic] for key in all_node_sentiments.keys() for topic in topics]
        
        sentiments = pd.DataFrame(node_sentiments, columns = ['node', 'sentiment', 'topic']).sort_values(by=['topic', 'node'])
        
        median_sentiment = np.median(sentiments.sentiment.to_list())
        deviations = [np.absolute(i - median_sentiment) for i in sentiments.sentiment.to_list()]
        sentiments['deviations'] = deviations
        sentiments['rank'] = np.where(sentiments['sentiment'] < median_sentiment,
                                                     -1*sentiments['deviations'].rank(method='max')/len(sentiments),
                                                     sentiments['deviations'].rank(method='max')/len(sentiments))
        
        sentiments_wide = pd.pivot(sentiments, index='node', columns = 'topic', values = 'rank').to_dict(orient='index')

        return sentiments_wide






    def choose_claim(self, value, num_claims):
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
            claim = random.sample(list(range(0,int(num_claims/3))), k=1)[0]
        elif value == 0:
            claim = random.sample(list(range(int(num_claims/3),int(num_claims/3)*2)), k=1)[0]
        elif value == 1:
            claim = random.sample(list(range(int(num_claims/3)*2,num_claims)), k=1)[0]
        return claim


    def subset_graph(self, G, communities=None):
        """
        If communities is not None, only return graph of nodes in communities subset.

        param G: input graph
        param communities: list of int
        """

        import networkx as nx
        # filter graph to desired community subset
        comm_list = nx.get_node_attributes(G, 'Community')
        nodes = list(G.nodes)
        G2 = G.copy()
        if communities is not None:
            for node in nodes:
                if comm_list[node] not in communities:
                    G2.remove_node(node)

        #nx.write_gexf(G2, outpath)

        return G2

    def set_node_attributes(self, G, perc_nodes_to_use, numTopics, perc_bots, impactednesses, sentiments):

        import numpy as np
        import networkx as nx
        import random
        '''

        Will create a network for simulation using input graph and provided community level attitudes towards topics

        :param G: input digraph
        :param perc_nodes_to_use: percentage of nodes from G you want to keep for simulation
        :param numTopics: number of topics in use this simulation round
        :perc_bots: what percentage of nodes will be bots
        :param impactednesses:  len(numTopics) list of dictionaries. impactednesses[i] contains dictionary where keys are communities and values are the community's
                                impactedness value towards topic i.
                                (this value will be used as mean for drawing distribution)
        :param sentiments: len(numTopics) list of dictionaries. sentiments[i] contains dictionary where keys are communities and values are the community's
                                sentiment value towards topic i.
                                (this value will be used as mean for drawing distribution)
        :return: returns new network where nodes have impactedness, sentiments, and are bots


        '''

        to_remove = random.sample(list(G.nodes), int(len(G.nodes)*(1-perc_nodes_to_use)))
        to_keep = list(set(list(G.nodes)) - set(to_remove))
        G.remove_nodes_from(to_remove)

        # Randomly select who will be a bot
        num_bots = int(np.round(len(to_keep)*perc_bots))
        bot_names = random.sample(to_keep, num_bots) #might want to make it so we sample a certain number from each community instead

        for node, data in G.nodes(data=True):

            #set things that matter if you are bot or not
            if node in bot_names:
                data['lambda'] = np.random.uniform(0.5,0.75)
                data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
                data['inbox'] = []
                data['kind'] = 'bot'
                data['mentioned_by'] = []
            else:
                data['lambda'] = np.random.uniform(0.001,0.75)
                data['wake'] = 0 + np.round(np.random.exponential(scale = 1 / data['lambda']))
                data['inbox'] = []
                data['mentioned_by'] = []
                data['kind'] = 'normal'

            #set everything else
            data['impactedness'] = {}
            data['sentiment'] = {}
            data['num_read'] = {}

            for topic in range(numTopics):
                data['impactedness'][topic] = np.max([0, np.random.normal(loc=impactednesses[topic][data['Community']], scale=0.1)]) #making it a gaussian for now
                data['sentiment'][topic] = np.max([0, np.random.normal(loc=sentiments[topic][data['Community']], scale=0.1)]) #making it a gaussian for now
                data['num_read'][topic] = np.max([0, np.random.normal(impactednesses[topic][data['Community']]*100, scale = 10)])

            data['belief'] = np.array(list(data['sentiment'].values())).mean() #make belief an average of sentiments? then what we are interested in are changes in belief due to misinfo?



            if data['kind'] != 'bot':
                if data['belief'] < 0.2: #this might need to be adjusted depending on how belief figures look
                    data['kind'] = 'beacon'



        ## Remove self_loops and isololates
        G.remove_edges_from(list(nx.selfloop_edges(G, data=True)))
        G.remove_nodes_from(list(nx.isolates(G)))

        bb = nx.betweenness_centrality(G, k=int(0.1*len(list(G.nodes()))))
        degrees = dict(G.in_degree())
        nx.set_node_attributes(G, bb, "centrality")
        nx.set_node_attributes(G, degrees, "degree")

        return G


    def retweet_behavior(self, topic, value, topic_sentiment, creator_prestige, claim_virality):
        if value == -1:
            retweet_perc = np.min([1, (1 - topic_sentiment)*creator_prestige*claim_virality])
        elif value == 0:
            retweet_perc = np.min([1, (0.5)*creator_prestige*claim_virality])
        elif value == 1: # misinfo is retweeted 70% more than real news
            retweet_perc = np.min([1, topic_sentiment*creator_prestige*claim_virality])
        return retweet_perc


    def update_topic_sentiment(self, current_sentiment, tweet_value, tweet_impactedness, num_read, learning_rate):
        # iterative mean updating of beliefs
        new_sentiment = ((num_read - 1)/num_read)*current_sentiment + (1/num_read)*tweet_value*(1+tweet_impactedness)
        difference = learning_rate*(new_sentiment - current_sentiment)
        return current_sentiment + difference

    def update_read_counts(self, community_read_tweets_by_type, topic, info_type, com, step):
        if info_type == 1:
            community_read_tweets_by_type[com][step][topic]['misinfo'] += 1
        elif info_type == 0:
            community_read_tweets_by_type[com][step][topic]['noise'] += 1
        else:
            community_read_tweets_by_type[com][step][topic]['anti-misinfo'] += 1
        return community_read_tweets_by_type



    def percentile(self, x):
        import numpy as np
        from scipy.stats import rankdata

        x = np.array(x)
        ranks = rankdata(x)
        return(ranks/len(x))

def random_community_sample(community_graph, min_network_size=100000, max_network_size=200000):
        """
        Select a random subset of communities such that total number of nodes is between min and max network size.
        """
        import networkx as nx
        from random import sample

        communities = []
        community_sizes = pd.Series(nx.get_node_attributes(community_graph, 'SIZE'))
        community_sizes.index = community_sizes.index.astype(int)
        community_sizes = community_sizes[community_sizes >= 10000]
        num_network_nodes = 0

        while num_network_nodes < min_network_size or len(communities) < 2:
            community = int(sample(list(community_sizes.index), 1)[0])
        
            if community_sizes.loc[communities + [community]].sum() < max_network_size and (community not in communities):
                communities.append(community)
                num_network_nodes += community_sizes[community]
        

        return communities

def make_comm_string(communities):
        comm_string = ''
        for comm in communities:
            comm_string += '_' + str(comm)

        return comm_string
