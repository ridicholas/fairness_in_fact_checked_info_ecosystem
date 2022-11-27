import pandas as pd
import networkx as nx
import random
import numpy as np
from numpy.random import choice
from scipy.stats import beta, rankdata
from checkworthy import *

def scale(x):
    '''
    Normalizes a vector by dividing each element by the vector max.
    '''
    x = np.array(x)
    return(x/np.max(x))


def percentile(x):
    x = np.array(x)
    ranks = rankdata(x)
    return(ranks/len(x))


def create_simulation_network(G: nx.digraph, perc_nodes_to_use: float, numTopics: int, perc_bots: float, impactednesses: list, sentiments: list):
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

    bb = nx.betweenness_centrality(G)
    degrees = dict(G.in_degree())
    nx.set_node_attributes(G, bb, "centrality")
    nx.set_node_attributes(G, degrees, "degree")

    return(G)



def calculate_sentiment_rankings(G: nx.DiGraph, topics: list):

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
    rankings = pd.DataFrame(index = all_node_sentiments.keys())

    for topic in topics:
        node_sentiments = [all_node_sentiments[key][topic] for key in all_node_sentiments.keys()]
        median = np.median(node_sentiments)
        deviations = [np.absolute(i - median) for i in node_sentiments]
        rankings['sentiment' + str(topic)] = node_sentiments
        rankings['deviation' + str(topic)] = deviations
        rankings['rank' + str(topic)] = np.where(rankings['sentiment' + str(topic)] < median,
                                                 -1*rankings['deviation' + str(topic)].rank(method='max')/len(rankings),
                                                 rankings['deviation' + str(topic)].rank(method='max')/len(rankings))

    return rankings



def choose_topic(data: dict):
    topic_probs = [i / sum(data['impactedness'].values()) for i in data['impactedness'].values()]
    topic = choice(np.arange(0, len(topic_probs)), p=topic_probs)
    return topic

def choose_info_quality(node: str, rankings: pd.DataFrame, topic: int, agent_type: str):

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
        deviation_rank = rankings.loc[node].loc['rank' + str(topic)]
        raw = beta.rvs(5 + deviation_rank, 5 - deviation_rank, size=1)
        value = np.where(raw < 0.3333, -1, np.where(raw >= 0.3333 and raw < 0.666, 0, 1))[0]
    else:
        value = np.where(np.random.uniform(size=1) > 0.2, 1, 0)[0]
    return value

def create_claims(num_claims):

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




def choose_claim(value: int, num_claims: int):
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

def subset_graph(G, communities=None):
    """
    If communities is not None, only return graph of nodes in communities subset.

    param G: input graph
    param communities: list of int
    """

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

def retweet_behavior(topic, value, topic_sentiment, creator_prestige, claim_virality):
    if value == -1:
        retweet_perc = np.min([1, (1 - topic_sentiment)*creator_prestige*claim_virality])
    elif value == 0:
        retweet_perc = np.min([1, (0.5)*creator_prestige*claim_virality])
    elif value == 1: # misinfo is retweeted 70% more than real news
        retweet_perc = np.min([1, topic_sentiment*creator_prestige*claim_virality])
    return retweet_perc


def update_topic_sentiment(current_sentiment, tweet_value, tweet_impactedness, num_read, learning_rate):
    # iterative mean updating of beliefs
    new_sentiment = ((num_read - 1)/num_read)*current_sentiment + (1/num_read)*tweet_value*(1+tweet_impactedness)
    difference = learning_rate*(new_sentiment - current_sentiment)
    return current_sentiment + difference





