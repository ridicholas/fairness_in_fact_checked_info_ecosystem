# fairness_in_fact_checked_info_ecosystem

create_network.py - contains create_network(path) function. 

- Loads edgelist from path

- loads communities from pickle if 'communities.pkl' exists or runs lovain alg to identify communities and generate pickle

- Assigns communities to node as 'community' node attribute
    
calc_community_stats.py - contains functions for calculating overall network and community sub-graph statistics. 

