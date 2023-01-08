# fairness_in_fact_checked_info_ecosystem

This is the code for our Fairness in a Fact Checked Information Ecosystem course project. 

1_detect_communities contains the create_network script which loads in the initial network, identifies communities, and appends the community as a node attribute. 

2_calc_community_stats contains a script to analyze the network overall, producing total edge counts, clustering coefficient, avg. shortest path length, etc. 

3_subsetting contains a script to subset our network to only include the communities we are interested in. 

4_simulation is the most important portion of our code-base. Topic_sim.py contains the class for simulation repetitions. This TopicSim class stores information about the network, information trajectories over time, and contains a run function for running the simulation. The checkworthy.py file contains a class which stores data used to build our check-worthiness models and all functions which update this dataset to be called from within the our simulation run. Eval_model_predictions.py is used to assess the check-worthiness model quality, and sim_util.py contains helper functions for the simulation. 

5_analysis contains the process_data.py and make_plots.py scripts which in conjuction are used to generate the plots seen in the paper. 




