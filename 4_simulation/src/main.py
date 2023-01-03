from checkworthy import Checkworthy
from TopicSim import TopicSim
import yaml
import pickle
import gc
import os
import sys, getopt

#making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__))) 

def main(argv):
    """
    specify run parameters. p is period (pre, post), m is mitigation method, l is label method, s is sample method
    """
    opts, args = getopt.getopt(argv, "p:m:l:s:g")

    for opt, arg in opts:
        if opt == "-m": 
            mitigation_method = arg
        elif opt == "-l":
            label_method = arg
        elif opt == "-s":
            sample_method = arg
        elif opt == "-p":
            period = arg
    

    reps = 10


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
            {3: 0.5, 56: 0.5, 43: 0.5},
            {3: 0.5, 56: 0.5, 43: 0.5},
            {3: 0.5, 56: 0.5, 43: 0.5}]

    gc.enable()

    for rep in range(reps):

        if period == 'pre':
            print('\n\n ---------- REPETITION #' + str(rep) + 'pre-period -------------- \n\n')
            
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
            If create, nodes in network must already have 'Community' attribute
            '''

            if config['load_data']:
                sim.load_simulation_network(ready_network_path = config['ready_network_path'])
            else:
                sim.create_simulation_network(
                    raw_network_path = config['raw_network_path'],
                    perc_nodes_to_subset = config['perc_nodes_to_subset'],
                    perc_bots = config['perc_bots']
                )

            # Pass network to Checkworthy object
            check.set_network(G = sim.return_network(), communities=sim.return_communities())
            # reset checkworthy object object
            sim.set_check(check = check)


            sim.run(
                period = period,
                learning_rate = config['learning_rate'],
                min_degree=config['min_degree'],
                fact_checks_per_step = config['fact_checks_per_step'],
                mitigation_type = 'None'
            )


            with open(config['output_sim_midpoint'] + str(rep) + '.pickle', 'wb') as file:
                pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)


            del sim
            del check
            gc.collect()

        elif period == 'post':

            if mitigation_method=='None':

                with open(config['output_sim_midpoint'] + str(rep) + '.pickle', 'rb') as file:
                    sim = pickle.load(file)

                sim.set_post_duration(config['post_duration'])

                sim.run(
                    period = period,
                    learning_rate = config['learning_rate'],
                    min_degree=config['min_degree'],
                    fact_checks_per_step = config['fact_checks_per_step'],
                    mitigation_type = 'None'
                )

                with open(config['output_sim_final_no_intervention'] + str(rep) + '.pickle', 'wb') as file:
                    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)

                del sim
                gc.collect()

            else:

                with open(config['output_sim_midpoint'] + str(rep) + '.pickle', 'rb') as file:
                    sim = pickle.load(file)

                sim.set_post_duration(config['post_duration'])

                check_pre = sim.return_check()

                
                print('\n\n\n ----------- Sampling Claims for Checkworthy Dataset --------- \n\n')
                check_pre.sample_claims(num_to_sample=config['claims_to_sample'], sample_method=sample_method)
                
                print('\n\n\n ----------- Random Sampling of Labels for Checkworthy Dataset --------- \n\n')
                check_pre.sample_labels_for_claims(labels_per_claim = config['nodes_to_sample'], sample_method = label_method)
                
                print('\n\n\n ----------- Training Model with Label = average_truth_perception_random --------- \n\n')

                check_pre.train_model(label_to_use='average_truth_perception_{}'.format(label_method))

                sim.set_check(check=check_pre)

                sim.run(
                    period = period,
                    learning_rate = config['learning_rate'],
                    min_degree=config['min_degree'],
                    fact_checks_per_step = config['fact_checks_per_step'],
                    mitigation_type = mitigation_method
                )

                with open(config['output_sim_final_intervention_random'] + str(rep) + '.pickle', 'wb') as file:
                    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)

                del sim
                gc.collect()


        
        else:
            print('invalid period')


if __name__ == "__main__":
   main(sys.argv[1:])
        

    



    





