from checkworthy import Checkworthy
from TopicSim import TopicSim, random_community_sample, make_comm_string
import yaml
import pickle
import gc
import os
import sys, getopt
import networkx as nx

#making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main(argv):
    """
    specify run parameters. p is period (pre, post), m is mitigation method, l is label method, s is sample method
    """
    opts, args = getopt.getopt(argv, "p:m:l:s:g:k")

    for opt, arg in opts:
        if opt == "-m":
            mitigation_method = arg
        elif opt == "-l":
            label_method = arg
        elif opt == "-s":
            sample_method = arg
        elif opt == "-p":
            period = arg
        elif opt == '-k':
            moderator = arg





    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)




    reps = config['reps']


    '''
    First topic (row) impacts every group the same, the other topics each impact
    one group significantly more than the others
    '''

    if config['random_communities'] and period=='pre':


        communities = random_community_sample(nx.read_gexf(config['community_graph_path']))


        config['communities'] = communities
        with open('config.yaml', 'w') as file:
            file.write(yaml.dump(config, default_flow_style=False))


    else:
        communities = config['communities']

    
    if config['state_of_world'] == 1:
        
        num_topics = len(communities) + 1
    
        impactednesses = [{comm: 0.5 for comm in communities} for i in range(num_topics)]
    
        for i in range(1, len(impactednesses)):
            keys = list(impactednesses[i].keys())
            for key in impactednesses[i].keys():
                if key == keys[i-1]:
                    impactednesses[i][key] = config['high_impactedness'][config['state_of_world']-1]
                else:
                    impactednesses[i][key] = config['low_impactedness'][config['state_of_world']-1]
    
    
        '''
        For the first topic (which everyone cares about equally), belief is roughly average (50%).
        For the other topics, if a community is more impacted by a topic, we assume that their
        average belief is lower, indicating that they have more knowledge of the truth than the
        other communities that are not impacted  by the topic.
        '''
    
        beliefs = [{comm: 0.5 for comm in communities} for i in range(num_topics)]
    
        for i in range(1, len(beliefs)):
            keys = list(beliefs[i].keys())
            for key in beliefs[i].keys():
                if key == keys[i-1]:
                    beliefs[i][key] = config['low_belief'][config['state_of_world']-1]
                else:
                    beliefs[i][key] = config['high_belief'][config['state_of_world']-1]
                    
    elif config['state_of_world'] == 2:
        
        '''
        This is a world where there exist N groups, and N - 1 topics. A majority and minority community are impacted differently
        by each topic. A minority knowledgeable community is impacted equally by both topics.
        '''

        num_topics = len(communities) - 1
        impactednesses = [{comm: 0.5 for comm in communities} for i in range(num_topics)]
        for com in range(len(communities)):
            for i in range(num_topics):
                if com == i:
                    impactednesses[i][communities[com]] = config['high_impactedness'][config['state_of_world']-1]
                elif com == (len(communities)-1):
                    impactednesses[i][communities[com]] = config['middle_impactedness']
                else:
                    impactednesses[i][communities[com]] = config['low_impactedness'][config['state_of_world']-1]

    
    
        '''
        In this construct, both the minority and majority communities are equally unknowledgable, and there exists
        one knowledgeable minority community about all topics (a little unrealistic, but demonstrates the concept).
        '''
    
        beliefs = [{comm: 0.5 for comm in communities} for i in range(num_topics)]
    
        for com in range(len(communities)):
            for i in range(num_topics):
                if com == (len(communities)-1):
                    beliefs[i][communities[com]] = config['low_belief'][config['state_of_world']-1]
                else:
                    beliefs[i][communities[com]] = config['high_belief'][config['state_of_world']-1]

        
   
    
    gc.enable()



    for rep in range(reps):

        if period == 'pre':
            print('\n\n ---------- REPETITION #' + str(rep) + 'pre-period -------------- \n\n')

            check = Checkworthy(
                agg_interval = config['agg_interval'],
                agg_steps = config['agg_steps'],
                depths = config['depths'],
                impactednesses = impactednesses,
                beliefs = beliefs
            )


            sim = TopicSim(
                impactedness = impactednesses,
                beliefs = beliefs,
                num_topics = num_topics,
                runtime = config['runtime'],
                communities = communities,
                num_claims = config['num_claims'],
                rep=rep
            )

            '''
            Import or create networks.
            If create, nodes in network must already have 'Community' attribute
            '''

            if config['load_data'] and not(config['random_communities']):
                sim.load_simulation_network(ready_network_path = config['ready_network_path'])
            else: #no need to recreate network on subsequent reps
                sim.create_simulation_network(
                    raw_network_path = config['raw_network_path'],
                    perc_nodes_to_subset = config['perc_nodes_to_subset'],
                    perc_bots = config['perc_bots'],
                    moderator = moderator
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


            with open(config['output_sim_midpoint'] + '_run' + str(rep) + '_communities' + sim.comm_string + '.pickle', 'wb') as file:
                pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)


            del sim
            del check
            gc.collect()

        elif period == 'post':

            if mitigation_method=='None':

                with open(config['output_sim_midpoint'] + '_run' + str(rep) + '_communities' + make_comm_string(communities) + '.pickle', 'rb') as file:
                    sim = pickle.load(file)

                sim.set_post_duration(config['post_duration'])

                sim.run(
                    period = period,
                    learning_rate = config['learning_rate'],
                    min_degree=config['min_degree'],
                    fact_checks_per_step = config['fact_checks_per_step'],
                    mitigation_type = 'None'
                )

                with open(config['output_sim_final_no_intervention'] + '_run' + str(rep) + '_communities' + sim.comm_string + '.pickle', 'wb') as file:
                    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)

                del sim
                gc.collect()

            else:

                with open(config['output_sim_midpoint'] + '_run' + str(rep) + '_communities' + make_comm_string(communities) + '.pickle', 'rb') as file:
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

                with open('../output/simulation_{}_{}_{}'.format(mitigation_method, label_method, sample_method) + '_run' + \
                    str(rep) +  '_communities' + sim.comm_string + '.pickle', 'wb') as file:
                    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)

                del sim
                gc.collect()



        else:
            print('invalid period')


if __name__ == "__main__":
   main(sys.argv[1:])
