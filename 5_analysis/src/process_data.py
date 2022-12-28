import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import progressbar

# making sure wd is file directory so hardcoded paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#read in files from 4_simulation/output


def make_reads_by_time_frame(infile, reps, modules):


    import TopicSim
    import checkworthy
    import gc

    result = pd.DataFrame()
    gc.enable()
    for rep in range(reps):

        print('\n\n\n\n\n Processing Reads Data - Repetition #' + str(rep) + '------ \n\n\n\n')

        for mod in modules:

            with open(infile + mod + str(rep) + '.pickle', 'rb') as file:
                sim = pickle.load(file)

            community_read_over_time = sim.community_read_tweets_by_type

            del sim
            gc.collect()

            cols = ['Community', 'Step', 'Topic', 'Type', 'Rep','Intervention','Reads']
            anti_frame = pd.DataFrame(columns = cols)
            noise_frame = pd.DataFrame(columns = cols)
            misinfo_frame = pd.DataFrame(columns = cols)

            bar = progressbar.ProgressBar()
            for com in bar(list(community_read_over_time.keys())):
                for step in list(community_read_over_time[com].keys()):
                    for topic in list(community_read_over_time[com][step].keys()):
                        count_anti = community_read_over_time[com][step][topic]['anti-misinfo']
                        count_noise = community_read_over_time[com][step][topic]['noise']
                        count_misinfo = community_read_over_time[com][step][topic]['misinfo']
                        # append rows
                        anti_frame.loc[len(anti_frame)] = [com, step, topic, 'anti-misinfo', rep, mod, count_anti]
                        noise_frame.loc[len(noise_frame)] = [com, step, topic, 'noise', rep, mod, count_noise]
                        misinfo_frame.loc[len(misinfo_frame)] = [com, step, topic, 'misinfo', rep, mod, count_misinfo]

            result = pd.concat([result, anti_frame, noise_frame, misinfo_frame])

    return result





def process_community_belief(infile, reps, modules):

    import TopicSim
    import checkworthy
    import gc


    result = pd.DataFrame(columns = ['Community','Topic','Time', 'Rep', 'Intervention', 'Mean Belief'])
    gc.enable()

    for rep in range(reps):

        print('\n\n\n\n\n Processing Belief Data - Repetition #' + str(rep) + '------ \n\n\n\n')

        for mod in modules:

            with open(infile + mod + str(rep) + '.pickle', 'rb') as file:
                sim = pickle.load(file)

            community_sentiment_through_time = sim.community_sentiment_through_time

            del sim
            gc.collect()

            for comm in list(community_sentiment_through_time.keys()):
                for t in list(community_sentiment_through_time[comm].keys()):
                    for topic in list(community_sentiment_through_time[comm][t].keys()):
                        mean_sentiment = np.mean(community_sentiment_through_time[comm][t][topic])
                        result.loc[len(result)] = [comm,topic,t,rep,mod,mean_sentiment]

    result = result.sort_values(by=['Rep', 'Intervention', 'Community','Topic','Time'])

    return result



print('\n\n\n ------- Processing Community Sentiment over Time ------- \n\n\n')

clean_community_belief = process_community_belief(infile = '../../4_simulation/output/simulation_final_',
                                                  reps = 8,
                                                  modules=['no_intervention','intervention_random_label', 'intervention_stratified_label', 'intervention_kc_label'])

clean_community_belief.to_csv('../output/exp_results_community_belief.csv', index = False)


print('\n\n\n ------- Processing Information Read Over Time ------- \n\n\n')

result = make_reads_by_time_frame(infile = '../../4_simulation/output/simulation_final_',
                                  reps = 8,
                                  modules = ['no_intervention','intervention_random_label', 'intervention_stratified_label', 'intervention_kc_label'])

result.to_pickle('../output/exp_results_information_read_aggregated.pickle')
