import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import progressbar
import fastparquet

os.chdir(os.path.dirname(os.path.abspath(__file__)))

my_dpi = 96
communities_to_subset = [3, 56, 43]
runtime = 500
def plot_total_reads_over_time(by_community=False, specific_topic=None):

    if specific_topic != None:
        misinfo = pd.read_pickle('../output/topic{}_node_by_time_misinfo.pickle'.format(specific_topic))
        anti = pd.read_pickle('../output/topic{}_node_by_time_anti.pickle'.format(specific_topic))
        noise = pd.read_pickle('../output/topic{}_node_by_time_noise.pickle'.format(specific_topic))
    else:
        misinfo = pd.read_pickle('../output/node_by_time_misinfo.pickle')
        anti = pd.read_pickle('../output/node_by_time_anti.pickle')
        noise = pd.read_pickle('../output/node_by_time_noise.pickle')

    plt.figure(figsize=(1000/my_dpi, 800/my_dpi), dpi=my_dpi)
    plt.plot(list(range(runtime)), misinfo.iloc[:, 1:].sum(axis=0), label = 'misinfo', color='red')
    plt.plot(list(range(runtime)), noise.iloc[:, 1:].sum(axis=0), label = 'noise', color='gray')
    plt.plot(list(range(runtime)), anti.iloc[:, 1:].sum(axis=0), label = 'anti', color='blue')
    plt.xlabel('Time')
    plt.ylabel('# of Reads')
    if specific_topic:
        plt.title('Total Topic {} Information Read over Time'.format(specific_topic))
    else:
        plt.title('Total Information Read over Time')
    plt.legend()
    #plt.show()
    plt.savefig('../output/total_topic_{}.png'.format(specific_topic))

    if by_community:
        plt.clf()
        plt.figure(figsize=(8000/my_dpi, 800/my_dpi), dpi=my_dpi)
        fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        i=0
        for community in communities_to_subset:
            axs[i].plot(list(range(runtime)), misinfo[misinfo['Community']==community].iloc[:, 1:].sum(axis=0), label = 'misinfo', color='red')
            axs[i].plot(list(range(runtime)), noise[misinfo['Community']==community].iloc[:, 1:].sum(axis=0), label = 'noise', color='gray')
            axs[i].plot(list(range(runtime)), anti[misinfo['Community']==community].iloc[:, 1:].sum(axis=0), label = 'anti', color='blue')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('# of Reads')
            if specific_topic:
                axs[i].set_title('Topic {} Information Read over Time by Community {}'.format(specific_topic, community))
            else:
                axs[i].set_title('Information Read over Time by Community {}'.format(community))
            axs[i].legend()
            i+=1
            output = '../community_' + str(community) + '_topic_' + str(specific_topic) + '.png'
            plt.savefig(output, dpi = my_dpi)
    else:
        plt.clf()
        plt.figure(figsize=(1000/my_dpi, 800/my_dpi), dpi=my_dpi)
        plt.plot(list(range(runtime)), misinfo.iloc[:, 1:].sum(axis=0), label = 'misinfo', color='red')
        plt.plot(list(range(runtime)), noise.iloc[:, 1:].sum(axis=0), label = 'noise', color='gray')
        plt.plot(list(range(runtime)), anti.iloc[:, 1:].sum(axis=0), label = 'anti', color='blue')
        plt.xlabel('Time')
        plt.ylabel('# of Reads')
        plt.title('Total Information Read over Time')
        plt.legend()
    plt.show()

    del misinfo
    del anti
    del noise



plot_total_reads_over_time(by_community=True, specific_topic='0')
plot_total_reads_over_time(by_community=True, specific_topic='1')
plot_total_reads_over_time(by_community=True, specific_topic='2')
plot_total_reads_over_time(by_community=True, specific_topic='3')
