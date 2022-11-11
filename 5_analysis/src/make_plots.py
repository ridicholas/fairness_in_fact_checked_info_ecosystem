import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import progressbar

os.chdir(os.path.dirname(os.path.abspath(__file__)))





def plot_total_reads_over_time(by_community=False):

    misinfo = pd.read_pickle('../output/node_by_time_misinfo.pickle')
    anti = pd.read_pickle('../output/node_by_time_anti.pickle')
    noise = pd.read_pickle('../output/node_by_time_noise.pickle')
    
    plt.plot(list(range(1000)), misinfo.iloc[:, 1:].sum(axis=0), label = 'misinfo', color='red')
    plt.plot(list(range(1000)), noise.iloc[:, 1:].sum(axis=0), label = 'noise', color='gray')
    plt.plot(list(range(1000)), anti.iloc[:, 1:].sum(axis=0), label = 'anti', color='blue')
    plt.xlabel('Time')
    plt.ylabel('# of Reads')
    plt.title('Total Information Read over Time')
    plt.legend()
    plt.show()

    if by_community():
        communities = [3, 56, 43]
        plt.clf()
        fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        i=0
        for community in communities:

            axs[i].plot(list(range(1000)), misinfo[misinfo['Community']==community].iloc[:, 1:].sum(axis=0), label = 'misinfo', color='red')
            axs[i].plot(list(range(1000)), noise[misinfo['Community']==community].iloc[:, 1:].sum(axis=0), label = 'noise', color='gray')
            axs[i].plot(list(range(1000)), anti[misinfo['Community']==community].iloc[:, 1:].sum(axis=0), label = 'anti', color='blue')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('# of Reads')
            axs[i].set_title('Information Read over Time by Community {}'.format(community))
            axs[i].legend()
            i+=1
    else:
        plt.clf()
        plt.plot(list(range(1000)), misinfo.iloc[:, 1:].sum(axis=0), label = 'misinfo', color='red')
        plt.plot(list(range(1000)), noise.iloc[:, 1:].sum(axis=0), label = 'noise', color='gray')
        plt.plot(list(range(1000)), anti.iloc[:, 1:].sum(axis=0), label = 'anti', color='blue')
        plt.xlabel('Time')
        plt.ylabel('# of Reads')
        plt.title('Total Information Read over Time')
        plt.legend()
    plt.show()

    del misinfo
    del anti
    del noise

def plot_claims_over_time(claims=None, topics=None, by_community=False):
    '''
    Have to provide claims or topics to plot
    '''

    if claims!=None and topics==None:
        comm3 = pd.read_pickle('../output/claim_by_time_community3.pickle')
        community3spread = comm3.loc[claims, :]
        del comm3
        comm56 = pd.read_pickle('../output/claim_by_time_community56.pickle')
        community56spread = comm56.loc[claims, :]
        del comm56
        comm43 = pd.read_pickle('../output/claim_by_time_community56.pickle')
        community43spread = comm43.loc[claims, :]
        del comm43

        if not(by_community):
            plt.clf()
            spread = community3spread + community43spread + community56spread
            for claim in claims:
                plt.plot(list(range(1000)), spread.loc[claim, :], label=claim)
            
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('# of Reads')
            plt.title('Claims Read Over Time')
            plt.show()
        else: 
            spread = [community3spread , community56spread , community43spread]
            plt.clf()
            fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
            i=0
            comNames = [3, 56, 43]
            for community in spread:
                for claim in claims:
                    axs[i].plot(list(range(1000)), community.loc[claim, :], label=claim)
                    axs[i].set_xlabel('Time')
                    axs[i].set_ylabel('# of Reads')
                    axs[i].set_title('Claims Read Over Time by Community {}'.format(comNames[i]))
                    axs[i].legend()
                i+=1
            plt.show()
    
    elif topics!=None and claims==None:
        pass
    else:
        print('Gotta provide either topics or claims try again friend')



#Some example things we can do, should come up with more interesting stuff to plot
#Should also consider a better way to work with the claim by time data which are 3 yuge files

plot_total_reads_over_time(by_community=False)
plot_total_reads_over_time(by_community=True)
plot_claims_over_time(claims = ['1-53-93-0', '1-38-428-0'], by_community=False)
plot_claims_over_time(claims = ['1-53-93-0', '1-38-428-0'], by_community=True)