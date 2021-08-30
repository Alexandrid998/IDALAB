
import os
import sys
import traceback
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels import robust
from statsmodels.distributions.empirical_distribution import ECDF
from scapy.all import *
import pickle
import beepy

def Preprocessing(string):
    '''
    Parameters
    ----------
    string : Stringa
        Nome del dataset da cui estrarre informazioni

    Returns
    -------
    a : Pandas Dataframe
        Dataframe contenente le statistiche dei pacchetti relative al pcap in input
        a cui vengono associati anche label, protocollo e classe (attacco)
    b : Array
        Output identico ad a, soltanto in formato numpy
    c : Array
        Output contenente solo pacchetti con relative features con label 0 (benigni)
    d : Array
         Output contenente solo pacchetti con relative features con label 1 (maligni)

    Estrae tutti i pacchetti dal file pcap, crea un dataframe che aggiunge alle features 
    dei pacchetti la label, il protocollo del pacchetto e la classe di traffico (tipo di attacco)
    '''
    packet= rdpcap("Dataset/"+str(string)+"/"+str(string)+"_pcap.pcapng")
    Protocollo=np.empty(len(packet),dtype='object')
    Attacco=np.empty(len(packet),dtype='object')
    for i in range(0,len(packet)):
        try:
            if packet[i].proto == 1:
                Protocollo[i]= "ICMP"
            elif packet[i].proto == 2:
                Protocollo[i]= "IGMP"
            elif packet[i].proto == 6:
                Protocollo[i]= "TCP"
            elif packet[i].proto == 17:
                Protocollo[i]= "UDP"
            else:
                Protocollo[i]= "OTHERS"
        except AttributeError:
            Protocollo[i]= "ARP"
        Attacco[i]= str(string)
    proto=pd.DataFrame(Protocollo)
    attack=pd.DataFrame(Attacco)
    dataset=pd.read_csv('Dataset/'+str(string)+'/'+str(string)+'_dataset.csv', delimiter=',',header=None,index_col=False)
    labels=pd.read_csv('Dataset/'+str(string)+'/'+str(string)+'_labels.csv', delimiter=',')
    labels=labels.drop(columns= 'Unnamed: 0')
    a=pd.concat([dataset,labels,proto,attack],ignore_index=True,axis=1)
    b=a.to_numpy()
    c=b[(b[:,115]==0)]
    d=b[(b[:,115]!=0)]
    return a,b,c,d


    
    