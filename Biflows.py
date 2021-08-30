import pandas as pd
import numpy as np
import scipy
import sys
import os
import matplotlib.pyplot as plt
from statsmodels import robust
from statsmodels.distributions.empirical_distribution import ECDF
from scapy.all import *
import pickle
import pyarrow
import beepy

#Biflows_Check legge il pcap relativo ad ogni singolo attacco del dataset e raggruppa i singoli pacchetti in biflussi
def Biflows_Check(string):
    all_biflows = dict()
    packets_pcap = rdpcap('Dataset/'+str(string)+'/'+str(string)+'_pcap.pcapng')
    packets_labels = pd.read_csv('Dataset/'+str(string)+'/'+str(string)+'_labels.csv',delimiter =',')
    packets_labels = packets_labels.drop(columns='Unnamed: 0')
    packets_labels = packets_labels.to_numpy()
    for i,packet in enumerate(packets_pcap):
        if packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP)):
            if not(packet.haslayer(NTP) or packet.haslayer(DNS) or packet.sport == 5353 or packet.dport == 5353 or packet[IP].src == '0.0.0.0'):
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                src_port = packet.sport
                dst_port = packet.dport

                src = (src_ip,src_port)
                dst = (dst_ip,dst_port)
                proto = packet[IP].proto

              
                quintupla = (src,dst,proto)
                inverse_quintupla = (dst,src,proto)
                
                if quintupla not in all_biflows and inverse_quintupla not in all_biflows:
                    all_biflows[quintupla] = []
                    all_biflows[quintupla].append([packet,packets_labels[i][0]])
            
                elif quintupla in all_biflows:
                    all_biflows[quintupla].append([packet,packets_labels[i][0]])
           
                elif inverse_quintupla in all_biflows:
                    all_biflows[inverse_quintupla].append([packet,packets_labels[i][0]])
    
    pickle.dump(all_biflows,open('Biflussi/'+str(string)+'/all_biflows_'+str(string)+'.p','wb')) 
    
    
#Estrai_statistiche_biflows estrae le statistiche di ogni biflusso. Successivamente distingue i biflussi malevoli da quelli benigni e infine salva anche i biflussi
#che hanno sia pacchetti benigni che malevoli
def Estrai_statistiche_biflows(string):
    
    all_biflows=pickle.load(open('Biflussi/'+str(string)+'/all_biflows_'+str(string)+'.p','rb'))
    
    traffic_biflows = dict()

    for quintupla in all_biflows:    
        traffic_biflows[quintupla] = {
            'ttl': [],
            'TCP_Window':[],
            'timestamp': [],
            'pkt_length': [],
            'iat': [],
            'l4_payload': [],
            'pay_length': [],
            'malign': []
        }
        cons_pkt = []
        for pkt in all_biflows[quintupla]:
            timestamp = pkt[0].time
            packet_length = pkt[0][IP].len
            time_tl=pkt[0].ttl
            if pkt[0].haslayer(TCP):
                window=pkt[0].window
            else:
                window=0

            if pkt[0].haslayer(Raw):
                payload_data = pkt[0].getlayer(Raw).load
                payload_length = len(pkt[0].getlayer(Raw).load)
            else:
                payload_data = ''
                payload_length = 0

            cons_pkt.append(pkt[0].time)
            if len(cons_pkt) < 2:
                interarrival_time = 0
            else:

                interarrival_time = np.diff(cons_pkt)[0]
                cons_pkt = cons_pkt[1:]

            malevolo=pkt[1]

            traffic_biflows[quintupla]['ttl'].append(time_tl)
            traffic_biflows[quintupla]['TCP_Window'].append(window)
            traffic_biflows[quintupla]['timestamp'].append(timestamp)
            traffic_biflows[quintupla]['pkt_length'].append(packet_length)
            traffic_biflows[quintupla]['iat'].append(interarrival_time)
            traffic_biflows[quintupla]['l4_payload'].append(payload_data)
            traffic_biflows[quintupla]['pay_length'].append(payload_length)
            traffic_biflows[quintupla]['malign'].append(malevolo)

    pkt_length_statistics_biflows_benign = dict()
    pkt_length_statistics_biflows_malign = dict()
    benign_biflows= dict()
    malign_biflows=dict()
    mix_biflows=dict()
    for quintupla in traffic_biflows:
        misti=0
        benign_biflows[quintupla] = {
            'ttl': [],
            'TCP_Window': [],
            'timestamp' : [],
            'pkt_length' : [],
            'iat' : [],
            'l4_payload' : [],
            'pay_length' : [],
        }
        malign_biflows[quintupla] = {
            'ttl': [],
            'TCP_Window':[],
            'timestamp' : [],
            'pkt_length' : [],
            'iat' : [],
            'l4_payload' : [],
            'pay_length' : [],   
        } 
        for i in range(np.size(traffic_biflows[quintupla]['malign'])):
            if traffic_biflows[quintupla]['malign'][i] == 0:
                benign_biflows[quintupla]['ttl'].append(traffic_biflows[quintupla]['ttl'][i])
                benign_biflows[quintupla]['TCP_Window'].append(traffic_biflows[quintupla]['TCP_Window'][i])
                benign_biflows[quintupla]['timestamp'].append(traffic_biflows[quintupla]['timestamp'][i])
                benign_biflows[quintupla]['pkt_length'].append(traffic_biflows[quintupla]['pkt_length'][i])
                benign_biflows[quintupla]['iat'].append(traffic_biflows[quintupla]['iat'][i])
                benign_biflows[quintupla]['l4_payload'].append(traffic_biflows[quintupla]['l4_payload'][i])
                benign_biflows[quintupla]['pay_length'].append(traffic_biflows[quintupla]['pay_length'][i])
            else:
                malign_biflows[quintupla]['ttl'].append(traffic_biflows[quintupla]['ttl'][i])
                malign_biflows[quintupla]['TCP_Window'].append(traffic_biflows[quintupla]['TCP_Window'][i])
                malign_biflows[quintupla]['timestamp'].append(traffic_biflows[quintupla]['timestamp'][i])
                malign_biflows[quintupla]['pkt_length'].append(traffic_biflows[quintupla]['pkt_length'][i])
                malign_biflows[quintupla]['iat'].append(traffic_biflows[quintupla]['iat'][i])
                malign_biflows[quintupla]['l4_payload'].append(traffic_biflows[quintupla]['l4_payload'][i])
                malign_biflows[quintupla]['pay_length'].append(traffic_biflows[quintupla]['pay_length'][i])
        if np.size(benign_biflows[quintupla]['pkt_length'])>0 and np.size(malign_biflows[quintupla]['pkt_length'])>0:
            mix_biflows[quintupla]= {
                'ttl': [],
                'TCP_Window':[],
                'timestamp' : [],
                'pkt_length' : [],
                'iat' : [],
                'l4_payload' : [],
                'pay_length' : [],
                'malign' : []  
            }
            for i in range(np.size(traffic_biflows[quintupla]['malign'])):
                mix_biflows[quintupla]['ttl'].append(traffic_biflows[quintupla]['ttl'][i])
                mix_biflows[quintupla]['TCP_Window'].append(traffic_biflows[quintupla]['TCP_Window'][i])
                mix_biflows[quintupla]['timestamp'].append(traffic_biflows[quintupla]['timestamp'][i])
                mix_biflows[quintupla]['pkt_length'].append(traffic_biflows[quintupla]['pkt_length'][i])
                mix_biflows[quintupla]['iat'].append(traffic_biflows[quintupla]['iat'][i])
                mix_biflows[quintupla]['l4_payload'].append(traffic_biflows[quintupla]['l4_payload'][i])
                mix_biflows[quintupla]['pay_length'].append(traffic_biflows[quintupla]['pay_length'][i])
                mix_biflows[quintupla]['malign'].append(traffic_biflows[quintupla]['malign'][i])

        if np.size(benign_biflows[quintupla]['pkt_length'])==0:
            del benign_biflows[quintupla]
        if np.size(malign_biflows[quintupla]['pkt_length'])==0:
            del malign_biflows[quintupla]

    pickle.dump(mix_biflows,open('Biflussi/'+str(string)+'/mix_biflows_'+str(string)+'.p','wb'))
    pickle.dump(benign_biflows,open('Biflussi/'+str(string)+'/benign_biflows_'+str(string)+'.p','wb'))
    pickle.dump(malign_biflows,open('Biflussi/'+str(string)+'/malign_biflows_'+str(string)+'.p','wb'))
    
#Mix_Biflows_Division divide i biflussi misti in biflussi benigni o malevoli tramite un filtro
def Mix_Biflows_Division(string):
    c=pickle.load(open("Biflussi/"+str(string)+"/mix_biflows_"+str(string)+".p","rb"))
    mix = dict()
    for quintupla in c:
        mix[quintupla] = {
            'pkt_benign': 0,
            'pkt_malign': 0
        }
    
        for i in range(len(c[quintupla]['malign'])):
            if c[quintupla]['malign'][i] == 0:
                mix[quintupla]['pkt_benign'] += 1
            else:
                mix[quintupla]['pkt_malign'] += 1
            
    mix_buoni = dict()
    mix_malign = dict()

    for quintupla in mix :
        mix_buoni[quintupla] = {
        'ttl':[],
        'TCP_Window':[],
        'timestamp': [],
        'pkt_length': [],
        'iat': [],
        'l4_payload': [],
        'pay_length': [],
        'malign': []
        }
        mix_malign[quintupla] = {
        'ttl':[],
        'TCP_Window':[],
        'timestamp': [],
        'pkt_length': [],
        'iat': [],
        'l4_payload': [],
        'pay_length': [],
        'malign': []
        }
    
    
        for i in range(np.size(c[quintupla]['malign'])):
            if (mix[quintupla]['pkt_malign']==1 and mix[quintupla]['pkt_benign']<20) or (mix[quintupla]['pkt_malign'] / mix[quintupla]['pkt_benign'] < 0.01):
                mix_buoni[quintupla]['ttl'].append(c[quintupla]['ttl'][i])
                mix_buoni[quintupla]['TCP_Window'].append(c[quintupla]['TCP_Window'][i])
                mix_buoni[quintupla]['timestamp'].append(c[quintupla]['timestamp'][i])
                mix_buoni[quintupla]['pkt_length'].append(c[quintupla]['pkt_length'][i])
                mix_buoni[quintupla]['iat'].append(c[quintupla]['iat'][i])
                mix_buoni[quintupla]['l4_payload'].append(c[quintupla]['l4_payload'][i])
                mix_buoni[quintupla]['pay_length'].append(c[quintupla]['pay_length'][i])
            else:
                mix_malign[quintupla]['ttl'].append(c[quintupla]['ttl'][i])
                mix_malign[quintupla]['TCP_Window'].append(c[quintupla]['TCP_Window'][i])
                mix_malign[quintupla]['timestamp'].append(c[quintupla]['timestamp'][i])
                mix_malign[quintupla]['pkt_length'].append(c[quintupla]['pkt_length'][i])
                mix_malign[quintupla]['iat'].append(c[quintupla]['iat'][i])
                mix_malign[quintupla]['l4_payload'].append(c[quintupla]['l4_payload'][i])
                mix_malign[quintupla]['pay_length'].append(c[quintupla]['pay_length'][i])

        if np.size(mix_buoni[quintupla]['timestamp']) == 0:
            del mix_buoni[quintupla]
        if np.size(mix_malign[quintupla]['timestamp']) == 0:
            del mix_malign[quintupla]
            
    pickle.dump(mix_buoni,open('Biflussi/'+str(string)+'/mix_benign_biflows_'+str(string)+'.p','wb'))
    pickle.dump(mix_malign,open('Biflussi/'+str(string)+'/mix_malign_biflows_'+str(string)+'.p','wb'))
    
    
def load_biflows(string):
    
    benign_biflows=pickle.load(open("Biflussi/"+str(string)+"/benign_biflows_"+str(string)+".p","rb"))
    malign_biflows=pickle.load(open("Biflussi/"+str(string)+"/malign_biflows_"+str(string)+".p","rb"))
    mix_biflows=pickle.load(open("Biflussi/"+str(string)+"/mix_biflows_"+str(string)+".p","rb"))
    mix_b_biflows = pickle.load(open("Biflussi/"+str(string)+"/mix_benign_biflows_"+str(string)+".p","rb"))
    mix_m_biflows = pickle.load(open("Biflussi/"+str(string)+"/mix_malign_biflows_"+str(string)+".p","rb"))
    return benign_biflows,malign_biflows,mix_biflows,mix_b_biflows,mix_m_biflows

#Aggiungi_filtraggio prende i misti buoni e i misti cattivi e li aggiunge ai biflussi buoni e cattivi salvati in precedenza
def Aggiungi_filtraggio(string):
    a,b,c,d,e = load_biflows(string)
    
    for quintupla in c:
        del a[quintupla]
        del b[quintupla]
    for quintupla in d:
        a[quintupla] = d[quintupla]

    for quintupla in e:
        b[quintupla] = e[quintupla]
        
    pickle.dump(a,open('Biflussi/'+str(string)+'/final_benign_biflows_'+str(string)+'.p','wb'))
    pickle.dump(b,open('Biflussi/'+str(string)+'/final_malign_biflows_'+str(string)+'.p','wb'))
    
#extract_statistics calcola varie statistiche relative alle features di ogni singolo biflusso
def extract_statistics(packettino):
    packet=list(map(float,packettino))
    
    try:
        packet_field_statistics = dict()
        packet_field_statistics['min'] = np.min(packet)
        packet_field_statistics['max'] = np.max(packet)
        packet_field_statistics['mean'] = np.mean(packet)
        packet_field_statistics['std'] = np.std(packet)
        packet_field_statistics['var'] = np.var(packet)
        packet_field_statistics['mad'] = robust.mad(packet,c=1)
        packet_field_statistics['skew'] = scipy.stats.skew(packet,bias=False)
        packet_field_statistics['kurtosis'] = scipy.stats.kurtosis(packet,bias=False)
        packet_field_statistics['10_percentile'] = np.percentile(packet,10)
        packet_field_statistics['20_percentile'] = np.percentile(packet,20)
        packet_field_statistics['30_percentile'] = np.percentile(packet,30)
        packet_field_statistics['40_percentile'] = np.percentile(packet,40)
        packet_field_statistics['50_percentile'] = np.percentile(packet,50)
        packet_field_statistics['60_percentile'] = np.percentile(packet,60)
        packet_field_statistics['70_percentile'] = np.percentile(packet,70)
        packet_field_statistics['80_percentile'] = np.percentile(packet,80)
        packet_field_statistics['90_percentile'] = np.percentile(packet,90)
        
    except ValueError:
        packet_field_statistics = dict()
        packet_field_statistics['min'] = np.nan
        packet_field_statistics['max'] = np.nan
        packet_field_statistics['mean'] = np.nan
        packet_field_statistics['std'] = np.nan
        packet_field_statistics['var'] = np.nan
        packet_field_statistics['mad'] = np.nan
        packet_field_statistics['skew'] = np.nan
        packet_field_statistics['kurtosis'] = np.nan
        packet_field_statistics['10_percentile'] = np.nan
        packet_field_statistics['20_percentile'] = np.nan
        packet_field_statistics['30_percentile'] = np.nan
        packet_field_statistics['40_percentile'] = np.nan
        packet_field_statistics['50_percentile'] = np.nan
        packet_field_statistics['60_percentile'] = np.nan
        packet_field_statistics['70_percentile'] = np.nan
        packet_field_statistics['80_percentile'] = np.nan
        packet_field_statistics['90_percentile'] = np.nan
        sys.stderr.write('WARNING\n!')
    
    return packet_field_statistics
    
#EstraiStatisticheBiflusso Estrae le statistiche di ogni biflusso prima benigni e poi malevoli e li raggruppa salvando poi il tutto in un csv delle statistiche
#dei biflussi benigni e poi quelli malevoli
def EstraiStatisticheBiflusso(string):
    benign_biflows=pickle.load(open('Biflussi/'+str(string)+'/final_benign_biflows_'+str(string)+'.p','rb'))
    malign_biflows=pickle.load(open('Biflussi/'+str(string)+'/final_malign_biflows_'+str(string)+'.p','rb'))
    pkt_length_tot_quintupla=0
    n_pkt_quintupla=0
    statistics=dict()
    statistiche_biflusso=dict()
    for (pkt,stringa) in [(benign_biflows,"benign_biflows"),(malign_biflows,"malign_biflows")]:
        for quintupla in pkt:
            statistiche_biflusso[quintupla]={
            'num_pkt':int,
            'tot_pay_length':float,
            'mean_pay_length': float,
            'std_pay_length':float,
            'max_pay_length':float,
            'min_pay_length':float,
            'mad_pay_length': float,
            'kurtosis_pay_length':float,
            'skew_pay_length':float,
            'var_pay_length':float,
            '10_percentile_pay_length': float,
            '20_percentile_pay_length':float,
            '30_percentile_pay_length':float,
            '40_percentile_pay_length':float,
            '50_percentile_pay_length': float,
            '60_percentile_pay_length':float,
            '70_percentile_pay_length':float,
            '80_percentile_pay_length':float,
            '90_percentile_pay_length':float,
            'mean_iat':float,
            'std_iat':float,
            'max_iat':float,
            'min_iat':float,
            'mad_iat': float,
            'kurtosis_iat':float,
            'skew_iat':float,
            'var_iat':float,
            '10_percentile_iat': float,
            '20_percentile_iat':float,
            '30_percentile_iat':float,
            '40_percentile_iat':float,
            '50_percentile_iat': float,
            '60_percentile_iat':float,
            '70_percentile_iat':float,
            '80_percentile_iat':float,
            '90_percentile_iat':float,
            'byte_rate':float,
            'mean_ttl':float,
            'std_ttl':float,
            'max_ttl':float,
            'min_ttl':float,
            'mad_ttl': float,
            'kurtosis_ttl':float,
            'skew_ttl':float,
            'var_ttl':float,
            '10_percentile_ttl': float,
            '20_percentile_ttl':float,
            '30_percentile_ttl':float,
            '40_percentile_ttl':float,
            '50_percentile_ttl': float,
            '60_percentile_ttl':float,
            '70_percentile_ttl':float,
            '80_percentile_ttl':float,
            '90_percentile_ttl':float,
            'mean_TCP_Window':float,
            'std_TCP_Window':float,
            'max_TCP_Window':float,
            'min_TCP_Window':float,
            'mad_TCP_Window': float,
            'kurtosis_TCP_Window':float,
            'skew_TCP_Window':float,
            'var_TCP_Window':float,
            '10_percentile_TCP_Window': float,
            '20_percentile_TCP_Window':float,
            '30_percentile_TCP_Window':float,
            '40_percentile_TCP_Window':float,
            '50_percentile_TCP_Window': float,
            '60_percentile_TCP_Window':float,
            '70_percentile_TCP_Window':float,
            '80_percentile_TCP_Window':float,
            '90_percentile_TCP_Window':float,
        }
            statistiche_biflusso[quintupla]['num_pkt']=len(pkt[quintupla]['timestamp'])
            for i in range(0,len(pkt[quintupla]['timestamp'])):
                pkt_length_tot_quintupla+=pkt[quintupla]['pay_length'][i]


            statistiche_biflusso[quintupla]['tot_pay_length']=pkt_length_tot_quintupla
            statistics[quintupla] = extract_statistics(pkt[quintupla]['pay_length'])
            statistiche_biflusso[quintupla]['mean_pay_length']=statistics[quintupla]['mean']
            statistiche_biflusso[quintupla]['std_pay_length']=statistics[quintupla]['std']
            statistiche_biflusso[quintupla]['min_pay_length']=statistics[quintupla]['min']
            statistiche_biflusso[quintupla]['max_pay_length']=statistics[quintupla]['max']
            statistiche_biflusso[quintupla]['mad_pay_length']=statistics[quintupla]['mad']
            statistiche_biflusso[quintupla]['kurtosis_pay_length']=statistics[quintupla]['kurtosis']
            statistiche_biflusso[quintupla]['skew_pay_length']=statistics[quintupla]['skew']
            statistiche_biflusso[quintupla]['var_pay_length']=statistics[quintupla]['var']
            statistiche_biflusso[quintupla]['10_percentile_pay_length']=statistics[quintupla]['10_percentile']
            statistiche_biflusso[quintupla]['20_percentile_pay_length']=statistics[quintupla]['20_percentile']
            statistiche_biflusso[quintupla]['30_percentile_pay_length']=statistics[quintupla]['30_percentile']
            statistiche_biflusso[quintupla]['40_percentile_pay_length']=statistics[quintupla]['40_percentile']
            statistiche_biflusso[quintupla]['50_percentile_pay_length']=statistics[quintupla]['50_percentile']
            statistiche_biflusso[quintupla]['60_percentile_pay_length']=statistics[quintupla]['60_percentile']
            statistiche_biflusso[quintupla]['70_percentile_pay_length']=statistics[quintupla]['70_percentile']
            statistiche_biflusso[quintupla]['80_percentile_pay_length']=statistics[quintupla]['80_percentile']
            statistiche_biflusso[quintupla]['90_percentile_pay_length']=statistics[quintupla]['90_percentile']
            

            statistics[quintupla] = extract_statistics(pkt[quintupla]['iat'])
            statistiche_biflusso[quintupla]['mean_iat']=statistics[quintupla]['mean']
            statistiche_biflusso[quintupla]['std_iat']=statistics[quintupla]['std']
            statistiche_biflusso[quintupla]['min_iat']=statistics[quintupla]['min']
            statistiche_biflusso[quintupla]['max_iat']=statistics[quintupla]['max']
            statistiche_biflusso[quintupla]['mad_iat']=statistics[quintupla]['mad']
            statistiche_biflusso[quintupla]['kurtosis_iat']=statistics[quintupla]['kurtosis']
            statistiche_biflusso[quintupla]['skew_iat']=statistics[quintupla]['skew']
            statistiche_biflusso[quintupla]['var_iat']=statistics[quintupla]['var']
            statistiche_biflusso[quintupla]['10_percentile_iat']=statistics[quintupla]['10_percentile']
            statistiche_biflusso[quintupla]['20_percentile_iat']=statistics[quintupla]['20_percentile']
            statistiche_biflusso[quintupla]['30_percentile_iat']=statistics[quintupla]['30_percentile']
            statistiche_biflusso[quintupla]['40_percentile_iat']=statistics[quintupla]['40_percentile']
            statistiche_biflusso[quintupla]['50_percentile_iat']=statistics[quintupla]['50_percentile']
            statistiche_biflusso[quintupla]['60_percentile_iat']=statistics[quintupla]['60_percentile']
            statistiche_biflusso[quintupla]['70_percentile_iat']=statistics[quintupla]['70_percentile']
            statistiche_biflusso[quintupla]['80_percentile_iat']=statistics[quintupla]['80_percentile']
            statistiche_biflusso[quintupla]['90_percentile_iat']=statistics[quintupla]['90_percentile']

            statistics[quintupla] = extract_statistics(pkt[quintupla]['ttl'])
            statistiche_biflusso[quintupla]['mean_ttl']=statistics[quintupla]['mean']
            statistiche_biflusso[quintupla]['std_ttl']=statistics[quintupla]['std']
            statistiche_biflusso[quintupla]['min_ttl']=statistics[quintupla]['min']
            statistiche_biflusso[quintupla]['max_ttl']=statistics[quintupla]['max']
            statistiche_biflusso[quintupla]['mad_ttl']=statistics[quintupla]['mad']
            statistiche_biflusso[quintupla]['kurtosis_ttl']=statistics[quintupla]['kurtosis']
            statistiche_biflusso[quintupla]['skew_ttl']=statistics[quintupla]['skew']
            statistiche_biflusso[quintupla]['var_ttl']=statistics[quintupla]['var']
            statistiche_biflusso[quintupla]['10_percentile_ttl']=statistics[quintupla]['10_percentile']
            statistiche_biflusso[quintupla]['20_percentile_ttl']=statistics[quintupla]['20_percentile']
            statistiche_biflusso[quintupla]['30_percentile_ttl']=statistics[quintupla]['30_percentile']
            statistiche_biflusso[quintupla]['40_percentile_ttl']=statistics[quintupla]['40_percentile']
            statistiche_biflusso[quintupla]['50_percentile_ttl']=statistics[quintupla]['50_percentile']
            statistiche_biflusso[quintupla]['60_percentile_ttl']=statistics[quintupla]['60_percentile']
            statistiche_biflusso[quintupla]['70_percentile_ttl']=statistics[quintupla]['70_percentile']
            statistiche_biflusso[quintupla]['80_percentile_ttl']=statistics[quintupla]['80_percentile']
            statistiche_biflusso[quintupla]['90_percentile_ttl']=statistics[quintupla]['90_percentile']

            statistics[quintupla] = extract_statistics(pkt[quintupla]['TCP_Window'])
            statistiche_biflusso[quintupla]['mean_TCP_Window']=statistics[quintupla]['mean']
            statistiche_biflusso[quintupla]['std_TCP_Window']=statistics[quintupla]['std']
            statistiche_biflusso[quintupla]['min_TCP_Window']=statistics[quintupla]['min']
            statistiche_biflusso[quintupla]['max_TCP_Window']=statistics[quintupla]['max']
            statistiche_biflusso[quintupla]['mad_TCP_Window']=statistics[quintupla]['mad']
            statistiche_biflusso[quintupla]['kurtosis_TCP_Window']=statistics[quintupla]['kurtosis']
            statistiche_biflusso[quintupla]['skew_TCP_Window']=statistics[quintupla]['skew']
            statistiche_biflusso[quintupla]['var_TCP_Window']=statistics[quintupla]['var']
            statistiche_biflusso[quintupla]['10_percentile_TCP_Window']=statistics[quintupla]['10_percentile']
            statistiche_biflusso[quintupla]['20_percentile_TCP_Window']=statistics[quintupla]['20_percentile']
            statistiche_biflusso[quintupla]['30_percentile_TCP_Window']=statistics[quintupla]['30_percentile']
            statistiche_biflusso[quintupla]['40_percentile_TCP_Window']=statistics[quintupla]['40_percentile']
            statistiche_biflusso[quintupla]['50_percentile_TCP_Window']=statistics[quintupla]['50_percentile']
            statistiche_biflusso[quintupla]['60_percentile_TCP_Window']=statistics[quintupla]['60_percentile']
            statistiche_biflusso[quintupla]['70_percentile_TCP_Window']=statistics[quintupla]['70_percentile']
            statistiche_biflusso[quintupla]['80_percentile_TCP_Window']=statistics[quintupla]['80_percentile']
            statistiche_biflusso[quintupla]['90_percentile_TCP_Window']=statistics[quintupla]['90_percentile']

            if len(pkt[quintupla]['timestamp'])>1:
                statistiche_biflusso[quintupla]['byte_rate']=pkt_length_tot_quintupla/(pkt[quintupla]['timestamp'][-1]-pkt[quintupla]['timestamp'][0])
            else:
                statistiche_biflusso[quintupla]['byte_rate']=0

            pkt_length_tot_quintupla=0
            

        data=pd.DataFrame.from_dict(statistiche_biflusso,orient='index')
        data.columns = data.columns.astype(str)
        data.to_csv('Statistiche/'+str(string)+'/Statistiche_Biflussi_'+str(stringa)+'.csv')
        statistiche_biflusso.clear()
        
        

    
    
                    
