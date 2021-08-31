import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt


def StampaF1_score_Tprs():
    for percent_poisoning in (0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10):
        tprs=np.zeros(10)
        f1_score=np.zeros(10)
        for Fold in range(10):
            dataset=pd.read_csv('D:/IDALAB/Training/Autoencoder/Best/'+str(percent_poisoning)+'%_poisoning/'+str(Fold+1)+'Fold.csv', delimiter=',')
            dataset=dataset.to_numpy()
            dataset=np.delete(dataset,0,axis=1)
            fpr, tpr,thresholds= metrics.roc_curve(dataset[:,0],dataset[:,2])
            indices=np.where(fpr>=0.01)
            index=np.min(indices)
            tprs[Fold]=tpr[index]
            f1_score[Fold]=metrics.f1_score(dataset[:,0],dataset[:,1],average='macro')
        dataset=pd.DataFrame({'f1_score': f1_score, 'tprs': tprs})
        dataset.to_csv('D:/IDALAB/Training/Autoencoder/Best/'+str(percent_poisoning)+'%_poisoning/F1_Score_Tprs_'+str(percent_poisoning)+'%.csv',index=False,sep=',')
        
def StampaRoc_CurveConfigurazione():
    fig=plt.figure(1)
    for percent_poisoning in (0,1,2,2.5,3,3.5,4,4.5,5):
        tprs=[]
        mean_fpr = np.linspace(0, 1, 500)
        for Fold in range(10):
            dataset=pd.read_csv('D:/IDALAB/Codice/Statistiche/Kitsune/Poisoning/'+str(percent_poisoning)+'%_poisoning/'+str(Fold+1)+'Fold.csv', delimiter=',')
            dataset=dataset.to_numpy()
            dataset=np.delete(dataset,0,axis=1)
            fpr,tpr,thresholds= metrics.roc_curve(dataset[:,0],dataset[:,2])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr= np.std(tprs,axis=0)
        plt.plot(mean_fpr, mean_tpr,
        label='Mean ROC '+str(percent_poisoning)+'% Poisoning')
        plt.fill_between(mean_fpr,mean_tpr-std_tpr,mean_tpr+std_tpr,
        label='Std ROC '+str(percent_poisoning)+'% poisoning',alpha=0.2)
        plt.xscale('symlog',linthresh=1e-2)
        plt.ylim([0, 1.01])
        plt.xticks()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC CURVE Kitsune Model')
        plt.legend(loc='lower right',fontsize=7.5)
    plt.show()
    fig.savefig('D:/IDALAB/Codice/Statistiche/Kitsune/ROC_CURVE_Kitsune.png')
    
def Stampa_F1_Score_finale():
    f1_score_Autoencoder=np.zeros(11)
    f1_score_Deep_Autoencoder=np.zeros(11)
    f1_score_Kitsune=np.zeros(11)
    for i in range(0,11):
        dataset1 = pd.read_csv('Statistiche/Autoencoder/F1_Score_Tprs_'+str(i)+'%.csv')
        dataset1=dataset1.to_numpy()
        f1_score_Autoencoder[i]=np.mean(dataset1[:,0])
        dataset2 = pd.read_csv('Statistiche/Deep_Autoencoder/F1_Score_Tprs_'+str(i)+'%.csv')
        dataset2 = dataset2.to_numpy()
        f1_score_Deep_Autoencoder[i]=np.mean(dataset2[:,0])
        dataset3 = pd.read_csv('Statistiche/Kitsune/Poisoning/'+str(i)+'%_Poisoning/F1_Score_Tprs_'+str(i)+'%.csv')
        dataset3 = dataset3.to_numpy()
        f1_score_Kitsune[i]=np.mean(dataset3[:,0])


    x = [0,1,2,3,4,5,6,7,8,9,10]

    y1 = list()
    y2 = list()
    y3 = list()
    for i in range(0,11):
        y1.append(f1_score_Autoencoder[i])
        y2.append(f1_score_Deep_Autoencoder[i])
        y3.append(f1_score_Kitsune[i])
    y = np.zeros([3,11])
    y[0] = y1
    y[1] = y2
    y[2] = y3

    plt.xticks(np.arange(0, 11, step=1))
    plt.title('F1_Score Comparison')
    plt.xlim(0,10)
    plt.ylim(0.3,1)
    plt.xlabel('poisoning [%]')
    plt.ylabel('f1_score')
    plt.plot(x,y[0],label='autoencoder',marker = "o")
    plt.plot(x,y[1],label='deep_autoencoder', marker = "o")
    plt.plot(x,y[2],label='kitsune', marker = "o")
    plt.legend(loc='upper right',ncol=1,fontsize=12)
    plt.savefig('Statistiche/plot_f1.png')
    
    
def plot_f1(string):
    path = "D:/IDALAB/Training/"
    fig, ax = plt.subplots()
    f1_list = list()
    for n_feat in selected_features.keys():
        dataset = pd.read_csv(str(path)+str(string)+'/fix/plot_{}.csv'.format(n_feat))
        dataset = dataset.to_numpy()
        f1_score = metrics.f1_score(dataset[:,1],dataset[:,2])
        f1_list.append(f1_score)
        
    bar_width = 0.7
    index = np.arange(len(selected_features.keys()))
    #f1_score_values = ax.bar(index, f1_list,bar_width,label='F1_Score')
    ax.set_xlabel('N_Features')
    ax.set_ylabel('F1_Score')
    ax.set_xticks(index + bar_width /2)
    ax.set_xticklabels(selected_features.keys())
    ax.text(bar_width,bar_width,f1_list)
    ax.legend()
    plt.show()
    fig.savefig(str(path)+'F1_SCORE'+str(string))
    

def plot_roc(string):
    path = "D:/IDALAB/Training/"
    fig=plt.figure(1)
    for n_feat in selected_features.keys():
        mean_fpr = np.linspace(0,1,500)
        dataset = pd.read_csv(str(path)+str(string)+'/fix/plot_{}.csv'.format(n_feat))
        dataset = dataset.to_numpy()
        fpr,tpr,thresholds = metrics.roc_curve(dataset[:,1],dataset[:,3])
        interp_tpr = np.interp(mean_fpr,fpr,tpr)
        interp_tpr[0] = 0.0
        plt.plot([0,1],[0,1],'k--')
        interp_tpr[-1] = 1
        plt.plot(mean_fpr,interp_tpr,label = 'ROC CURVE'+str(n_feat)+'Features')
        plt.xscale('symlog',linthresh=1e-2)
        plt.ylim([0, 1.01])
        plt.xticks()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC CURVE '+str(string))
        plt.legend(loc='lower right',ncol=1,fontsize=9)
    plt.savefig(str(path)+'ROC_CURVE'+str(string))
    plt.show()
    
def Model_Comparison():
    f1_score_Autoencoder=np.zeros(20)
    f1_std_score_Autoencoder=np.zeros(20)
    f1_std_score_Deep=np.zeros(20)
    f1_score_Deep_Autoencoder=np.zeros(20)
    f1_score_Kitsune=np.zeros(20)
    f1_std_score_Kitsune=np.zeros(20)
    i=0
    for percent_poisoning in (0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10):
        dataset1 = pd.read_csv('D:/IDALAB/New_Training/Autoencoder/Poisoning/'+str(percent_poisoning)+'%/F1_Score_Tprs_'+str(percent_poisoning)+'%.csv')
        dataset1=dataset1.to_numpy()
        f1_score_Autoencoder[i]=np.mean(dataset1[:,0])
        f1_std_score_Autoencoder[i]=np.std(dataset1[:,0])
        dataset2 = pd.read_csv('D:/IDALAB/New_Training/Deep_Autoencoder/Poisoning/'+str(percent_poisoning)+'%/F1_Score_Tprs_'+str(percent_poisoning)+'%.csv')
        dataset2 = dataset2.to_numpy()
        f1_score_Deep_Autoencoder[i]=np.mean(dataset2[:,0])
        f1_std_score_Deep[i]=np.std(dataset2[:,0])
        dataset3 = pd.read_csv('D:/IDALAB/New_Training/Kitsune/Poisoning/'+str(percent_poisoning)+'%_Poisoning/F1_Score_Tprs_'+str(percent_poisoning)+'%.csv')
        dataset3 = dataset3.to_numpy()
        f1_score_Kitsune[i]=np.mean(dataset3[:,0])
        f1_std_score_Kitsune[i]= np.std(dataset3[:,0])
        i+=1

    x = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]

    y1 = list()
    y2 = list()
    y3 = list()
    for i in range(20):
        y1.append(f1_score_Autoencoder[i])
        y2.append(f1_score_Deep_Autoencoder[i])
        y3.append(f1_score_Kitsune[i])
    y = np.zeros([3,20])
    y[0] = y1
    y[1] = y2
    y[2] = y3

    #fig, ax = plt.subplots(figsize=(12,6))
    plt.xticks(np.arange(0.5, 11, step=0.5))
    plt.title('F1_Score Comparison')
    plt.xlim(1,10)
    plt.ylim(0.3,1)
    plt.xlabel('poisoning [%]')
    plt.ylabel('f1_score')
    plt.plot(x,y[0],label='autoencoder',marker="o")
    plt.plot(x,y[1],label='deep_autoencoder',marker="o")
    plt.plot(x,y[2],label='kitsune',marker="o")
    #plt.errorbar(x,y[0],label='autoencoder',yerr=f1_std_score_Autoencoder,marker = "o")
    #plt.errorbar(x,y[1],label='deep_autoencoder',yerr=f1_std_score_Deep,marker = "o")
    #plt.errorbar(x,y[2],label='kitsune',yerr=f1_std_score_Kitsune, marker = "o")
    #plt.set_xtickslabels([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10])
    plt.legend(loc='best',ncol=1,fontsize=10)
    plt.savefig('D:/IDALAB/New_Trainingplot_f1_comparison.png')
