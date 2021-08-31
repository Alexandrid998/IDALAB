import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import f1_score

def n_features_selection(x,y):
    num_feats = [x.shape[1],65,60,55,50,45,40,35,30,25,20,15,10,5]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(x)
    X_norm = scaler.transform(x)
    selected_features = list()
    for n_feats in num_feats:
        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=n_feats, step=10, verbose=2)
        rfe_selector.fit(X_norm, y)
        rfe_support = rfe_selector.get_support()
        rfe_feature = x.loc[:,rfe_support].columns.tolist()
        selected_features.append(rfe_feature)
        
    
    features = dict()
    for i in range(0,len(selected_features)):
        features[num_feats[i]] ={
            'selected_features':int
        }
        features[num_feats[i]] = selected_features[i]
        
    return features

#selected_features è il ritorno dalla funzione n_features_selection   
def Kitsune_Features(selected_features):
    Skf=StratifiedKFold(n_splits=10,shuffle=True)

    for n in (5,10,15,20,25,30,35,40,45,50,55,60,65,71):
        train_data_n = pd.read_csv("Statistiche/Biflussi_Totali/training_set.csv")
        train_data_n = np.array(train_data_n)
        features=selected_features[n]
        train_data_labels=train_data_n[:,71]
        train_data_n=train_data_n[:,features]
        iteration=0
        for train_index, test_index in Skf.split(train_data_n,train_data_labels):
            #for percent_poisoning in (1,2,3,4,5,6,7,8,9,10):
            train_index=train_index.astype('int32')
            test_index=test_index.astype('int32')
            x_train1=train_data_n[train_index]
            y_train = train_data_labels[train_index]
            x_test=train_data_n[test_index]
            y_test=train_data_labels[test_index]
            y_test=y_test.astype('int')
        #Costruisco Training e Test Set con un dataset sporcato del percent_poisoning di traffico malevolo
            #benign_x_train = x_train1[(y_train[:]==0)]
            x_train = x_train1[(y_train[:]==0)]
            #malign_x_train = x_train1[(y_train[:]==1)]
            #x_train=np.concatenate([benign_x_train, malign_x_train[:math.floor(percent_poisoning/100*benign_x_train.shape[0])]],axis=0)
            #np.random.shuffle(x_train)
            training_features = x_train
            test_features = x_test
            n_autoencoder=5
            n_autoencoder1=x_train.shape[1]%n_autoencoder
            n_features2=math.floor(x_train.shape[1]/n_autoencoder)
            n_features1=n_features2+1
            n_autoencoder2=n_autoencoder-n_autoencoder1
            Ensemble1=np.empty(n_autoencoder1,dtype=object) #Ensemble layer1
            Ensemble2=np.empty(n_autoencoder2,dtype=object) #Ensemble layer2
            for i in range(n_autoencoder1):
                Ensemble1[i]= Sequential()
                Ensemble1[i].add(Dense(units=n_features1,activation='relu',input_shape=(n_features1,)))
                Ensemble1[i].add(Dense(units=math.floor(0.75*n_features1),activation='relu'))
                Ensemble1[i].add(Dense(units=n_features1,activation='sigmoid'))
                Ensemble1[i].compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
            for i in range(n_autoencoder2):
                Ensemble2[i]= Sequential()
                Ensemble2[i].add(Dense(units=n_features2,activation='relu',input_shape=(n_features2,)))
                Ensemble2[i].add(Dense(units=math.floor(0.75*n_features2),activation='relu'))
                Ensemble2[i].add(Dense(units=n_features2,activation='sigmoid'))
                Ensemble2[i].compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
            #Creo un unico autoencoder di output che ha una dimensione di ingresso pari al n totale di autoencoder nell'ensemble layer
            Output= Sequential()
            Output.add(Dense(units=n_autoencoder,activation='relu',input_shape=(n_autoencoder,)))
            Output.add(Dense(units=math.floor(0.75*n_autoencoder),activation='relu'))
            Output.add(Dense(units=n_autoencoder,activation='sigmoid'))
            Output.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
            scaler1=MinMaxScaler(feature_range=(0,1))
            training_features=scaler1.fit_transform(training_features)
            #faccio la fit degli autoencoder 
            for i in range(n_autoencoder1):
                Ensemble1[i].fit(training_features[:,i*n_features1:(i+1)*n_features1],training_features[:,i*n_features1:(i+1)*n_features1], epochs=1, batch_size=32)
            for i in range(n_autoencoder2):
                Ensemble2[i].fit(training_features[:,n_autoencoder1*n_features1+i*n_features2:n_autoencoder1*n_features1+(i+1)*n_features2],training_features[:,n_autoencoder1*n_features1+i*n_features2:n_autoencoder1*n_features1+(i+1)*n_features2], epochs=1, batch_size=32)
            score=np.zeros((training_features.shape[0],n_autoencoder))
            #calcolo gli RMSE centrali del livello ensemble da dare all'Output layer
            for j in range(n_autoencoder1):
                pred=Ensemble1[j].predict(training_features[:,j*n_features1:(j+1)*n_features1])
                for i in range(training_features.shape[0]):
                    score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],training_features[i,j*n_features1:(j+1)*n_features1]))
            for j in range(n_autoencoder2):
                pred=Ensemble2[j].predict(training_features[:,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2])
                for i in range(training_features.shape[0]):
                    score[i,j+n_autoencoder1]= np.sqrt(metrics.mean_squared_error(pred[i],training_features[i,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2]))
            #faccio fare la fit all'Output layer tramite gli RMSE calcolati precedentemente
            scaler2=MinMaxScaler(feature_range=(0,1))
            score=scaler2.fit_transform(score)
            Output.fit(score,score,epochs=1,batch_size=32)
            RMSE=np.zeros(score.shape[0])
            pred=Output.predict(score)
            for i in range(score.shape[0]):
                RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],score[i]))
            # FASE DI TESTING
            test_features=scaler1.transform(test_features)
            test_score=np.zeros((test_features.shape[0],n_autoencoder))
            #calcolo RMSE in fase di Testing
            for j in range(n_autoencoder1):
                pred=Ensemble1[j].predict(test_features[:,j*n_features1:(j+1)*n_features1])
                for i in range(test_features.shape[0]):
                    test_score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],test_features[i,j*n_features1:(j+1)*n_features1]))
            for j in range(n_autoencoder2):
                pred=Ensemble2[j].predict(test_features[:,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2])
                for i in range(test_features.shape[0]):
                    test_score[i,j+n_autoencoder1]= np.sqrt(metrics.mean_squared_error(pred[i],test_features[i,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2]))
            #calcolo RMSE FINALE
            test_score=scaler2.transform(test_score)
            RMSE=np.zeros(test_score.shape[0])
            pred=Output.predict(test_score)
            for i in range(test_score.shape[0]):
                RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],test_score[i]))
            fpr, tpr,thresholds= metrics.roc_curve(y_test,RMSE)
            indices=np.where(fpr>=0.01)
            index=np.min(indices)
            soglia=thresholds[index]
            labels=np.zeros(RMSE.shape[0])
            for i in range(RMSE.shape[0]):
                if RMSE[i] < soglia:
                    labels[i] = 0
                else:
                    labels[i] = 1
            for_plot = dict()
            for_plot = {'y_true': y_test, 'y_pred': labels, 'RMSE': RMSE}
            plot_df = pd.DataFrame.from_dict(for_plot)
            plot_df.to_csv("Statistiche/Kitsune/1_epoch/"+str(n)+"/"+str(iteration+1)+"Fold.csv")
            iteration+=1
            
def Kitsune_Poisoning(selected_features):
    train_data_n = pd.read_csv("Statistiche/Biflussi_Totali/training_set.csv")
    train_data_n = np.array(train_data_n)
    features=selected_features[71]
    train_data_labels=train_data_n[:,71]
    train_data_n=train_data_n[:,features]
    iteration=0
    for train_index, test_index in Skf.split(train_data_n,train_data_labels):
        for percent_poisoning in (0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10):
            train_index=train_index.astype('int32')
            test_index=test_index.astype('int32')
            x_train1=train_data_n[train_index]
            y_train = train_data_labels[train_index]
            x_test=train_data_n[test_index]
            y_test=train_data_labels[test_index]
            y_test=y_test.astype('int')
        #Costruisco Training e Test Set con un dataset sporcato del percent_poisoning di traffico malevolo
            benign_x_train = x_train1[(y_train[:]==0)]
            malign_x_train = x_train1[(y_train[:]==1)]
            x_train=np.concatenate([benign_x_train, malign_x_train[:math.floor(percent_poisoning/100*benign_x_train.shape[0])]],axis=0)
            np.random.shuffle(x_train)
            training_features = x_train
            test_features = x_test
            n_autoencoder=5
            n_autoencoder1=x_train.shape[1]%n_autoencoder
            n_features2=math.floor(x_train.shape[1]/n_autoencoder)
            n_features1=n_features2+1
            n_autoencoder2=n_autoencoder-n_autoencoder1
            Ensemble1=np.empty(n_autoencoder1,dtype=object) #Ensemble layer1
            Ensemble2=np.empty(n_autoencoder2,dtype=object) #Ensemble layer2
            for i in range(n_autoencoder1):
                Ensemble1[i]= Sequential()
                Ensemble1[i].add(Dense(units=n_features1,activation='relu',input_shape=(n_features1,)))
                Ensemble1[i].add(Dense(units=math.floor(0.75*n_features1),activation='relu'))
                Ensemble1[i].add(Dense(units=n_features1,activation='sigmoid'))
                Ensemble1[i].compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
            for i in range(n_autoencoder2):
                Ensemble2[i]= Sequential()
                Ensemble2[i].add(Dense(units=n_features2,activation='relu',input_shape=(n_features2,)))
                Ensemble2[i].add(Dense(units=math.floor(0.75*n_features2),activation='relu'))
                Ensemble2[i].add(Dense(units=n_features2,activation='sigmoid'))
                Ensemble2[i].compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
            #Creo un unico autoencoder di output che ha una dimensione di ingresso pari al n totale di autoencoder nell'ensemble layer
            Output= Sequential()
            Output.add(Dense(units=n_autoencoder,activation='relu',input_shape=(n_autoencoder,)))
            Output.add(Dense(units=math.floor(0.75*n_autoencoder),activation='relu'))
            Output.add(Dense(units=n_autoencoder,activation='sigmoid'))
            Output.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
            scaler1=MinMaxScaler(feature_range=(0,1))
            training_features=scaler1.fit_transform(training_features)
            #faccio la fit degli autoencoder 
            for i in range(n_autoencoder1):
                Ensemble1[i].fit(training_features[:,i*n_features1:(i+1)*n_features1],training_features[:,i*n_features1:(i+1)*n_features1], epochs=5, batch_size=32)
            for i in range(n_autoencoder2):
                Ensemble2[i].fit(training_features[:,n_autoencoder1*n_features1+i*n_features2:n_autoencoder1*n_features1+(i+1)*n_features2],training_features[:,n_autoencoder1*n_features1+i*n_features2:n_autoencoder1*n_features1+(i+1)*n_features2], epochs=5, batch_size=32)
            score=np.zeros((training_features.shape[0],n_autoencoder))
            #calcolo gli RMSE centrali del livello ensemble da dare all'Output layer
            for j in range(n_autoencoder1):
                pred=Ensemble1[j].predict(training_features[:,j*n_features1:(j+1)*n_features1])
                for i in range(training_features.shape[0]):
                    score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],training_features[i,j*n_features1:(j+1)*n_features1]))
            for j in range(n_autoencoder2):
                pred=Ensemble2[j].predict(training_features[:,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2])
                for i in range(training_features.shape[0]):
                    score[i,j+n_autoencoder1]= np.sqrt(metrics.mean_squared_error(pred[i],training_features[i,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2]))
            #faccio fare la fit all'Output layer tramite gli RMSE calcolati precedentemente
            scaler2=MinMaxScaler(feature_range=(0,1))
            score=scaler2.fit_transform(score)
            Output.fit(score,score,epochs=5,batch_size=32)
            RMSE=np.zeros(score.shape[0])
            pred=Output.predict(score)
            for i in range(score.shape[0]):
                RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],score[i]))
            # FASE DI TESTING
            test_features=scaler1.transform(test_features)
            test_score=np.zeros((test_features.shape[0],n_autoencoder))
            #calcolo RMSE in fase di Testing
            for j in range(n_autoencoder1):
                pred=Ensemble1[j].predict(test_features[:,j*n_features1:(j+1)*n_features1])
                for i in range(test_features.shape[0]):
                    test_score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],test_features[i,j*n_features1:(j+1)*n_features1]))
            for j in range(n_autoencoder2):
                pred=Ensemble2[j].predict(test_features[:,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2])
                for i in range(test_features.shape[0]):
                    test_score[i,j+n_autoencoder1]= np.sqrt(metrics.mean_squared_error(pred[i],test_features[i,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2]))
            #calcolo RMSE FINALE
            test_score=scaler2.transform(test_score)
            RMSE=np.zeros(test_score.shape[0])
            pred=Output.predict(test_score)
            for i in range(test_score.shape[0]):
                RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],test_score[i]))
            fpr, tpr,thresholds= metrics.roc_curve(y_test,RMSE)
            indices=np.where(fpr>=0.01)
            index=np.min(indices)
            soglia=thresholds[index]
            labels=np.zeros(RMSE.shape[0])
            for i in range(RMSE.shape[0]):
                if RMSE[i] < soglia:
                    labels[i] = 0
                else:
                    labels[i] = 1
            for_plot = dict()
            for_plot = {'y_true': y_test, 'y_pred': labels, 'RMSE': RMSE}
            plot_df = pd.DataFrame.from_dict(for_plot)
            plot_df.to_csv("Statistiche/"+str(percent_poisoning)+"%_poisoning/"+str(iteration+1)+"Fold.csv")
        iteration+=1
        
def Kitsune_Summary(path,path1,num_epochs):
    feature_list = [5,10,15,20,25,30,35,40,45,50,55,60,65,71]
    f1_list = list()
    big_list = list()
    stats_dict = dict()
    for feat in feature_list:
        for i in range(1,11):
            dataset = pd.read_csv(str(path)+"/{}/{}Fold.csv".format(feat,i))
            f1_score = metrics.f1_score(dataset[:]['y_true'], dataset[:]['y_pred'],average='macro')
            f1_list.append(f1_score)
        mean = statistics.mean(f1_list)
        std = statistics.stdev(f1_list)

 


        stats_dict['{}_features'.format(feat)] = [mean,std]
    stats_df =pd.DataFrame.from_dict(stats_dict,orient='index')
    stats_df.columns = ['mean','std']
    stats_df.to_csv(str(path1)+"/"+str(num_epochs)+"/summary.csv")
