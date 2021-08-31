import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import timeit
import statistics
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import sklearn
import imblearn
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def dataset_split(n_samples, y=None, k=10, random_state=0):
    """
    :param n_samples: number of samples compose the dataset
    :param y: labels, for stratified splitting
    :param k: number of splits
    :param random_state: random state for the splitting function
    :return:
    """
    splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
    indexes = [(train_index, test_index) for train_index, test_index in splitter.split(np.zeros((n_samples, 1)))]
    return indexes

def create_train_dataset():
    path = "D:/IDALAB/Statistiche/"
    benign_df = pd.DataFrame()
    benign_label = pd.DataFrame()
    malign_label = pd.DataFrame()
    malign_df = pd.DataFrame()
    for string,num in [("Active_Wiretap",0),("ARP_MitM",1),("Fuzzing",2),("OS_Scan",3),("SSDP_Flood",4),("SSL_Renegotiation",5),("SYN_DoS",6),("Video_Injection",7),("Mirai_Botnet/IP",8)]:
        temp_benign = pd.read_csv(str(path)+str(string)+'/Statistiche_Biflussi_benign_biflows.csv')
        temp_malign = pd.read_csv(str(path)+str(string)+'/Statistiche_Biflussi_malign_biflows.csv')
        if temp_benign.empty:
            print(str(string)+"benign vuoto")
        else:
            temp_benign = temp_benign.drop(['Unnamed: 0','Unnamed: 1','Unnamed: 2'],axis=1)
        benign_labels = np.zeros((len(temp_benign),2))
        for i in range(len(temp_benign)):
            benign_labels[i][1] = num
        if len(benign_labels)==0:
            None
        else:
            benign_labels = pd.DataFrame(benign_labels,columns=['Malign','Class'])
            benign_df = pd.concat([benign_df,temp_benign])
            benign_label = pd.concat([benign_label,benign_labels])
            
        temp_malign = temp_malign.drop(['Unnamed: 0','Unnamed: 1','Unnamed: 2'],axis=1)
        malign_labels = np.ones((len(temp_malign),2))
        for i in range(len(temp_malign)):
            malign_labels[i][1] = num
        malign_labels = pd.DataFrame(malign_labels,columns=['Malign','Class'])
        #total = pd.concat([temp_benign,temp_malign],axis=0)
        #total_labels = np.concatenate((benign_labels,malign_labels))

        malign_df = pd.concat([malign_df,temp_malign])
        malign_label = pd.concat([malign_label,malign_labels])

    benign_df.to_csv(str(path)+"Biflussi_Totali/benign_biflows_data.csv",index =False)
    benign_label.to_csv(str(path)+"Biflussi_Totali/benign_biflows_labels.csv",index =False)
    malign_df.to_csv(str(path)+"Biflussi_Totali/malign_biflows_data.csv",index =False)
    malign_label.to_csv(str(path)+"Biflussi_Totali/malign_biflows_labels.csv",index =False)

    
def data_compact():
    path = "D:/IDALAB/Statistiche/"
    x = pd.read_csv("D:/IDALAB/Statistiche/Biflussi_Totali/benign_biflows_data.csv")
    x_label = pd.read_csv("D:/IDALAB/Statistiche/Biflussi_Totali/benign_biflows_labels.csv")
    x_label = x_label.drop(columns="Class")
    x1 = pd.read_csv("D:/IDALAB/Statistiche/Biflussi_Totali/malign_biflows_data.csv")
    x1_label = pd.read_csv("D:/IDALAB/Statistiche/Biflussi_Totali/malign_biflows_labels.csv")
    x1_label = x1_label.drop(columns="Class")

    complete_x = pd.concat([x,x_label],axis=1)
    complete_x1 = pd.concat([x1,x1_label],axis=1)
    train_data = pd.concat([complete_x,complete_x1],axis=0)
    train_data.to_csv(str(path)+"Biflussi_Totali/training_set.csv",index=False)

def create_autoencoder(x_train):
    #Tentativo usando autoencoder

    #Numero di input al modello
    input_dim = x_train.shape[1]

    #Dimensione del layer compresso
    encoding_dim = int(round(input_dim*0.75))

    #Dimensione dell'input
    input_data = keras.Input(shape=(input_dim,))

    #Costruzione layer di codifica
    encoded = layers.Dense(encoding_dim, activation = 'relu')(input_data)

    #Costruzione layer di decodifica
    decoded = layers.Dense(input_dim, activation = 'sigmoid' )(encoded)

    #Costruzione Autoencoder
    autoencoder = keras.Model(input_data, decoded)

    #Costruzione modello di codifica (a parte)
    encoder = keras.Model(input_data,encoded)

    #Dimensione input codificato
    encoded_input = keras.Input(shape=(encoding_dim,))

    #Riprendo l'ultimo layer del modello autoencoder
    decoder_layer = autoencoder.layers[-1]

    #Costruzione modello di decodifica a parte
    decoder = keras.Model(encoded_input,decoder_layer(encoded_input))

    opt = keras.optimizers.Adam(learning_rate=0.001)

    autoencoder.compile(optimizer=opt,loss='mse',metrics=['accuracy'])

    return autoencoder

def create_deep_autoencoder(x_train):
    #Tentativo con Deep Autoencoder (8 layer fully connected)
    input_dim = x_train.shape[1]
    input_data = keras.Input(shape=(input_dim,))
    enc_dim1 = int(round(input_dim*0.75))
    enc_dim2 = int(round(input_dim*0.5))
    enc_dim3 = int(round(input_dim*0.33))
    enc_dim4 = int(round(input_dim*0.25))

    encoded = layers.Dense(enc_dim1, activation = 'relu')(input_data)
    encoded = layers.Dense(enc_dim2, activation = 'relu')(encoded)
    encoded = layers.Dense(enc_dim3, activation = 'relu')(encoded)
    encoded = layers.Dense(enc_dim4, activation = 'relu')(encoded)
    decoded = layers.Dense(enc_dim3, activation = 'relu')(encoded)
    decoded = layers.Dense(enc_dim2, activation = 'relu')(decoded)
    decoded = layers.Dense(enc_dim1, activation = 'relu')(decoded)
    decoded = layers.Dense(input_dim, activation = 'sigmoid')(decoded)

    deep_autoencoder = keras.Model(input_data,decoded)
    opt = keras.optimizers.Adam(learning_rate=0.001)

    deep_autoencoder.compile(optimizer=opt,loss='mse',metrics=['accuracy'])
    return deep_autoencoder

    
def train_model(my_model,data,data_labels):
    x_train,x_test,_,_ = train_test_split(data,data_labels)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=5)]
    history = my_model.fit(x_train_s,x_train_s,epochs=100,batch_size=256,shuffle=True,validation_data=(x_test_s,x_test_s),callbacks=my_callbacks)

    predictions = deep_autoencoder.predict(x_test_s)
    x_true = scaler.inverse_transform(x_test_s)
    x_hat = scaler.inverse_transform(predictions)
    test_score = np.sqrt(mean_squared_error(x_test,predictions))
    print('Test Score: %.2f RMSE' % (test_score))

    true_score = np.sqrt(mean_squared_error(x_true, x_hat))
    print('True Score: %.2f RMSE' % (true_score))
    return my_model,history


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
        #print(str(len(rfe_feature)), 'selected features')
        #print(rfe_feature)
        
    
    features = dict()
    for i in range(0,len(selected_features)):
        features[num_feats[i]] ={
            'selected_features':int
        }
        features[num_feats[i]] = selected_features[i]
        
    return features


def test_model_n_feats(x_train,x_test,y_test,selected_features):
    #Training e Test del modello con tutte le n features più importanti in selected features
    for n_features,features in selected_features.items():
        #Costruisco Training e Test Set con le sole n features più importanti e li normalizzo
        training_set = x_train[:,features]
        test_set = x_test[:,features]
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler = scaler.fit(training_set)
        train = scaler.transform(training_set)
        test = scaler.transform(test_set)
        
        my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=5)]
        
        #Creo, Alleno e misuro il tempo di training relativo al modello
        
        my_model = create_deep_autoencoder(train)
        
        start = timeit.timeit()

        history = my_model.fit(train,train,epochs=10,batch_size=256,shuffle=True,validation_data=(test,test),callbacks=my_callbacks)
        
        end = timeit.timeit()
        elapsed= end-start
        
        #Salvo in formato csv le statistiche di training del modello
        history_df =pd.DataFrame(history.history)
        hist_csv_file ='D:/IDALAB/Training/Deep_Autoencoder/fix/history_fold{}.csv'.format(n_features)
        with open(hist_csv_file, mode='w') as f:
            history_df.to_csv(f)
            
        
        #Predict del modello sul train set
        train_pred = my_model.predict(train)

        #Calcolo RMSE sulle features predette sui singoli biflussi
        train_rmse = list()
        for i in range(0,len(train)):
            train_rmse.append(np.sqrt(mean_squared_error(train[i],train_pred[i])))

        #Calcolo media, deviazione standard e stabire soglia per l'anomaly detection    
        mean = statistics.mean(train_rmse)
        dev = statistics.stdev(train_rmse)
        #treshold = mean+4*dev


        #Predict del modello sul test set
        test_pred = my_model.predict(test)
        test_rmse = list()
        for i in range(0,len(test)):
            test_rmse.append(np.sqrt(mean_squared_error(test[i],test_pred[i])))


        #Calcolo Soglia con tpr fpr e roc curve
    
        fpr,tpr,thresholds = metrics.roc_curve(y_test,test_rmse)
        indices = np.where(fpr>=0.01)
        index = np.min(indices)
        soglia = thresholds[index]
        label_pred = list()
        for i in range(0,len(test)):
            if test_rmse[i] >= soglia:
                label_pred.append(1)
            else:
                label_pred.append(0)
                
        f1_score = metrics.f1_score(y_test,label_pred,average='macro')
        
        #Salvo in csv varie informazioni
        stats = dict()
        stats = {'mean' : mean, 'dev' : dev, 'threshold' : soglia, 'f1_score':f1_score,'time':elapsed}
        stats_df =pd.DataFrame.from_dict(stats,orient='index')
        stats_df.to_csv("D:/IDALAB/Training/Deep_Autoencoder/fix/stats_{}.csv".format(n_features))
        
        #Salvo in csv le info che servivanno a plottare roc_curve e f1_score
        for_plot = dict()
        for_plot = {'y_true': y_test, 'y_pred':label_pred,'RMSE': test_rmse}
        plot_df = pd.DataFrame.from_dict(for_plot)
        plot_df.to_csv("D:/IDALAB/Training/Deep_Autoencoder/fix/plot_{}.csv".format(n_features))
        
        #Plotto e salvo la confusion matrix
        mat = confusion_matrix(y_test, label_pred, normalize = 'true') # evaluate
        plt.figure(figsize=(12, 8), dpi=70)
        sns.heatmap(mat.T*100, square=True, annot=True, vmin=0.0, vmax=100.0, fmt='.1f', cbar=True) # plot via seaborn
        sns.set(font_scale=1.4) # for label size
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label');
        plt.title('Confusion Matrix of Autoencoder')
        plt.savefig("D:/IDALAB/Training/Deep_Autoencoder/fix/confusion_matrix_{}".format(n_features))
    

def fold_test_features():
    train_data_n = pd.read_csv("D:/IDALAB/Statistiche/Biflussi_Totali/training_set.csv")
    train_data_n = np.array(train_data_n)
    #features=selected_features[71]
    train_data_labels=train_data_n[:,71]
    train_data_n=train_data_n[:,:71]
    iteration=0
    for percent_poisoning in (0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10):
        for train_index, test_index in Skf.split(train_data_n,train_data_labels):
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
            scaler = MinMaxScaler(feature_range=(0,1))
            scaler = scaler.fit(x_train)
            training_features = scaler.transform(x_train)
            test_features = scaler.transform(x_test)

            my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=5)]

            #Creo, Alleno e misuro il tempo di training relativo al modello

            my_model = create_deep_autoencoder(training_features)

            start = timeit.timeit()

            history = my_model.fit(training_features,training_features,epochs=5,batch_size=32,shuffle=True,validation_data=(test_features,test_features),callbacks=my_callbacks)

            end = timeit.timeit()
            elapsed= end-start

            #Salvo in formato csv le statistiche di training del modello
            history_df =pd.DataFrame(history.history)
            hist_csv_file ='D:/IDALAB/New_Training/Deep_Autoencoder/Poisoning/{}%/Fold{}/history.csv'.format(percent_poisoning,iteration+1)
            with open(hist_csv_file, mode='w') as f:
                history_df.to_csv(f)


            #Predict del modello sul train set
            train_pred = my_model.predict(training_features)

            #Calcolo RMSE sulle features predette sui singoli biflussi
            train_rmse = list()
            for i in range(0,len(training_features)):
                train_rmse.append(np.sqrt(mean_squared_error(training_features[i],train_pred[i])))

            #Calcolo media, deviazione standard e stabire soglia per l'anomaly detection    
            mean = statistics.mean(train_rmse)
            dev = statistics.stdev(train_rmse)
            #treshold = mean+4*dev


            #Predict del modello sul test set
            test_pred = my_model.predict(test_features)
            test_rmse = list()
            for i in range(0,len(test_features)):
                test_rmse.append(np.sqrt(mean_squared_error(test_features[i],test_pred[i])))


            #Calcolo Soglia con tpr fpr e roc curve

            fpr,tpr,thresholds = metrics.roc_curve(y_test,test_rmse)
            indices = np.where(fpr>=0.01)
            index = np.min(indices)
            soglia = thresholds[index]
            label_pred = list()
            for i in range(0,len(test_features)):
                if test_rmse[i] >= soglia:
                    label_pred.append(1)
                else:
                    label_pred.append(0)

            f1_score = metrics.f1_score(y_test,label_pred,average='macro')

            #Salvo in csv varie informazioni
            stats = dict()
            stats = {'mean' : mean, 'dev' : dev, 'threshold' : soglia, 'f1_score':f1_score,'time':elapsed}
            stats_df =pd.DataFrame.from_dict(stats,orient='index')
            stats_df.to_csv('D:/IDALAB/New_Training/Deep_Autoencoder/Poisoning/{}%/Fold{}/stats.csv'.format(percent_poisoning,iteration+1))




            for_plot = dict()
            for_plot = {'y_true': y_test, 'y_pred': label_pred, 'RMSE': test_rmse}
            plot_df = pd.DataFrame.from_dict(for_plot)
            plot_df.to_csv("D:/IDALAB/New_Training/Deep_Autoencoder/Poisoning/{}%/Fold{}/plot.csv".format(percent_poisoning,iteration+1))
            iteration+=1
        iteration = 0