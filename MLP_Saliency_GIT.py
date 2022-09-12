# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:36:40 2022

@author: ASUS
"""


import ast
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Activation
from keras.models import Sequential


#from tensorflow.keras.layers.normalization import BatchNormalization



tf.compat.v1.enable_eager_execution()

#==================definitions==========
def acc(test,pred,maxdiff): #returns accuracy to a given difference

    count=0
    for i in range (len(pred)):
        if abs(pred[i]-test[i])<=maxdiff:
            count=count+1
    return (count/len(pred))

def flatten(lst): #flatten a list of matrices to a list of vectors
    ret = []
    for mat in lst:
        vect = []
        for row in mat:
            for it in row:
                vect.append(it)
        ret.append(vect)
    return ret
#concatenates a flat dataset to the right of the original
def join(fts1, fts2):  
    res = []
    
    for i in range(len(fts1)):
        row = np.hstack((np.array(fts1[i]), np.array(fts2[i])))
        res.append(row)
    return np.array(res)

#normalises the given dataset to mean 0 standard deviation 1
def normalize(features, add_target=True):  
  
  sigma = features.std(axis=0)
  
  mu = features.mean(axis=0)
  normed_features = (features - mu) / sigma
  
  return normed_features
#=====================Import data=============

#unplickle the data dictionaries, names of variables are there
with open('X_dict.pkl', 'rb') as f:
    X_dict = pickle.load(f)    

with open('Y_dict.pkl', 'rb') as f:
    Y_dict = pickle.load(f)


action1_inds, action2_inds, action3_inds, action4_inds, action4_sample, balanced_inds,reebs, canreebs,finalFeatures, dualFeatures,  normals,  pointFeatures, scaledFeatures, polys,scaledPoints, polyvols, topo  = X_dict.values()

vols, reebDeg, balanced_degs = Y_dict.values()

#for the saliency displays for multi-component features find length of datasets 
lengths = [32, 98, 545, 3644, 675, 1350, 3, 3, 5, 3, 42, 2, 15, 42, 5, 1, 2]

#dictionary of lengths 
X_lens_dict = {list(X_dict.keys())[i]: lengths[i] for i in range(len(list(X_dict.keys())))}
    


#===================================NETWORK===================================

#regressor neural network, verbose activates display of results in console
def NNreg(X, Y, title, iters=1, verbose=1):
    
    
    #find the range of Y for plotting
     rang = max(Y)-min(Y)
    
     print('Value range', rang)
    
    
     #training-test split
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5,
         shuffle=True)
    
     #the results will be averaged before metrics
     Y_pred = []
     Y_trained = []
    
     for i in range(iters): 
         #NN setup
         network = Sequential() #make an MLP
         network.add(Dense(60,activation=None,input_dim=len(X[0]))) #set 50 nodes with input 5 len(X[0])
         network.add(Activation('sigmoid')) #sigmoid (logistic) activation
         network.add(Dense(5,activation=None)) #10 node output with id activation
         network.add(Activation('sigmoid')) #sigmoid (logistic) activation
         network.add(Dense(1)) #single output 
         network.compile(loss='mae', optimizer = 'adam', metrics=['accuracy'])
        
         #train
         network.fit(X_train, Y_train, batch_size=30,epochs=1000,verbose=0, validation_data=(X_test,Y_test))
        
        
         #predict 
         Y_pred.append(network.predict(X_test,verbose = 0)[:,0])
         Y_trained.append(network.predict(X_train,verbose = 0)[:,0])
        
     
     #average over the chosen number of iterations
     Y_pred_avg = np.average(np.array(Y_pred), axis=0)
     Y_pred_sd  = np.std(np.array(Y_pred), axis=0)
     Y_trained_avg = np.average(np.array(Y_trained), axis=0)
     Y_trained_sd  = np.std(np.array(Y_trained), axis=0)
     
     
     #find the performance metrics
     metrics = []  
     
     metrics.append(mean_absolute_error(Y_train, Y_trained_avg))
     metrics.append(mean_absolute_error(Y_train, Y_trained_avg)/rang)
     #metrics.append(acc(Y_train, Y_trained,0.5*rang)) #for classification problems
     metrics.append(acc(Y_train, Y_trained_avg,rang*0.05))
     metrics.append(acc(Y_train, Y_trained_avg,rang*0.025))
     metrics.append(acc(Y_train, Y_trained_avg,rang*0.01))
        
     metrics.append(mean_absolute_error(Y_pred_avg, Y_test))
     metrics.append(mean_absolute_error(Y_pred_avg, Y_test)/rang)
     #metrics.append(acc(Y_test,Y_pred,0.5*rang)) #for classification problems
     metrics.append(acc(Y_test,Y_pred_avg,rang*0.05))
     metrics.append(acc(Y_test,Y_pred_avg,rang*0.025))
     metrics.append(acc(Y_test,Y_pred_avg,rang*0.01))
        
     #display the metrics
     if verbose == 1:
            print(title)
            print('TRAINING SET')
            print('MAE: ', metrics[0])
            print('MAE-scaled: ', metrics[1])
            print('Accuracy 5%', metrics[2])
            print('Accuracy 2.5%', metrics[3])
            print('Accuracy 1%', metrics[4])
            print('TEST SET')
            print('MAE: ', metrics[5])
            print('MAE-scaled: ', metrics[6])
            print('Accuracy 5%', metrics[7])
            print('Accuracy 2.5%', metrics[8])
            print('Accuracy 1%', metrics[9])
       
     return X_train, X_test, Y_test, Y_train, Y_pred_avg, Y_trained_avg, np.array(metrics), network


#classifier neural network, verbose activates display of results in console
def NNclass(X, Y, title, iters=1, verbose=1):
    
    
     rang = max(Y)-min(Y)
    
    
     #training-test split
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5,
         shuffle=True)
    
     #the results will be averaged before metrics
     Y_pred = []
     Y_trained = []
    
     for i in range(iters): 
         #NN setup
         network = Sequential() #make an MLP
         network.add(Dense(60,activation=None,input_dim=len(X[0]))) #set 50 nodes with input 5 len(X[0])
         network.add(Activation('sigmoid')) #sigmoid (logistic) activation
         network.add(Dense(5,activation=None)) #10 node output with id activation
         network.add(Activation('sigmoid')) #sigmoid (logistic) activation
         network.add(Dense(1)) #single output 
         network.compile(loss='mae', optimizer = 'adam', metrics=['accuracy'])
        
         #train
         network.fit(X_train, Y_train, batch_size=30,epochs=1000,verbose=0, validation_data=(X_test,Y_test))
        
        
         #predict 
         Y_pred.append(network.predict(X_test,verbose = 0)[:,0])
         Y_trained.append(network.predict(X_train,verbose = 0)[:,0])
        
     
     #average over the chosen number of iterations
     Y_pred_avg = np.average(np.array(Y_pred), axis=0)
     Y_pred_sd  = np.std(np.array(Y_pred), axis=0)
     Y_trained_avg = np.average(np.array(Y_trained), axis=0)
     Y_trained_sd  = np.std(np.array(Y_trained), axis=0)
    
     #find the performance metrics
     metrics = []
          
     metrics.append(mean_absolute_error(Y_train, Y_trained_avg))
     metrics.append(acc(Y_train, Y_trained_avg,0.5)) 
     metrics.append(acc(Y_train, Y_trained_avg,rang*0.01))
        
     metrics.append(mean_absolute_error(Y_pred_avg, Y_test))
     metrics.append(acc(Y_test,Y_pred_avg,0.5)) #for classification problems
     metrics.append(acc(Y_test,Y_pred_avg,rang*0.01))
        
     #display the metrics
     if verbose == 1:
            print(title)
            print('TRAINING SET')
            print('MAE: ', metrics[0])
            print('Classification accuracy', metrics[1])
            print('Accuracy 1%', metrics[2])
            print('TEST SET')
            print('MAE: ', metrics[3])
            print('Classification accuracy', metrics[4])
            print('Accuracy 1%', metrics[5])
       
     return X_train, X_test, Y_test, Y_train, Y_pred_avg, Y_trained_avg, np.array(metrics), network



#======================Visualisers==========
#prediction vs data plotter for the regressor 
def visualiserReg(Y_test, Y_pred, rang, title):
    plt.scatter(Y_pred, Y_test) #abs to get rid of weird -0.
    plt.plot(np.arange(4319)/4319*rang, np.arange(4319)/4319*rang, color='black')
    
    
   # plt.title(title)
    
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    
    plt.show()
    plt.close()
#prediction vs data plotter for the regressor for the classifier
def visualiserClass(Y_test, Y_pred, rang, title):
    plt.scatter(np.rint(Y_pred), Y_test) #abs to get rid of weird -0.
    plt.plot([0, 5], [0, 5], color='black')
    
    
    #plt.title(title, wrap=True)
    
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    
    plt.show()
    plt.close()



#===================================SALIENCE ANALYSIS===============================
#performs gradient saliency analysis, produces the corresponding bar graph
#the procedure is performed on the TEST set 
def salience_analysis(X_test, net, title, labels):
    #find the feature variables involved
    X_var = tf.Variable(X_test, name = 'x')
    #perform the backprop with respect to the features
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(X_var)
        Y_grads = net(X_var)
        loss = tf.reduce_mean(Y_grads**2)
    
    #extract gradients    
    grad = tape.gradient(Y_grads, X_var)
    
    avggrads = np.average(np.abs(grad.numpy()), axis = 0)
    
    
    #compress multi-component features for figures 
    
    red_grads = []
    
    #sum the elements in a weighed mean to avoid the fact how long features tend to overweigh
    for label in labels:
        subarr = avggrads[:X_lens_dict[label]]
        red_grads.append(np.sum(subarr)/X_lens_dict[label])
        avggrads = np.delete(avggrads, np.s_[:X_lens_dict[label]])
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    
    
    #ax.set_title(title, loc='center', wrap=True)
    
    
    
    
    #normalise bars in the plot
    norm = np.sum(np.array(red_grads))
    
    #display the bar graph with labels
    
    ax.set_yticks(np.arange(len(red_grads)), labels=labels)
    
    for i in range(len(labels)):
    
        ax.barh(i, red_grads[i]/norm, color='blue')
    
    
    #ax.legend(labels, prop={'size': 8})
    
    plt.show()
    plt.close()





#=================Experiments============
#demo of the experiment finding the critical volume
def volumes_exp(iters=1, verbose=1):
    
    
    labels = ['point features', 'polytope volumes','topological numbers', 'toric diagrams']
    title = 'Learning volumes from '
    for label in labels:
        title+= label+', '
    title = title[:-2]
    
    
    X = join(normalize(finalFeatures), pad_sequences(flatten(polys), padding='post',value=0))

    Y = vols

    rang = max(Y)-min(Y)

    X_train, X_test, Y_test, Y_train, Y_pred, Y_trained, metrics, net = NNreg(X, Y, title, iters=iters, verbose = verbose)

    

    visualiserReg(Y_test, Y_pred, rang, title)
    salience_analysis(X_test, net, title, labels)
    
    
    labels = ['point features', 'polytope volumes','topological numbers']
    
    title = 'Learning volumes from '
    for label in labels:
        title+= label+', '
    title = title[:-2]
    
    X = normalize(finalFeatures)

    X_train, X_test, Y_test, Y_train, Y_pred, Y_trained, metrics, net = NNreg(X, Y, title, iters=iters, verbose = verbose)

    

    visualiserReg(Y_test, Y_pred, rang, title)
    salience_analysis(X_test, net, title, labels)
    
    
    labels = ['point features', 'polytope volumes','topological numbers', 'scaled point features']
    title = 'Learning volumes from '
    for label in labels:
        title+= label+', '
    title = title[:-2]
    
    X = normalize(join(finalFeatures, scaledFeatures))


    X_train, X_test, Y_test, Y_train, Y_pred, Y_trained, metrics, net = NNreg(X, Y, title, iters=iters, verbose = verbose)

    

    visualiserReg(Y_test, Y_pred, rang, title)
    salience_analysis(X_test, net, title, labels)
    
    
    
    labels = ['point features', 'polytope volumes','topological numbers', 'scaled point features', 'Reeb vectors']
    title = 'Learning volumes from '
    for label in labels:
        title+= label+', '
    title = title[:-2]
    
    X = normalize(join(join(finalFeatures, scaledFeatures), reebs))#, pad_sequences(flatten(polys), padding='post',value=0))


    X_train, X_test, Y_test, Y_train, Y_pred, Y_trained, metrics, net = NNreg(X, Y, title, iters=iters, verbose = verbose)

    

    visualiserReg(Y_test, Y_pred, rang, title)
    salience_analysis(X_test, net, title, labels)
    
    
#demo of experiment finding the degree action
def degrees_exp(iters=1, verbose=1):
    
    labels = ['point features', 'polytope volumes','topological numbers', 'canonical Reeb vectors', 'toric diagrams']
    title = 'Learning balanced set for action degree from '
    for label in labels:
        title+= label+', '
    title = title[:-2]
    
    X = join(normalize(join(finalFeatures, canreebs)), pad_sequences(flatten(polys), padding='post',value=0))[balanced_inds]

    Y = balanced_degs

    rang= max(Y)-min(Y)
    

    X_train, X_test, Y_test, Y_train, Y_pred, Y_trained, metrics, net = NNclass(X, Y, title, iters=iters, verbose = verbose)

    

    visualiserClass(Y_test, Y_pred, rang, title)
    salience_analysis(X_train, net, title, labels)
    
    labels = ['canonical Reeb vectors', 'toric diagrams']
    title = 'Learning balanced set for action degree from '
    for label in labels:
        title+= label+', '
    title = title[:-2]
    
    X = join(normalize(canreebs), pad_sequences(flatten(polys), padding='post',value=0))[balanced_inds]

    Y = balanced_degs

    rang = max(Y)-min(Y)

    X_train, X_test, Y_test, Y_train, Y_pred, Y_trained, metrics, net = NNclass(X, Y, title, iters=iters, verbose = verbose)

    

    visualiserClass(Y_test, Y_pred, rang, title)
    salience_analysis(X_train, net, title, labels)
    
    labels = ['point features', 'polytope volumes','topological numbers', 'toric diagrams']
    title = 'Learning balanced set for action degree from '
    for label in labels:
        title+= label+', '
    title = title[:-2]
    
    X = join(normalize(finalFeatures), pad_sequences(flatten(polys), padding='post',value=0))[balanced_inds]

    Y = balanced_degs

    rang = max(Y)-min(Y)

    X_train, X_test, Y_test, Y_train, Y_pred, Y_trained, metrics, net = NNclass(X, Y, title, iters=iters, verbose = verbose)

    

    visualiserClass(Y_test, Y_pred, rang, title)
    salience_analysis(X_train, net, title, labels)
    
    labels = ['canonical Reeb vectors', 'facet normals']
    title = 'Learning balanced set for action degree from '
    for label in labels:
        title+= label+', '
    title = title[:-2]
    
    X = join(normalize(canreebs), pad_sequences(flatten(normals), padding='post',value=0))[balanced_inds]

    Y = balanced_degs

    rang = max(Y)-min(Y)

    X_train, X_test, Y_test, Y_train, Y_pred, Y_trained, metrics, net = NNclass(X, Y, title, iters=iters, verbose = verbose)

    

    visualiserClass(Y_test, Y_pred, rang, title)
    salience_analysis(X_train, net, title, labels)
    
      
#pickle the data
def pickle():
    X_dict = {"deg 1 action indices": action1_inds, 
              "deg 2 action indices": action2_inds, 
              "deg 3 action indices": action3_inds, 
              "deg 4 action indices": action4_inds, 
              "deg 4 action indices sample": action4_sample,
              "balanced index set for degrees": balanced_inds,
              "Reeb vectors": reebs,
              "canonical Reeb vectors": canreebs,
              "combined features": finalFeatures, 
              "dual features": dualFeatures, 
              "facet normals": normals, 
              "point features": pointFeatures, 
              "scaled point features": scaledFeatures, 
              "toric diagrams": polys,
              "scaled toric diagrams": scaledPoints, 
              "polytope volumes": polyvols, 
              "topological numbers": topo} 

    Y_dict = {"critical volumes": vols, "Reeb action degrees": reebDeg, "balanced set for degrees": balanced_degs}

    with open('X_dict.pkl', 'wb') as f:
        pickle.dump(X_dict, f)
        
    with open('Y_dict.pkl', 'wb') as f:
        pickle.dump(Y_dict, f)








#=======MAIN======

#uncomment the demo you'd like to see
#iters refers to the number of training cycles on the same sample to average on 

volumes_exp(iters = 1, verbose=1)

degrees_exp(iters = 1, verbose=1)





print('Done')
