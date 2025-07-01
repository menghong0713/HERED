# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import Titanic_featureSelection_GA as tfg
from SVC import skl
import os

data = pd.read_excel('train_dataset.xlsx')

X = data.iloc[:,2:].apply(lambda x: round(x, 4)) 

x_train11 = X.values
y_train = data.iloc[:,1].values

columns=data.columns[2 :]
columns_new=[]
columns_new.append('generation')
columns_new.append('accuracy')
columns_new.append('descripter number')

columns_new2=['generation','combination number','Max accuracy','Average accuracy','Min accuracy','descriptor number of Max accuracy']
for i in range(len(columns)):
    columns_new.append(columns[i])
    columns_new2.append(columns[i])
    
#Initialize params for GA
generation = 0
population = 100  
num_features = data.shape[1]-2
feature_combines = []
fitness_ls = []

MAXGEN=50
trace = np.zeros((MAXGEN, 2))

offspring_outs=[]
offspring_outs_Gen=[]
#Train the model by GA(feature selection)
while generation<MAXGEN:
    print('\n')
    print('Generation: ', generation,end='')
    
    #Update train and test by each feature combination
    feature_combines = tfg.feature_selection(population, num_features, feature_combines, fitness_ls)
    fitness_ls = []
    for feature_combine in feature_combines:
        offspring_out=[]
        x_train=pd.DataFrame(x_train11)
        x_train = x_train[x_train.columns[feature_combine]]
        
        try:
            sls13=skl()
            score=sls13.training(x_train.copy(),y_train.copy())
            
            offspring_out.append(generation)
            offspring_out.append(score)
            offspring_out.append(feature_combine.count(1))
            for i in feature_combine:
                if i:
                    offspring_out.append('True')
                else:
                    offspring_out.append('')
            offspring_outs.append(offspring_out)
            
            fitness_ls.append(score)
        except Exception as e:
            print(e)
            score = 0
            fitness_ls.append(score)
        
    trace[generation, 0] = max(fitness_ls)
    trace[generation, 1] =(sum(fitness_ls)/len(fitness_ls))
    print('')
    print('combination number: '+str(len(fitness_ls)))
    print('Max score: ', max(fitness_ls))
    print('Average score: ', sum(fitness_ls) / len(fitness_ls))
    print('Min score: ', min(fitness_ls))
    
    
    sheet2_out=[]
    sheet2_out.append(generation)
    sheet2_out.append(len(fitness_ls))
    sheet2_out.append(max(fitness_ls))
    sheet2_out.append(sum(fitness_ls) / len(fitness_ls))
    sheet2_out.append(min(fitness_ls))
    sheet2_out.append(feature_combines[fitness_ls.index(max(fitness_ls))].count(1))
    for i in feature_combines[fitness_ls.index(max(fitness_ls))]:
        if i:
            sheet2_out.append('True')
        else:
            sheet2_out.append('')
    
    offspring_outs_Gen.append(sheet2_out)
    generation += 1

filefullpath='output.xlsx'
if os.path.exists(filefullpath):
    os.remove(filefullpath)

with pd.ExcelWriter(filefullpath) as writer:
    apd11=pd.DataFrame(offspring_outs,columns=columns_new)   
    apd12=pd.DataFrame(offspring_outs_Gen,columns=columns_new2)
    
    apd11.to_excel(writer,sheet_name='details for all combinations') 
    apd12.to_excel(writer,sheet_name='details for each iteration')  

os.startfile(filefullpath)

