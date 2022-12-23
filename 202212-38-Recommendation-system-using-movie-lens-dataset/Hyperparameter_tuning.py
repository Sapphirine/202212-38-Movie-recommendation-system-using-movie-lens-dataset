#!/usr/bin/env python
# coding: utf-8

# In[15]:


import getpass
import pyspark.sql.functions as func 
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
# import pandas as pd #uncomment if necessary
import csv


# In[16]:


def _func(val):
    _List=[]
    for x in val.rdd.toLocalIterator():
        _IdRow=[]
        for rec_ in x:
            for _tuple in rec_:
                m_Id=_tuple[0]
                _IdRow.append(m_Id )
        _List.append(_IdRow)
    return _List


# In[17]:


def main(spark,sc):
    err=0
    best_MAP=0
    best_rank=-1
    i=0
    _train=spark.read.parquet("gs://6893-projects/data/train_large.parquet") #change path here 
    _val=spark.read.option("header","true").parquet("gs://6893-projects/data/validate_large.parquet") #change path here 
    _test=spark.read.option("header","true").parquet("gs://6893-projects/data/ test_small.parquet") #change path here   

    
    _train= _train.dropna()
    _train.createOrReplaceTempView('_train')
    als_mod=ALS(maxIter=10,implicitPrefs=False,userCol='userId',itemCol='movieId',ratingCol='rating',coldStartStrategy="drop",seed=4321)
    _movie_list=_val.groupby('userId').agg(func.collect_list('movieId'))
    ranks__=[250,225,200,175,150,125,100,75,50,25,1] #Can be changed and initially other value were used 
    reg=[0.1,0.001,0.01,0.02,0.0025,0.5,0.005,0.0005,0.25] #Can be changed and initially other value were used 
    errors=[[0]*len(ranks__)]*len(reg)
    models=[[0]*len(ranks__)]*len(reg)
    
    
    #grid search algorithm
    for r in reg:
        j=0
        for rank in ranks__:
            ground_list=[]
            als_mod.setParams(rank=rank,regParam=r)
            model=als_mod.fit(_train) #fitting training data with above mentioned parameters 
            users=_movie_list.select(als_mod.getUserCol()) #gets random userid's
            predicted_=model.recommendForUserSubset(users,100) #100 predictions for each user
            
            #Ground_truth
            ground_truth=predicted_.join(_movie_list, "userId","inner").select('collect_list(movieId)') #ground_truth
            recs=predicted_.join(_movie_list,"userId","inner").select('recommendations') 
            recs_list=_func(recs)
            for row in ground_truth.rdd.toLocalIterator():
                row=list(row)
                ground_list.append(row[0])
                
            #evaluation using MAP metric     
            lab_sco=list(zip(recs_list, ground_list))
            eval_=sc.parallelize(lab_sco)
            m=RankingMetrics(eval_)
            MAP=m.meanAveragePrecision 
            MAP_1=m.precisionAt(100)
            errors[i][j]=MAP
            models[i][j]=model
            
            #Selecting the best parameter 
            if MAP>best_MAP:
                min_error=MAP
                best_parameters=[i,j]   
            
            _col=[rank,r,MAP,MAP_1]    
            #saving in a csv file 
            
            with open("gs://6893-projects/data/params.csv",'w') as f: #change path here 
                w=csv.writer(f)
                w.writerow(_col)
            print('rank: %s, regularization parameter: %s, MAP: %s, MAP@100 : %s' % (rank,r, MAP, MAP_1))   # 
            j += 1
        i += 1
        
        
    als.setRegParam(regParams[best_parameters[0]]) #setting best value for rank to model
    als.setRank(ranks[best_parameters[1]]) #setting best model for regParam to model
    print ('Best regularization parameter %s' % regParams[best_parameters[0]])
    print ('Best rank %s' % ranks[best_parameters[1]])
    my_model = models[best_parameters[0]][best_parameters[1]] #uncomment if necessary


# In[ ]:


spark=SparkSession.builder.appName('rec_sys').getOrCreate()
sc=spark.sparkContext
main(spark,sc)


# In[ ]:




