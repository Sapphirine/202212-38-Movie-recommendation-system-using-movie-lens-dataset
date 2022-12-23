#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8 
import pyspark as pys
from pyspark.sql import SparkSession
import pyspark.sql.functions as func 
from pyspark.sql.functions import * 
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import lit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
spark = SparkSession.builder.appName('split_data').getOrCreate()


# In[8]:


train_data=spark.read.option("header","false").csv("gs://6893-projects/data/train_large.csv/part-00000-f934b5ae-f98c-48e1-a101-5a799be63f86-c000.csv", schema = 'userId INT, rating FLOAT, movieId INT, Name STRING,genres STRING')
test_data=spark.read.option("header","true").csv("gs://6893-projects/data/test_large.csv/part-00000-e46bc692-bdfe-4857-ba84-3cd3967048fa-c000.csv",  schema = 'userId INT, rating FLOAT, movieId INT, Name STRING,genres STRING')
data=spark.read.csv("gs://6893-projects/data/ratings.csv",inferSchema=True,header=True)
k=spark.read.csv("gs://6893-projects/data/movies1.csv",inferSchema=True,header=True)


# ## 5 Predictions for all users with the best parameters

# In[3]:


als_mod=ALS(rank=250,regParam=0.1,maxIter=10,implicitPrefs=False,userCol='userId',itemCol='movieId',ratingCol='rating',coldStartStrategy="drop",seed=1234) #update best parameters here
model=als_mod.fit(train_data)


# In[5]:


v=test_data.groupby('userId').agg(func.collect_list('movieId')) 
users=v.select(als_mod.getUserCol())
pred=model.recommendForUserSubset(users,5) #5 prediction per user, change value according to requirement


# In[6]:


anda=pred.toPandas()
print(anda)
k1=anda.to_csv()


#    ## Prediction for all user and just user id=11

# In[9]:


(train_data,test_data)=data.randomSplit([0.7,0.8])


# In[ ]:


#Build ALS on the training data with the best paarameters found
als_mod=ALS(rank = 250,maxIter=10,regParam=0.1,userCol="userId",itemCol="movieId",ratingCol="rating")
model=als_mod.fit(train_data)


# In[12]:


#Predict
predictions=model.transform(test_data)


# In[13]:


new_join=predictions.join(k,['movieID'])


# In[15]:


new_join.coalesce(1).write.csv("gs://6893-projects/data/all_rec3.csv",header="True")


# In[18]:


s_u=test_data.filter(test_data['userId']==11).select(['movieId','userId'])
rec=model.transform(s_u)
rec.orderBy('prediction',ascending=False).show()
#reccomendations.coalesce(1).write.csv("gs://6893-projects/data/rec_11.csv",header="True")


# In[ ]:




