#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
import pyspark.sql.functions as func 
from pyspark.sql.functions import * 
from pyspark.sql.functions import col
from pyspark.sql.functions import lit
spark = SparkSession.builder.appName('split_data').getOrCreate()
import pandas as pd 
from wordcloud import WordCloud,STOPWORDS
import numpy as np 
import matplotlib.pyplot as plt
import random


# In[2]:


file_path=f"gs://6893-projects/data/"
ratings=spark.read.csv(file_path+"ratings.csv",header="true")
movies=spark.read.csv(file_path+"movies1.csv",header='true')
ratings=ratings.dropna() #dropping null rows
movies=movies.dropna()


# In[3]:


movies = movies.toPandas()  #converting RDD to pandas DF


# In[4]:


movies


# In[5]:


ratings = ratings.toPandas()  #converting RDD to pandas DF


# In[6]:


ratings.info()


# In[7]:


movies.info()   #movies information


# In[8]:


ratings.describe() # movies description


# In[9]:


movies.describe()


# In[10]:


d=movies['genres'].str.contains('Drama') #looking at movies with Drama Genre
movies[d].head()


# In[11]:


del ratings['timestamp']


# In[12]:


data_merged=movies.merge(ratings,on = 'movieId',how = 'inner')
data_merged.head(3)


# In[13]:


most_rated=data_merged.groupby('title').size().sort_values(ascending=False)[:10]
most_rated.head(10)


# In[14]:


movies['year']=movies['title'].str.extract('.*\((.*)\).*',expand = False)
movies.head(5)


# In[17]:


labels=set()
for x in movies['genres'].str.split('|').values:
    labels=labels.union(set(x))


# In[19]:


#counts no. of times each genre appear:
data_f=movies
col='genres'
liste=labels
k_occ=[]
key=dict()
for x in liste: key[x] = 0
for l_k in data_f[col].str.split('|'):
    if type(l_k)==float and pd.isnull(l_k): continue
    for x in l_k: 
        if pd.notnull(x):key[x]+=1
for k,v in key.items():
    k_occ.append([k,v])
k_occ.sort(key=lambda x:x[1],reverse =True)
occ=k_occ
num=key


# In[20]:


words=dict()
#Indexing
t_occ=occ[0:70]

for s in t_occ:
    words[s[0]] = s[1]
f,ax=plt.subplots(figsize=(16, 6))
wordcloud=WordCloud(width=400,height=400,background_color='blue',max_words=1600,relative_scaling=0.7,normalize_plurals=False)

wordcloud.generate_from_frequencies(words)

plt.imshow(wordcloud) #plotting word cloud
plt.axis('off')
plt.show()


# In[21]:



fig=plt.figure(1,figsize=(16,13))
ax1=fig.add_subplot(2,1,2)
y=[d[1] for d in t_occ]
x=[k for k,i in enumerate(t_occ)]
x_l=[l[0] for l in t_occ]

#plotting
plt.xticks(rotation=90)
plt.yticks(fontsize=10)
plt.xticks(x,x_l)
plt.ylabel("occurences")
ax1.bar(x,y,align ='center',color='b')
plt.title("Popularity of Genres")
plt.show()


# In[ ]:




