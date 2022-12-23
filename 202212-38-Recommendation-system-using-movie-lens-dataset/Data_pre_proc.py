#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# coding: utf-8 
import pyspark as pys
from pyspark.sql import SparkSession
import pyspark.sql.functions as func 
from pyspark.sql.functions import * 
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import lit
spark = SparkSession.builder.appName('split_data').getOrCreate()


# In[6]:


file_path = f"gs://6893-projects/data/"
ratings=spark.read.csv(file_path+"ratings.csv",header="true")
movies=spark.read.csv(file_path+"movies1.csv",header='true')
movies=movies.dropna() #Dropping Null columns
ratings=ratings.dropna()

#merging based on movie id
ratings=ratings.join(movies,ratings.movieId==movies.movieId).drop(ratings.movieId).drop(ratings.timestamp)
fractions_init=ratings.select("userId").distinct().withColumn("fraction", lit(0.8)).rdd.collectAsMap() 
train_init=ratings.stat.sampleBy("userId",fractions_init,1231) #stratified sampling 
test=ratings.subtract(train_init)

#splitting the data into train test and validate and saving it in the form of csv 
_fin=train_init.select("userId").distinct().withColumn("fraction",lit(0.8)).rdd.collectAsMap()
train=train_init.stat.sampleBy("userId",_fin,1231)
val = train_init.subtract(train)

#csv save
train_init.coalesce(1).write.csv(file_path+"train_large.csv")
test.coalesce(1).write.csv(file_path+"test_large.csv")
val.coalesce(1).write.csv(file_path+"validate_large.csv")


# In[1]:


def main(spark):
    
    sch = 'userId INT, rating FLOAT, movieId INT, str movie_name,timestamp LONG'
    
    #Reading csv files preprocessing the data and converting it to parquet files
    _train = spark.read.csv(f"gs://6893-projects/data/train_large.csv",schema = sch)
    _val=spark.read.option("header","true").csv(f"gs://6893-projects/data/validate_large.csv",schema=sch)
    _test=spark.read.option("header","true").csv(f"gs://6893-projects/data/test_large.csv",schema=sch)
    
    #Dropping Null rows preprocessing step
    _train=_train.dropna()
    _val=_val.dropna()
    _test=_test.dropna()
    
    
    _train_sorted=df_train.sort("userId",ascending= True)
    _val_sorted=df_val.sort("userId",ascending= True)
    _test_sorted=df_test.sort("userId",ascending= True)
    
    #Writing in parquet 
    _train_sorted.repartition(1).write.parquet("gs://6893-projects/data/train_large.parquet")
    _val_sorted.repartition(1).write.parquet("gs://6893-projects/data/val_large.parquet")
    _test_sorted.repartition(1).write.parquet("gs://6893-projects/data/test_small.parquet")


# In[ ]:


if __name__ == "__main__":
    spark = SparkSession.builder.appName('rec_sys').getOrCreate()
    main(spark)

