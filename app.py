import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer, VectorAssembler, Normalizer, StandardScaler, MinMaxScaler
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType, DateType
from pyspark.sql.functions import to_date, datediff
from pyspark.sql.functions import concat, lit, avg, split, isnan, when, count, col, sum, mean, stddev, min, max, round
from pyspark.sql import Window
from pyspark.ml.classification import LogisticRegression, GBTClassifier, NaiveBayes, RandomForestClassifier, LinearSVC, DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import Bucketizer
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel, GBTClassificationModel, NaiveBayesModel
import streamlit as st
import csv

spark = SparkSession.builder.appName('customer_retention') \
            .getOrCreate()
sqlContext = SQLContext(spark)


st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title('''Customer Retention Analysis for Music Streaming Services''')
#st.subheader('Predict if the User is likely to churn')
st.subheader( 'Github repo [here](https://github.com/manasikhandekar9/bda-project)')

uploaded_file = st.file_uploader("Choose a file", type="csv")


def create_features(uploaded_file):
    stringIndexerGender = StringIndexer(inputCol="gender", outputCol="genderIndex", handleInvalid = 'skip')
    stringIndexerLevel = StringIndexer(inputCol="last_level", outputCol="levelIndex", handleInvalid = 'skip')
    stringIndexerState = StringIndexer(inputCol="last_state", outputCol="stateIndex", handleInvalid = 'skip')
    CategoricalFeatures = ['gender', 'last_level', 'last_state']
    indexers = [StringIndexer(inputCol = col,
    outputCol = "{}Index".format(col))\
                          for col in CategoricalFeatures]
    encoder = [OneHotEncoder(inputCols=["genderIndex", "last_levelIndex", "last_stateIndex"],
                                       outputCols=["genderVec", "levelVec", "stateVec"],
                                handleInvalid = 'keep')]

    features = ['genderVec', 'levelVec', 'stateVec', 'days_active', 'avg_songs', 'avg_events', 'thumbs_up', 'thumbs_down', 'addfriend']
    assembler = [VectorAssembler(inputCols=features, outputCol="rawFeatures")]

    normalizer = Normalizer(inputCol="rawFeatures", outputCol="features", p=1.0)

    preprocessor = Pipeline(stages = indexers + encoder + assembler + [normalizer]).fit(uploaded_file)
    preprocessed_df = preprocessor.transform(uploaded_file)
    preprocessed_df = preprocessed_df.withColumnRenamed("last_levelIndex", "levelIndex")\
       .withColumnRenamed("last_stateIndex", "stateIndex")
    
    return preprocessed_df
    

def trained_model(test):
                rf_model = RandomForestClassificationModel.load("model")
                rf_pred_test = rf_model.transform(test).cache()
                #rf_predictionAndLabels_test = rf_pred_test.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))
                # Instantiate metrics object
                #metrics_test = MulticlassMetrics(predictionAndLabels_test)
                #rf_test = metrics_test.weightedFMeasure()
                rf_test = 0.77
                results = rf_pred_test[['prediction', 'label']]
                return results
            
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    data = create_features(sqlContext.createDataFrame(dataframe))
    #st.dataframe(data = data.toPandas().head(10))
    st.text('Our target variable is churn and we are giving vectorized data to the model.')
    if st.button('Predict', key='1'):
                data = data.withColumnRenamed("churn", "label")
                results_data = trained_model(data)
                st.dataframe(data = results_data.toPandas().head(10))
                st.write("The user is likely to churn")

st.write("OR")
st.write("Enter Attributes")
uid = st.number_input("User Id")
gender = st.radio("Gender", ('M', 'F'))
level = st.radio("Subscription Level", ('Free','Paid'))
active_days = st.number_input("Active days")
state = st.selectbox(
    'Last State',
    ('PA','TX', 'FL', 'WI', 'IL', 'NC', 'SC', 'AZ', 'CT', 'NH', 'OTHER'))
avg_songs = st.number_input("Avg Songs")
avg_events = st.number_input("Avg Events")
thumbsup = st.number_input("Thumbs Up")
thumbsdown = st.number_input("Thumbs Down")
add_friend = st.number_input("Add Friend")
fields = [uid, gender, level,active_days, state, avg_songs, avg_events, thumbsup, thumbsdown, add_friend]
 

if st.button('Predict', key='2'):
            with open('user.csv','a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(fields)
            df = spark.read.format("csv").options(header="false", inferschema="true").load("user.csv")
            d = {
                        "userId": [uid],
                        "gender": [gender],
                        "churn": "0",	
                        "last_level": [level],	
                        "days_active": [active_days],	
                        "last_state": [state],	
                        "avg_songs": [avg_songs],	
                        "avg_events": [avg_events],	
                        "thumbs_up": [thumbsup],	
                        "thumbs_down": [thumbsdown],	
                        "addfriend": [add_friend]
            }
            df = pd.DataFrame(data=d)
            #st.write(df.info())
            #st.dataframe(data = df.head(10))
            data = create_features(sqlContext.createDataFrame(df))
            data_ml = data.withColumnRenamed("churn", "label")
            #st.dataframe(data = data_ml.toPandas().head(10))
            results_data = trained_model(data_ml)
            st.text("results:",results_data)
            st.dataframe(data = results_data.toPandas().head(1))
            #results =  results_data.toPandas()
            st.write("The user is likely to churn")
 
            
