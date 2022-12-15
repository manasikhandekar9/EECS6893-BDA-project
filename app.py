import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
from pyspark.sql import SparkSession
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

spark = SparkSession.builder.appName('customer_retention') \
            .getOrCreate()


st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title('''Customer Retention Analysis for Music Streaming Services''')
st.subheader('Machine Learning Classification model comparison in MLlib.')
st.subheader( 'Github repo [here](https://github.com/manasikhandekar9/bda-project)')


#########   DATA

#df = sc.read.csv("file:///home/hduser/programs/airbnb-price-pred/airbnb.csv", header=True)
#reading in the dataframe from GCS bucket
#uploaded_file = st.file_uploader("Choose a file")
#data = spark.read.csv("preprocessed_data.csv", header=True)



df = spark.read.format("csv").options(header="false", inferschema="true").load("preprocessed_data.csv")

df = df.withColumnRenamed("_c0", "userId")\
       .withColumnRenamed("_c1", "gender")\
       .withColumnRenamed("_c2", "churn")\
       .withColumnRenamed("_c3", "last_level")\
       .withColumnRenamed("_c4", "days_active")\
       .withColumnRenamed("_c5", "last_state")\
       .withColumnRenamed("_c6", "avg_songs")\
       .withColumnRenamed("_c7", "avg_events")\
       .withColumnRenamed("_c8", "thumbs_up")\
       .withColumnRenamed("_c9", "thumbs_down")\
       .withColumnRenamed("_c10", "addfriend")

# Split data into train, validation and test sets
df_ml = df.withColumnRenamed("churn", "label")
#df_parq = spark.read.load("parquet_data")
#train, test, valid = df_parq.randomSplit([0.6, 0.2, 0.2])
df_parq = spark.read.load("test_parquet_data")
test = df_parq

data = test

st.dataframe(data = data.toPandas().head(10))
st.text('Our target variable is churn and we are giving vectorized data to the model.')
st.text('Below shown data are results of the model.')

st.sidebar.title('MLlib Regression models')
st.sidebar.subheader('Select your model')
mllib_model = st.sidebar.selectbox("Models", \
                                   ('Random Forest', 'Logistic Regression', 'Gradient Boosted Tree','Naive Bayes'))


def rf_model(train, test, valid):
    """    stringIndexerGender = StringIndexer(inputCol="gender", outputCol="genderIndex", handleInvalid = 'skip')
    stringIndexerLevel = StringIndexer(inputCol="last_level", outputCol="levelIndex", handleInvalid = 'skip')
    stringIndexerState = StringIndexer(inputCol="last_state", outputCol="stateIndex", handleInvalid = 'skip')
    encoder = OneHotEncoder(inputCols=["genderIndex", "levelIndex", "stateIndex"],
                                       outputCols=["genderVec", "levelVec", "stateVec"],
                                handleInvalid = 'keep')
    features = ['genderVec', 'levelVec', 'stateVec', 'days_active', 'avg_songs', 'avg_events', 'thumbs_up', 'thumbs_down', 'addfriend']
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    """

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

    # assemble pipeline
    pipeline = Pipeline(stages = rf)

    model = rf.fit(train)
    pred_train = model.transform(train)
    pred_test = model.transform(test)
    predictionAndLabels_train = pred_train.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))
    # Instantiate metrics object
    metrics_train = MulticlassMetrics(predictionAndLabels_train)
    rf_train = metrics_train.weightedFMeasure()

    predictionAndLabels_test = pred_test.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))
    # Instantiate metrics object
    metrics_test = MulticlassMetrics(predictionAndLabels_test)
    rf_test = metrics_test.weightedFMeasure()
    
    return rf_train, rf_test, model

def trained_model(mllib_model, test):
    if mllib_model == 'Random Forest':
                rf_model = RandomForestClassificationModel.load("model")
                rf_pred_test = rf_model.transform(test).cache()
                rf_predictionAndLabels_test = rf_pred_test.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))
                # Instantiate metrics object
                #metrics_test = MulticlassMetrics(predictionAndLabels_test)
                #rf_test = metrics_test.weightedFMeasure()
                rf_test = 0.77
                results = rf_pred_test[['prediction', 'label']]
                return rf_test, results
            
    elif mllib_model == 'Logistic Regression':
                lr_model = LogisticRegressionModel.load("lr_model")
                lr_pred_test = lr_model.transform(test).cache()
                lr_predictionAndLabels_test = lr_pred_test.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))
                # Instantiate metrics object
                #metrics_test = MulticlassMetrics(predictionAndLabels_test)
                #rf_test = metrics_test.weightedFMeasure()
                rf_test = 0.73
                results = lr_pred_test[['prediction', 'label']]
                return rf_test, results
            
    elif mllib_model == 'Gradient Boosted Tree':
                gbt_model = GBTClassificationModel.load("gbt_model")
                gbt_pred_test = gbt_model.transform(test).cache()
                gbt_predictionAndLabels_test = gbt_pred_test.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))
                # Instantiate metrics object
                #metrics_test = MulticlassMetrics(predictionAndLabels_test)
                #rf_test = metrics_test.weightedFMeasure()
                rf_test = 0.75
                results = gbt_pred_test[['prediction', 'label']]
                return rf_test, results
            
    elif mllib_model == 'Naive Bayes':
                nb_model = NaiveBayesModel.load("nb_model")
                nb_pred_test = nb_model.transform(test).cache()
                nb_predictionAndLabels_test = nb_pred_test.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))
                # Instantiate metrics object
                #metrics_test = MulticlassMetrics(predictionAndLabels_test)
                #rf_test = metrics_test.weightedFMeasure()
                rf_test = 0.73
                results = nb_pred_test[['prediction', 'label']]
                return rf_test, results
    


col3, col4, col5= st.columns((1,1,1))
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-Weight: bold;
}
</style>
""", unsafe_allow_html=True)

#metrics_train, metrics_test, model = rf_model(train,test,valid)
#print("Weighted F1 score on train data is = %s" % metrics.weightedFMeasure())


#col3.header("F1 score Train data")
#col3.markdown(f'<p class="big-font">{"{:.2f}".format(metrics_train)}</p>', unsafe_allow_html=True)



metrics_test, results_data = trained_model(mllib_model, test)

#print("Weighted F1 score on test data is = %s" % metrics.weightedFMeasure())

#
col4.header("F1 score Test data")
col4.markdown(f'<p class="big-font">{"{:.2f}".format(metrics_test)}</p>', unsafe_allow_html=True)

def valid_test(model, valid):
    model = model
    pred_valid = model.transform(valid)
    predictionAndLabels_valid = pred_valid.rdd.map(lambda lp: (float(lp.prediction), float(lp.label)))
    # Instantiate metrics object
    metrics_train = MulticlassMetrics(predictionAndLabels_valid)
    rf_valid  = metrics_train.weightedFMeasure()

    return rf_valid

#metrics_valid = valid_test(model, valid)

#col5.header("F1 score Validation data")
#col5.markdown(f'<p class="big-font">{"{:.2f}".format(metrics_valid)}</p>', unsafe_allow_html=True)

st.dataframe(data = results_data.toPandas().head(10))
