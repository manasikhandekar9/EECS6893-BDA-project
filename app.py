rom pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import FMRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
#from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
import matplotlib.pyplot as plt

sc = SparkSession.builder.appName('customer_retention') \
            .getOrCreate()


st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title('''Customer Retention Prediction''')
st.subheader('Machine Learning Regression model comparison in MLlib. \
    Dataset [link](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)')
st.subheader('Built by [Manasi Khandekar](https://www.linkedin.com/in/manasikhandekar/) \
    Github repo [here](https://github.com/manasikhandekar9/bda-project)')


#########   DATA

#df = sc.read.csv("file:///home/hduser/programs/airbnb-price-pred/airbnb.csv", header=True)
df = sc.read.csv("preprocessed_data.csv", header=True)

st.dataframe(data = df.toPandas().head(10))
st.text('Our target variable is price and we are giving vectorized data to the mllib.')
st.text('Below shown data are results of the testing data.')