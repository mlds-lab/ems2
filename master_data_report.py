import pickle
import pandas as pd
import seaborn; seaborn.set()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import Imputer
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, accuracy_score
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy 
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from fancyimpute.knn import  KNN
from fancyimpute.mice import  MICE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
from IPython.display import display
import glob
from IPython.core.display import HTML

import userid_map
import csv_utility
    
# import experiment_engine2


def summary_report(df):
    """
    Produces a statistical report describing the pertinent features of the summarization run.

    Args:
        df (pandas.DataFrame): The summarized data set to analyze and report on.

    Returns:
        df_summary (pandas.DataFrame): A dataframe representing the results of the statistical analysis.
        unnamed dataframe (pandas.DataFrame): A list of the users whose summarized data was analyzed.
    """

    #Filter derived streams out    
    streams = df
    cols_to_drop=[]
    for s in(streams):
        if "(" in s:
            cols_to_drop.append(s)
    df = df.drop(columns = cols_to_drop) 

    #Basic staistics
    N = df.shape[0]
    D = df.shape[1]
    U = df.index.levels[0]
    NU = len(U)

    df_miss = df.apply(np.isnan)
    df_obs = np.logical_not(df.apply(np.isnan))

    #Missing Data Stats
    Total_Miss = df_miss.sum().sum()
    Total_Obs  = df_obs.sum().sum()
    Total_Miss_Rate = round(100*Total_Miss/float(Total_Obs+Total_Miss))

    u1 = set(U)
    df["count"]=1
    u2=set(df["count"].groupby("Participant").sum().index)
    NWD = len(u2)

    #Summary stats data frame
    df_summary=pd.DataFrame([("Number of Subjects",NU),
                             ("Number of Subjects with any data",NWD),
                             ("Number of Features", D),
                             ("Total Subject-Days of Data",N),
                             ("Total Missing Data Rate (%)",Total_Miss_Rate),
                            ],columns=["Statistic","Value"])
    #df_summary["Value"]=df_summary["Value"].apply(np.int)

    return(df_summary, pd.DataFrame(list(u1-u2),columns=["Users"]))


def subject_report(df):
    """
    Produces a statistical report of user data from a summarized data set.

    Args:
        df (pandas.DataFrame): The summarized data set.

    Returns:
        df_user (pandas.DataFrame): A dataframe representing the statistical features per user.
    """

    #df     = df.dropna(axis=0,how='all')

    #Filter derived streams out    
    streams = df
    cols_to_drop=[]
    for s in(streams):
        if "(" in s:
            cols_to_drop.append(s)
    df = df.drop(columns = cols_to_drop)       

    #Create data Frames
    df_user     = pd.DataFrame()

    df_miss = df.apply(np.isnan)
    df_obs = np.logical_not(df.apply(np.isnan))

    df["count"]=1
    df_user["Total Days"] = df["count"].groupby("Participant").sum()
    df_user["Total Observed Vals"] = df_obs.groupby("Participant").sum().sum(axis=1)
    df_user["Total Missing Vals"] = df_miss.groupby("Participant").sum().sum(axis=1)
    df_user["Missing Data Rate"] = df_user["Total Missing Vals"]/(df_user["Total Missing Vals"]+df_user["Total Observed Vals"])

    #df_user["Total Observed Target Vals"] = df_obs["target"].groupby("Participant").sum()
    #df_user["Total Missing Target Vals"] = df_miss["target"].groupby("Participant").sum()
    #df_user["Missing Target Rate"] = df_user["Total Missing Target Vals"]/(df_user["Total Missing Target Vals"]+df_user["Total Observed Target Vals"])

    df_user=df_user.sort_values(["Missing Data Rate"],ascending=False)
    df_user["Missing Data Rank"] = np.arange(len(df_user))
    #df_user=df_user.sort_index()
    
    return(df_user)

    
def feature_report(df):
    """
    Computes statistical features in terms of number of observed and missing values. In addition some statistics
    for the missing value model (e.g. rank of the model, and missing data rate) are computed.

    Args:
        df (pandas.DataFrame): input data

    Returns:
        data including computed statistical features (pandas.DataFrame)

    """
    
    #Filter derived streams out    
    streams = df
    cols_to_drop=[]
    for s in(streams):
        if "(" in s:
            cols_to_drop.append(s)
    #df = df.drop(columns = cols_to_drop)     
    
    df     = df.dropna(axis=0,how='all')
    
    df_features     = pd.DataFrame()
    df["count"]=1

    df_miss = df.apply(np.isnan)
    df_obs = np.logical_not(df.apply(np.isnan))

    df_features["Total Observed Vals"] = df_obs.sum(axis=0)
    df_features["Total Missing  Vals"] = df_miss.sum(axis=0)
    df_features["Missing Data Rate"] = 100*(df_features["Total Missing  Vals"]/(df_features["Total Missing  Vals"]+df_features["Total Observed Vals"]))

    temp = df_obs.groupby("Participant").mean()>0
    df_features["Total Subjects with some Values"] = temp.sum(axis=0)

    temp = df_obs.groupby("Participant").mean()==0
    df_features["Total Subjects with no Values"] = temp.sum(axis=0)

    df_features["Fraction of Subjects with no Values"] = 100*df_features["Total Subjects with no Values"]/(df_features["Total Subjects with no Values"]+df_features["Total Subjects with some Values"])


    df_features=df_features.sort_values(["Missing Data Rate"],ascending=False)
    df_features["Missing Data Rank"] = np.arange(len(df_features))
    
    
    #df_features.sort_index()
    return(df_features)
    
    

