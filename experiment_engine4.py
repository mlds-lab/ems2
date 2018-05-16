import sys
from sys import argv
import argparse
import os
import pickle
import pandas as pd
# import seaborn; seaborn.set()
# import matplotlib.pyplot as plt
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
from numpy import nanmean
from numpy import nanstd
import json

from sklearn.utils import shuffle
from sklearn.svm import SVR
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy 
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
import glob

import userid_map
import csv_utility
#from fancyimpute.knn import KNN
#from fancyimpute.soft_impute import SoftImpute
#from fancyimpute.iterative_svd import IterativeSVD
#import fancyimpute
#import fancyimpute.soft_impute
#import fancyimpute.iterative_svd

from pyspark import SparkConf, SparkContext
import socket
import copy

import environment_config as ec

ENVIRONMENT = socket.gethostname()

np.random.seed(42)
np.random.seed(42)

# # VM configuration
# if ENVIRONMENT == "cerebralcortex":

#     print("experiment_engine: detected VM environment")

#     # MINIO_IP = '127.0.0.1:9000'
#     # MINIO_ACCESS_KEY = 'ZngmrLWgbSfZUvgocyeH'
#     # MINIO_SECRET_KEY = 'IwUnI5w0f5Hf1v2qVwcr'
#     # EDD_DIRECTORY = "{}"
#     # cc = CerebralCortex('/home/vagrant/CerebralCortex-DockerCompose/cc_config_file/cc_vagrant_configuration.yml')
#     sc = SparkContext("local[8]", "MOSAIC")

# # production configuration
# elif ENVIRONMENT in ["dagobah10dot"]:

#     print("experiment_engine: detected production environment")

#     # cc = CerebralCortex('/cerebralcortex/code/config/cc_starwars_configuration.yml')
#     # minioClient = cc
#     conf = SparkConf().setMaster('spark://dagobah10dot:7077').setAppName('MOSAIC-EMS - ' + sys.argv[1]).set('spark.cores.max','64').set('spark.ui.port','4099').setExecutorEnv('PYTHONPATH',str(os.getcwd()))
#     sc = SparkContext(conf=conf)
#     sc.addPyFile('/cerebralcortex/code/eggs/MD2K_Cerebral_Cortex-2.2.2-py3.6.egg')

#     # MINIO_IP = cc.config['minio']['host'] + ':' + str(cc.config['minio']['port'])
#     # MINIO_ACCESS_KEY = cc.config['minio']['access_key']
#     # MINIO_SECRET_KEY = cc.config['minio']['secret_key']
#     # EDD_DIRECTORY = "/cerebralcortex/code/ems/EMS/{}"

#     # print(MINIO_IP,MINIO_ACCESS_KEY,MINIO_SECRET_KEY)
    
# else:
#     print("\n\n\n Unknown Environment!")

sc = ec.get_spark_context(ENVIRONMENT)

def df_shifted(df, target=None, lag=0, f="D"):
    """
    This function lags data features with respect to the target scores.

    Args:
        df (data frame): original data
        target (string): target name
        lag (int): period of the lag
        f (string): Increment to use for time frequency

    Returns:
        lagged data frame (dataframe)

    """

    if not lag and not target:
        return df       
    new = {}
    for c in df.columns:
        if c == target:
            new[c] = df[target]
        else:
            new[c] = df[c].shift(periods=lag, freq=f)
    return pd.DataFrame(data=new)

def corr_score(model,X,y):
    return np.corrcoef(y, model.predict(X))[0,1]

def kcorr_score(model,X,y):
    return scipy.stats.kendalltau(y,model.predict(X))[0]

def raw_df_to_train(df, tr_ids, te_ids, params):
    """
    This function filters the data, imputes missing values, and removes outliers.

    Args:
        df (data frame): input data
        tr_ids (list of ints): indices for the training data
        te_ids (list of ints): indices for the testing data
        params (dictionary): parameters required for filtering/imputation/outlier removal

    Returns:
        numpy arrays for training and testing data features, labels, and user ids.
    """
    
    df2     = copy.deepcopy(df)
    tr_ids2 = tr_ids.copy() 
    te_ids2 = te_ids.copy() 
    params2 = params.copy()
        
    print("Loaded Data Frame with %d instances and %d markers"%(df2.shape[0],df2.shape[1]) )

    df_new = df2[df2.index.get_level_values('Participant').isin(list(tr_ids2)+list(te_ids2) )]
    df=df_new.copy()


    #Fill in the nans on columns that are positive only labels with zeros
    features = list(df.columns)
    zero_fill_columns = []
    for f in features:
        if(".app_usage." in f):
            zero_fill_columns.append(f)
    df_zero_fill = df[zero_fill_columns]
    df_zero_fill=df_zero_fill.fillna(0)
    df[zero_fill_columns] = df_zero_fill
    print(len(zero_fill_columns))

    #Filter users with too few values
    df["Missing Indicator"] = df["org.md2k.data_analysis.feature.phone.driving_total.day"] + \
                                df["org.md2k.data_analysis.feature.phone.bicycle_total.day"] + \
                                df["org.md2k.data_analysis.feature.phone.still_total.day"] +\
                                df["org.md2k.data_analysis.feature.phone.on_foot_total.day"]+ \
                                df["org.md2k.data_analysis.feature.phone.tilting_total.day"]+ \
                                df["org.md2k.data_analysis.feature.phone.walking_total.day"]+\
                                df["org.md2k.data_analysis.feature.phone.running_total.day"]+\
                                df["org.md2k.data_analysis.feature.phone.unknown_total.day"]

    #Collapse the data if intake experiment
    if(params2["experiment-type"]=="intake"):
        df_mean = df.groupby("Participant").mean()
        df_std  = df.groupby("Participant").std()
        df_max  = df.groupby("Participant").max()
        df_min  = df.groupby("Participant").min()
        for c in  df.columns:
            if("target" not in c):
                df_mean[c+"-std"] = df_std[c]
                df_mean[c+"-min"] = 1.0*df_min[c]
                df_mean[c+"-max"] = 1.0*df_max[c]
        df = df_mean
    
    df.dropna(axis=0, subset=["Missing Indicator"],inplace=True)
    df=df.drop(columns=["Missing Indicator"])

    print("  ... Contains %d instances with core features available"%(df.shape[0]) )

    df.dropna(axis=0, subset=["target"],inplace=True)
    
    print("  ... Contains %d instances with defined targets"%(df.shape[0]) )
        
    df.dropna(axis=1, thresh=params["miss_thresh"]*df.shape[0],inplace=True)
    
    print("  ... Contains %d markers that are >=%.2f pct observed\n"%(df.shape[1], params["miss_thresh"]) )

    # Get original feature, no target
    features = list(df.columns)
    features.remove("target")

    # Add day of week indicators
    if (params2["add_days"]):
        for i in range(7):
            df["day%d"%(i)] = 1*np.array(df.index.get_level_values('Date').map(lambda x: x.dayofweek)==i)

    ###Remove!!!!!
    #df=df.fillna(0)

    # Run Imputation on data frame
    numeric = df[features].as_matrix()
    if(np.any(np.isnan(numeric))):
        
        rank=min(25,max(1,int(0.25*numeric.shape[1])))
        from fancyimpute.iterative_svd import IterativeSVD            
        imp=IterativeSVD(verbose=False, rank=rank,init_fill_method="mean",convergence_threshold=1e-5,random_state=42).complete(numeric)
        #imp=KNN(verbose=True, k=100).complete(numeric)
        
        df[features] = pd.DataFrame(data=imp, columns=features,index=df.index)

    if(params2["add_cum_mean"]):
        #Add cum-means for all original columns
        for f in features:   
            df["%s-cmean"%f]=df[f].groupby("Participant").expanding().mean().values
            #df["%s-cmax"%f]=df[f].groupby("Participant").expanding().max().values

    if(params2["add_cum_max"]):
        #Add cum-means for all original columns
        for f in features:   
            df["%s-cmax"%f]=df[f].groupby("Participant").expanding().max().values
            #df["%s-cmax"%f]=df[f].groupby("Participant").expanding().max().values

    if(params2["add_cum_std"]):
        #Add cum-means for all original columns
        for f in features:   
            df["%s-cstd"%f]=df[f].groupby("Participant").expanding().std().values
            #df["%s-cmax"%f]=df[f].groupby("Participant").expanding().max().values

                
    #Lag all columns except for target and day  of week    
    #Add specified lags, but not lag 0
    for l in params2["lags"]:
        if(l>0):
            for f in features:
                df["%s-%d"%(f,l)]=df.groupby(level=0)[f].shift(l)
    #Drop original columns if not using lag 0
    if not (0 in params["lags"]):
        df = df.drop(columns=features)
                
    #Make sure no missing values are left
    df=df.fillna(df.mean())
    
    
    #df_new = copy.deepcopy(df[["target"]])
    if(params2["add_pca"]):
        features = list(df.columns)
        features.remove("target")
        numeric = df[features].as_matrix()
        from sklearn.decomposition import IncrementalPCA
        K= min(params["max_pca_K"],max(1,numeric.shape[1]))
        ipca=IncrementalPCA(n_components=K)
        ipca.fit(numeric)
        Zs = ipca.transform(numeric)
        for k in range (K):
            df["PCA%d"%(k)] = Zs[:,k]


  
    #Sort all columns by name
    cols = list(df.columns)
    cols.sort()
    df = df[cols]
        
    
    df_tr = df[df.index.get_level_values('Participant').isin(list(tr_ids2))]
    df_te = df[df.index.get_level_values('Participant').isin(list(te_ids2))]
    
    #Extract data  matrices
    #Targets
    Y_tr = df_tr["target"].as_matrix().astype(float)
    Y_te = df_te["target"].as_matrix().astype(float)
                                                             
    #Features
    features = list(df.columns)
    features.remove("target")
    X_all = df[features].as_matrix().astype(float)                                                    
    X_tr  = df_tr[features].as_matrix().astype(float)
    X_te  = df_te[features].as_matrix().astype(float)
                                                                                                                 
    #Filter low std columns based on all data
    ind = np.std(X_all,axis=0)>1e-4 
    X_tr = X_tr[:,ind]
    X_te = X_te[:,ind]
    X_all = X_all[:,ind]
    
    #Scale data based on overall mean and std                                                         
    mean = np.mean(X_all,axis=0)
    std  = np.std(X_all,axis=0)                                            
    X_tr = (X_tr-mean)/std
    X_te = (X_te-mean)/std                                                         
                                                             
    features=np.array(features)[ind]
          
          
    if(params2["transfer_filter"]):
        Z = np.hstack((Y_tr[:,np.newaxis],X_tr))
        Corr = np.corrcoef(Z)
        
        X_tr_mean = np.mean(X_tr,axis=0)
        X_tr_std  = np.std(X_tr,axis=0)
        X_te_mean = np.mean(X_te,axis=0)
        X_te_std  = np.std(X_te,axis=0)
        
        Ntr = X_tr.shape[0]
        Nte = X_te.shape[0]
        
        SE = np.sqrt(X_tr_std**2/Ntr + X_te_std**2/Nte)
        tstat = np.abs(X_tr_mean-X_te_mean)/(1e-4+ SE)
        
        ind = tstat <1
        if(np.sum(ind)==0):
            ind[np.argmin(tstat)]=1
        
        X_tr = X_tr[:,ind]
        X_te = X_te[:,ind]
        features=np.array(features)[ind]
        
        print("Filtered %d features"%(np.sum(1-ind)))          
          
          
          
          
          
          
            
    #Row groupings by user id
    G_tr = np.array(df_tr.index.get_level_values(0))
    G_te = np.array(df_te.index.get_level_values(0))
                                                             

    #Quality matrix, full since data already imputed
    Q_tr=1-np.isnan(X_tr)
    Q_te=1-np.isnan(X_te)                                                             

    # Dummy marker groups
    MG=np.arange(X_all.shape[1])
    
    return(X_tr,Y_tr,Q_tr,G_tr,X_te,Y_te,Q_te,G_te,MG, features, df_tr, df_te)

class trainTestPerformanceEstimator:
    def __init__(self, indicator_name, model, features,  hyperparams, cvfolds, cvtype):
        """
        This class estimates the performance of a given estimator model different metrics; In addition, ablation
        testing is performed and a summary of results is produced.

        Args:
            indicator_name (string): score name
            model (object): learning model wrapped in a groupCVLearner's object
            features (list): list of feature names
        """
        
        # self.metrics = [mae, mse, r2_score, lambda x,y:  pearsonr(x,y)[0]]
        # self.metric_names=["MAE", "MSE", "R^2", "r"]
        
        self.metrics = [lambda x,y:  pearsonr(x,y)[0]]
        self.metric_names=["R", "R Best", "NS","ND"]
        # self.metrics = [mae, mse, lambda x,y:  pearsonr(x,y)[0]]
        # self.metric_names=["MAE", "MSE", "R", "NS","ND"]

        self.cvtype=cvtype
        self.cvfolds=cvfolds
        self.hyperparams=hyperparams
        self.model=model
        self.features=features
        self.ablation_scores=None
        self.num_metrics = len(self.metric_names)
        self.indicator_name = indicator_name
        self.bounds = {'stress.d': [1, 5],
                       'anxiety.d': [1, 5],
                       'pos.affect.d': [5, 25],
                       'neg.affect.d': [5, 25],
                       'irb.d': [7, 49],
                       'itp.d': [1, 5],
                       'ocb.d': [0, 8],
                       'cwb.d': [0, 8],
                       'sleep.d': [0, 24],
                       'alc.quantity.d': [0, 20],
                       'tob.quantity.d': [0, 30],
                       'total.pa.d': [0, 8000],
                       'neuroticism.d': [1, 5],
                       'conscientiousness.d': [1, 5],
                       'extraversion.d': [1, 5],
                       'agreeableness.d': [1, 5],
                       'openness.d': [1, 5],
                       'stress': [1, 5],
                       'anxiety': [1, 5],
                       'irb': [7, 49],
                       'itp': [1, 5],
                       'ocb': [20, 100],
                       'inter.deviance': [7, 49],
                       'org.deviance': [12, 84],
                       'shipley.abs': [0, 25],
                       'shipley.vocab': [0, 40],
                       'neuroticism': [1, 5],
                       'conscientiousness': [1, 5],
                       'extraversion': [1, 5],
                       'agreeableness': [1, 5],
                       'openness': [1, 5],
                       'pos.affect': [10, 50],
                       'neg.affect': [10, 50],                       
                       'stai.trait': [20, 80],  
                       'audit': [0, 40], 
                       'gats.status': [1,3], 
                       'gats.quantity': [0, 80], 
                       'ipaq': [0, 35000], 
                       'psqi': [0, 21]
                   }

    def get_indicator_non_outliers(self, y, score_name):
        """
        This method gets the indices of data labels that are not considered as outliers, i.e. data cases with labels
        within the permissible range for the relevant score name.

        Args:
            y (numpy array): labels
            score_name (string): score name

        Returns:
            indices of labels within the permissible range (numpy array of ints)
        """
        
        ind = np.ones(y.shape) > 0

        if score_name in self.bounds.keys():
            score_range = self.bounds[score_name]
        else:
            print("!! Warning -- score name does not exist in bounds list. No outlier removal.")
            return ind
    
        l, h = score_range[0], score_range[1] 
        if(np.isinf(h)):
           h=np.percentile(y, 95)            
            
        ind = np.logical_and(y>=l, y<=h)
    
        return(ind)

    def estimate_performance(self, Xtrain, ytrain, Gtrain, Xtest, ytest, Gtest):
        """
        This method estimates the performance of the trained model via different metrics.

        Args:
            Xtrain (numpy array): training data
            ytrain (numpy array): training labels
            Gtrain (numpy array): training user ids
            Xtest (numpy array): testing data
            ytest (numpy array): testing labels
            Gtest (numpy array): testing user ids
        """
        
        np.random.seed(42)
        np.random.seed(10)
                        
        self.results        = np.zeros((self.num_metrics,2))
        self.opt_params=[]
        
        # Drop y outliers and adjust  y scale to be [0,1]
        # Make sure to scale back when predicting!
        #ind = self.get_indicator_non_outliers(ytrain,self.indicator_name)
        #Xtrain_sub = Xtrain[ind,:]
        #ytrain_sub = ytrain[ind]        

        #Clip targets to range
        Xtrain_sub = Xtrain
        ytrain_sub = ytrain
        ytrain_sub[ytrain_sub<self.bounds[self.indicator_name][0]]=self.bounds[self.indicator_name][0]
        ytrain_sub[ytrain_sub>self.bounds[self.indicator_name][1]]=self.bounds[self.indicator_name][1]


        #scorer = make_scorer(mean_squared_error, greater_is_better=False)
        
        def safe_pearson(x,y):
            c = pearsonr(x,y)[0]
            if(np.isnan(c)): c=0
            return c
                    
        def mae(x,y):
            return np.mean(np.abs(x-y))
            
        def accuracy(x,y):
            return np.sum(x==y)

        if(self.indicator_name=="gats.status"):
            scorer = make_scorer(accuracy)
        else:
            scorer = make_scorer(safe_pearson)

        #scorer = make_scorer(mae, greater_is_better=False)            

        #scorer = make_scorer(r2_score, greater_is_better=True) 

        #Learn the model using grid search CV
        np.random.seed(0)
        model = self.model()

        y_straight_test=[]
        y_straight_train=[]

        if(self.indicator_name=="gats.status"):
            self.best_test_score  = 0
            self.best_train_score = 0
            pass
        else:
            for a in self.hyperparams["alpha_lasso"]:
                for a1 in self.hyperparams["alpha_ridge"]:
                    model.set_params(alpha_lasso=a, alpha_ridge=a1,bounds=self.bounds[self.indicator_name])
                    model.fit(Xtrain_sub, ytrain_sub)
                    y_hat       = model.predict(Xtest)
                    y_hat_train = model.predict(Xtrain)
                    print("a0: %e a1: %e R: %.4f  MSE: %.4f  MAE:%.4f  Lo: %f  Hi %f"%(a,a1,
                        safe_pearson(ytest,y_hat),
                        mean_squared_error(ytest,y_hat),
                          mae(ytest,y_hat),
                          min(y_hat),
                          max(y_hat)))
                    y_straight_test.append(safe_pearson(ytest,y_hat))
                    y_straight_train.append(safe_pearson(ytrain,y_hat_train))
                
            self.best_test_score = np.max(np.array(y_straight_test))
            self.best_train_score = np.max(np.array(y_straight_train))


        np.random.seed(0)
        if(self.cvtype=="shuffle"):
            from sklearn.model_selection import ShuffleSplit
            ss=ShuffleSplit(n_splits=self.cvfolds, random_state=1111, test_size=0.1, train_size=None)
            cv_splits = ss.split(Xtrain_sub)
        elif(self.cvtype=="group"):            
            group_kfold = GroupKFold(n_splits=self.cvfolds)
            cv_splits   = group_kfold.split(Xtrain_sub, ytrain_sub, groups=Gtrain)
        elif(self.cvtype=="groupshuffle"):
            gss = GroupShuffleSplit(n_splits=self.cvfolds, test_size=0.1,  random_state=1234)
            cv_splits   =  gss.split(Xtrain_sub, ytrain_sub, groups=Gtrain)
        elif(self.cvtype=="loo"):
            from sklearn.model_selection import LeaveOneOut
            loo = LeaveOneOut()
            cv_splits  = loo.get_n_splits(Xtrain_sub, ytrain_sub,)
        else:
            print("Error: Cross validation cv_type=%s not specified"%(self.cvtype))
            exit()
        
        this_hyper=self.hyperparams.copy()
        if(self.indicator_name=="gats.status"):
            pass
        else:
            this_hyper["bounds"]=[self.bounds[self.indicator_name]]    
        
        gs          = GridSearchCV(model, this_hyper, scoring=scorer, refit=True, return_train_score=True, cv=cv_splits, verbose=False)
        m           = gs.fit(Xtrain_sub,ytrain_sub)
        cv          = np.vstack((m.cv_results_['mean_test_score'], m.cv_results_['mean_train_score']) )

        self.bounds[self.indicator_name]
        print(self.hyperparams)
        print(cv.T)

        self.trained_model = m.best_estimator_
        self.opt_params = gs.best_params_

        self.yhat_train = m.predict(Xtrain)
        self.yhat_test  = m.predict(Xtest)
        
        print("Pred Extrema: %f %f %f %f"%(min(self.yhat_train),max(self.yhat_train),min(self.yhat_test),max(self.yhat_test)))
        
        self.ytest=ytest
        self.ytrain=ytrain

        self.Gtrain=Gtrain
        self.Gtest=Gtest

        # Ablation test:
        if(self.indicator_name=="gats.status"):
            pass
        else:      
            self.ablation_test(Xtrain_sub, Xtest, ytrain_sub, ytest)
            
        #self.opt_params.append(self.model.opt_params)
        
        
        self.results[0,0] = safe_pearson(ytest,self.yhat_test)
        self.results[0,1] = safe_pearson(ytrain,self.yhat_train)
        self.results[1,0] = self.best_test_score
        self.results[1,1] = self.best_train_score            
        self.results[2,0] = len(np.unique(Gtest))
        self.results[2,1] = len(np.unique(Gtrain))
        self.results[3,0] = len(ytest)
        self.results[3,1] = len(ytrain)

    def ablation_test(self, X_tr, X_te, y_tr, y_te):
        """
        This method performs ablation testing for the model.

        Args:
            X_tr (numpy array): training data features
            X_te (numpy array): testing data features
            y_tr (numpy array): training labels
            y_te (numpy array): testing labels

        Returns:
            ablation scores (numpy array)
        """

        params = self.opt_params
        feature_support = self.trained_model.feature_support

        # handling special cases:
        if len(feature_support) == 0:
            self.ablation_scores = np.array([])
            return self.ablation_scores
        elif len(feature_support) == 1:
            self.ablation_scores = np.array([0])
            return self.ablation_scores

        feature_support_temp = list(feature_support)
        self.ablation_scores = np.zeros((len(feature_support),))

        from MLE.linear_regression_one import LinearRegressionOne

        if isinstance(self.trained_model, LinearRegressionOne):

            from sklearn.linear_model import Ridge
            model = Ridge(alpha=params['alpha_ridge'])

        #elif isinstance(self.trained_model, NNRegressionOne):

        #    from MLE.nn_regression_one import RidgeNet
        #    ridge_layers = copy.copy(params['ridge_layers'])
        #    ridge_layers.insert(0, len(feature_support) - 1)
        #    model = RidgeNet(alpha=params['alpha_ridge'], layers=ridge_layers)

        for i, feature in enumerate(feature_support):

            feature_support_temp.remove(feature)

            model.fit(X_tr[:, feature_support_temp], y_tr.reshape((-1,1)))
            y_predict = model.predict(X_te[:, feature_support_temp]).reshape(-1)
            #yhat_predict = self.clip_prediction(self.Ymean + self.Yscale * y_predict, self.indicator_name)
            self.ablation_scores[i] = pearsonr(y_te, y_predict)[0]

            feature_support_temp.append(feature)

        return self.ablation_scores

    def report(self):
        """
        This method generates a report in table format from the ablation test.

        Returns:
            data frame with features, model weights, and ablation scores as columns
        """

        types  = ["Test","Train"]
        
        dfperf = pd.DataFrame(data=[self.indicator_name], columns=["Indicator"])      
        for i,t in enumerate(types):
            for m in range(len(self.metric_names)):
                dfperf["%s %s"%(t,self.metric_names[m])] = [self.results[m,i]]
        dfperf["Optimal Hyper-Parameters"] = [str(self.opt_params)]

        print('optimal params = {}'.format(self.opt_params))

        try:
            coef = self.trained_model.ridge_coef
            indf = np.argsort(-np.abs(coef))
            L    = len(self.trained_model.feature_support)
            inds = np.argsort(-np.abs(coef[self.trained_model.feature_support]))
            dffeatures = pd.DataFrame(data=list(zip(self.features[indf[:L]], coef[indf[:L]], self.ablation_scores[inds])),
                                     columns = ["Features", "Weight", "Ablation Scores"])
        except:
            dffeatures = pd.DataFrame(columns = ["Features", "Weight", "Ablation Scores"])
            pass

        pd.options.display.width = 300
        pd.options.display.max_colwidth= 300
        
        print(dffeatures)

        return dfperf, dffeatures

def group_train_test_split(X, y, G):
    """
    This function splits the data into training and testing.

    Args:
        X (numpy array): data features
        y (numpy array): labels
        G (numpy array): user ids

    Returns:
        training and testing data/labels/userIDs (numpy arrays)
    """

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=0)
    ttsplit = gss.split(X, groups=G)
    for train_index, test_index in ttsplit:
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        Gtrain, Gtest = G[train_index], G[test_index]
    return(Xtrain, Xtest, ytrain, ytest, Gtrain, Gtest)

def data_frame_to_csv(df, score_column, score_name, prefix="", results_folder="results/"):
    """
    This function writes the prediction results to csv files.

    Args:
        df (data frame): data cases with prediction results
        score_column (string): name of the column with prediction scores
        score_name (string): score name
        prefix (string): path to write the csv file to and the prefix for file name
    """

    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    ids     = np.array(df.index.get_level_values('Participant'))
    umn_ids = userid_map.perform_map(ids, 'data/mperf_ids.txt')
    vals    = np.array(df[score_column].fillna(0))

    dates   = [x.strftime("%-m/%-d/%Y") for x in df.index.get_level_values('Date')]

    csv_utility.write_csv_daily(results_folder + "%s"%(prefix), umn_ids, dates, np.array([""]*len(dates)), score_name, vals)

def user_split(pct,ids=None,counts=None,num="300"):

    """
    Splits users into training and test sets for input to the machine learning pipeline.

    Args:
        pct (int): The percent split between training and test sets.
        ids (List): Optional list of user IDs.
        counts (List): Sample counts.
        num (str): Size of the subject set.

    Returns:
        train_ids (List): 
        test_ids (List): 
    """

    if(num=="200"):
        orig_ids=np.array(["00ab666c-afb8-476e-9872-6472b4e66b68",
        "02e82ef0-acb8-4366-8c83-4c3f2f69f7ea",
        "03996723-2411-4167-b14b-eb11dfc33124",
        "03c26210-7c9f-4bf2-b1c2-59d0bd64ffac",
        "03ec3750-641a-4039-8b5d-74b485bde1ea",
        "055bed5b-60ec-43e6-9110-137f2a36d65b",
        "059a9d92-4d36-40cb-84cc-408f9210821b",
        "07a34e07-8ee4-434b-8c09-6cb197451464",
        "0990887a-6163-4c80-9c9e-468ea2598202",
        "0a11d9aa-1e9c-4f5e-94cf-faa6e796a855",
        "0c824653-a13b-4a4e-b907-660f1d8f8981",
        "110a98b6-ef01-44b2-a19c-9ab8db1dd6fa",
        "13555453-55fc-4b13-8b32-b49dfcc9a8f0",
        "135c9c3b-a5cf-47a4-9fcf-4fc418c5eb96",
        "136f8891-af6f-49c1-a69a-b4acd7116a3c",
        "15d069af-7a36-4c33-8693-cc814845b9c3",
        "166b9e2f-08a1-425f-9617-875d6ad3f834",
        "16db9568-bd11-494f-b712-28a2266ea3d0",
        "176dfa9b-725b-4813-9219-005041320db4",
        "17b07883-4959-4037-9b80-dde9a06b80ae",
        "1b524925-07d8-42ea-8876-ada7298369ec",
        "1bdf0668-a632-4290-ad94-c6269f9e924a",
        "1c47637b-d4a1-432f-9810-b6e2daea0a43",
        "20a07a1c-ee7e-4958-a2c8-0db6e4fe0ce9",
        "24809752-2bcd-424f-aac1-5d909026b0c4",
        "24e24816-b56b-4c16-9e6d-0ae8afc8650f",
        "260f551d-e3c1-475e-b242-f17aad20ba2c",
        "26d5159f-ca0f-4537-a97d-8ddda3c95f8a",
        "26eeb04e-ab64-45b2-bb12-7aff7a779f9a",
        "2946b01f-4cda-4f69-9b3f-24aed26c4d15",
        "2b4ae7c4-ee25-4588-a0cd-bffb64a07f7f",
        "2bc2ade2-b154-4412-878c-98466a00ba7b",
        "2d954abb-e11a-4f25-8f12-19a210c8d8e1",
        "2f256f48-8336-4846-9362-7349ae648dc5",
        "323cc961-5f66-4273-9d46-6097bc692269",
        "326a6c55-c963-42c2-bb8a-2591993aaaa2",
        "34192284-e4f1-4420-ab35-0eb02ce2e8d1",
        "34223626-fab9-48f5-82bd-0b0037745994",
        "342d9d87-787d-4836-9f61-4d59ed9f3289",
        "34521ae7-012a-400f-8794-3d76ff4e70ab",
        "35daf881-da7c-4779-a8e9-20a3985094e2",
        "35fff957-2ad7-40e6-b57a-30235da5c9a9",
        "360dda05-d7a7-4ec3-8988-256bb6381cda",
        "36caadf6-9bd5-4bac-9f13-75d2f439b4de",
        "36f6e239-816a-4508-9126-3b612741c26d",
        "37733a30-f84c-416d-977f-ac3a5b2a68c4",
        "3833df3d-dc81-4467-bdd9-16a7d99f7edb",
        "3893e3e7-24b7-45f3-90e8-e9ca6114e83f",
        "3aa0bbf3-e880-4a78-8a97-de9834b2eb18",
        "3b4966d8-f38a-46cc-ba12-4f059b753241",
        "3b4f130b-0287-42ac-975d-ecd4f7377b0f",
        "3b9ff2e4-dfec-4022-8994-1a0c4db7227a",
        "3ca29402-0a0d-4f08-8f6c-5b57672809f6",
        "3d635e50-7f62-444d-b302-8131aae56104",
        "3ea4837f-104a-44d3-a006-4bc586c0f829",
        "3f38d579-66fa-4126-b269-3f4e1ee42b39",
        "407d6e31-ffbd-4e84-97ab-54b2ff396823",
        "41bf78e2-4791-4d2e-9459-22f401074964",
        "42ded059-c890-42e1-adb8-4b339d0879b5",
        "44e31af4-7a40-4124-8dc4-d14b12dd66c7",
        "45a86c85-0bd3-4f3f-bcdc-2b27cd539288",
        "472c78e7-d814-4f61-9128-336d13fb9a0f",
        "4ac3f410-c8fd-4cf6-a065-405683c27499",
        "4b0f6e44-14a3-4cd9-abab-e279af6f45c6",
        "4d7d8db6-d135-4db5-a718-11582822ff1a",
        "5034a63c-298d-4d16-bfd7-7510c09b78d3",
        "50846a75-cde7-44d0-878d-bedc39726f75",
        "5134cb88-3e01-4861-a585-395201e803c9",
        "51ebb070-7d34-40b3-8520-55d2e318438a",
        "53b45c45-9ca0-4ec9-b54d-90ffdfec0c38",
        "53db5d5c-fe90-47d1-8252-c05566bf03e7",
        "55675812-1eac-44e0-a57d-30a5a9ae083e",
        "56b8d410-341c-4e90-ad7c-3e8bf1cbb0b6",
        "583dc6e7-fabe-4d2a-b633-4ad5aea67f8c",
        "58ec107b-68e2-4742-a497-d8d4af1523ee",
        "5aeb56d6-af68-4189-a8ec-1f8d8a0c9e02",
        "5af23884-b630-496c-b04e-b9db94250307",
        "5b1ab5af-701b-4717-9c43-98ab90a89325",
        "5cd4f692-3b13-4728-9df3-debc682e42dd",
        "61519ad0-2aea-4250-9a82-4dcdb93a569c",
        "622bf725-2471-4392-8f82-fcc9115a3745",
        "63f5e202-6b13-491a-bdfc-9f13b7e4c036",
        "65df4c37-b8a5-443d-b488-4b92ecc891d0",
        "66a5cdf8-3b0d-4d85-bdcc-68ae69205206",
        "6770c699-3b3c-4333-b70a-b7e7d839e38b",
        "67aeb5af-38f7-4949-b59f-608896d20077",
        "682564cb-62be-40ff-b5f6-0f53924d37e3",
        "6b6cee83-d69e-4a7a-b959-ee2de109d907",
        "6b74bbc2-128a-4206-82e1-eaec79579c51",
        "7260c523-b2d5-4bdc-a831-96b4815dca00",
        "77131a79-5ccb-4df8-afa4-14c71d4048b8",
        "7b8358f3-c96a-4a17-87ab-9414866e18db",
        "805f3a7b-a197-4834-a4e4-a56da3dde6b1",
        "814e0cc2-ed93-4e72-af35-40a41cef21cd",
        "83e9c0b8-b6a7-4c19-8deb-2bca29b93139",
        "83ff3fe4-b456-47ac-a5fa-1d152f900235",
        "84f93536-ce17-41eb-9241-10d4b91fd97f",
        "862ab494-446d-43f6-b752-af51e938547e",
        "879233b3-262c-4b19-a344-f4d595d08d26",
        "8861a2f4-80be-4c62-8c1e-252efad26ccc",
        "8a3533aa-d6d4-450c-8232-79e4851b6e11",
        "8aa1bd02-ee43-4e9d-b7f7-7ddc66b607f9",
        "8acefded-803b-4cf3-9829-18bd5508f907",
        "8b73cf24-6579-4a9f-b7db-62317feb4d58",
        "8c2f443e-b329-42f2-a1cf-31c150be47e0",
        "8c9f9374-4398-4b3c-92b7-2c66dc4fcbd5",
        "8cee7fce-00d0-45f1-81fe-b24bb7176679",
        "8d458ff0-0ed3-446d-bd87-cd23dae0a05f",
        "8fcc208a-30e9-49dc-b9de-51e6819a50b7",
        "900b0dc1-b236-4364-bbab-0371af4eef84",
        "9187dcd7-7b5c-4633-ae73-af8200f1dfba",
        "9324c6d7-3fdc-4614-8199-0c102f1b67c0",
        "942445c5-eb52-4128-98ca-0fbb5dd6aca4",
        "948b8d17-35af-40cc-aa50-70a12b56e433",
        "94eb0755-56a3-4235-b759-8dc2ced70875",
        "95085684-88ec-4d2a-8eba-a38268018193",
        "9580deeb-f28e-42e9-a877-1a6e1cea3dff",
        "95a070a7-086b-4f3d-a5ba-0a877f7fabf7",
        "9658fd5d-f8bf-4fd3-ba8d-6e40776039de",
        "985b1b37-40be-49aa-9df1-31aff97dfb1a",
        "9a917ae9-8e47-4eaf-aaff-e82bfc9f7d6b",
        "9b7483ba-9c6f-4c67-bf48-549384ac66c7",
        "9d7b2444-3411-4252-aa07-fdd2b1aa60ad",
        "9e4aeae9-8729-4b0f-9e84-5c1f4eeacc74",
        "9f5be69d-028c-4020-8140-439e652e2343",
        "a173be18-19f9-4485-8143-0ac56b45b1eb",
        "a262930d-17eb-4053-840c-a8f9fb035a00",
        "a54d9ef4-a46a-418b-b6cc-f10b49a946ac",
        "a60bbc2d-4d45-4896-b533-15c2cd54cf60",
        "a6c16f12-0987-4690-87fe-336710f96398",
        "a73e8ebf-1c4d-4a05-acf7-8b834b1f3b85",
        "a8049cb2-a13b-48f7-abe3-cf42f60368aa",
        "aa45a425-2c17-46de-8bc8-cbe080158c80",
        "ab6ffd82-1f4f-448a-a4a2-b0d8e861fc81",
        "ac9b4778-0bb5-4384-b4ed-3a5738ba99a4",
        "ae4e4f1a-098a-4906-a2c9-78a68f4cdd14",
        "af8c78cf-7d82-44f0-a277-abe61896b015",
        "b00fe5cd-a973-4f16-93e1-b15b5ea1d7e2",
        "b0d888c3-f3d3-4e5c-9dce-bbcb4c15ac58",
        "b1ea5c54-3b28-46c7-bc3a-8dab91d73d6a",
        "b3aed23e-d863-4a87-8c38-309cf958f930",
        "b43f7bd1-6472-4767-a175-d943f595e9ff",
        "b8bbd940-ba64-474c-bfea-37e578fc2133",
        "ba79fc54-e083-40e7-ae36-fae72980f2d0",
        "bbc41a1e-4bbe-4417-a40c-64635cc552e6",
        "bedb175f-d70a-41c0-91f4-573f7b22f675",
        "c1f31960-dee7-45ea-ac13-a4fea1c9235c",
        "c271feb1-dac7-4940-9ef0-cf4de3a6b1db",
        "c2b6746a-5780-47c2-bee3-db18bfccd409",
        "c4c0c196-c6a7-41fd-bdf1-99002be6e841",
        "c5677eca-f00e-45af-ab0c-7388438c85e3",
        "c696b0e8-299f-4270-9218-25f973bc64b4",
        "c73670a9-16ca-43bb-b7b9-110558e31798",
        "c7c1cc61-27fa-4bfd-bdef-f3e04c527987",
        "c81cafe4-2589-4207-b8ba-6abbc2e311c2",
        "c93a811e-1f47-43b6-aef9-c09338e43947",
        "c9b351d9-124b-4119-8be8-cb7c0f7e7994",
        "c9b4e8d8-34f7-474b-9233-db1d945d8fa0",
        "cd6f425b-92fe-4bdf-b277-53a558fc7c27",
        "d1cf1b0a-c3b8-45cc-93bc-792f820f3e92",
        "d2bb3e13-9a98-45e0-8b3d-71b3b89e598f",
        "d3268d64-57ae-4060-937a-7980cdc5ffcd",
        "d36edb42-c422-41b4-b82f-594d8f3850be",
        "d3af01d0-31b7-4fd9-af11-83b7bd594b12",
        "d3d33d63-101d-44fd-b6b9-4616a803225d",
        "d41622a7-9fe7-4f6c-9e19-62b431c9e36b",
        "d4691f19-57be-44c4-afc2-5b5f82ec27b5",
        "d72be89a-68d7-4ca8-bb97-1cb9be776001",
        "d7eb1a8a-344d-463c-a660-9b4ef56a8b63",
        "d83ac187-97cd-4ee0-a35f-5a1ffe6c7885",
        "d9aa9a5c-e496-4da7-ab67-adabc1f37c72",
        "da66d9bb-0d61-4a3a-afbd-792ac93ebddf",
        "dc9199c5-f484-4da4-8bf4-de8319628d35",
        "dd0b3f2a-d12e-4d58-ae4f-17b3ce7f95a4",
        "dd13f25f-77a0-4a2c-83af-bb187b79a389",
        "dde1ede1-50cc-4ce5-9f4a-fdd3a8f1e3dc",
        "df11d649-9c87-4802-b602-ec97c3e868b1",
        "df2b2506-6a64-4f94-8b7c-171a373387a3",
        "e0830d00-bcac-4084-893a-3b61de64ecf6",
        "e0f51af7-56c6-4c0e-9f0a-6c34ec0ff98a",
        "e118d556-2088-4cc2-b49a-82aad5974167",
        "e2ee5efe-d9cf-482c-8c73-a895185e3524",
        "e3c665c2-f7b6-4be6-99a4-197af6f01bdf",
        "e3d59499-2840-4a2c-81fc-19f8f12f0a65",
        "e4e6552b-35bf-4df8-8302-c9cc304d34f1",
        "e6b87c27-0b3b-4851-8fb6-7c366a6af508",
        "ea8ee8f7-b950-4cbb-95f7-c87da15d3e82",
        "eb351488-1a8b-4521-831d-9d9d0d1dc208",
        "ecfb8b5b-9e00-4faf-98f6-1e5d42140e3b",
        "ee641ee3-b4b9-4a2f-8394-ebda81037fe4",
        "f16d63b1-c63a-42e0-80c9-aa18ec513da3",
        "f2a3349e-b223-4d54-b7a1-c09664d7d91b",
        "f44b2de7-97bc-4539-8515-f110feec479d",
        "f486f8c2-ea21-4909-9af3-9cd2ff1638bc",
        "f4aafd07-9711-4850-a6b6-63efa2fe25c6",
        "f5abb4f1-ad31-4964-988c-14769501a8f7",
        "f611477f-7b2e-4f36-81da-c6cdee27d7a1",
        "f744e739-4b60-4f0e-95a5-42c9f2042f1a",
        "f77f3c8b-49e6-44fe-92a1-c0b07bbea9e9",
        "f9351b3e-e4de-4d94-b3bc-98786d35906b",
        "fbd7bc95-9f42-4c2c-94f4-27fd78a7273c",
        "fe84e1e8-87b7-463b-99e4-2e492576ccbd",
        "febb3cef-56cc-4f40-b7c2-7c2663b0dc33"])


    elif(num=="300"):
        orig_ids=np.array(['15d069af-7a36-4c33-8693-cc814845b9c3', '65df4c37-b8a5-443d-b488-4b92ecc891d0',
      'b404aea2-e1a5-4dba-ad61-8132a3841dcf', 'cb8a3787-1e23-4265-bf3d-264bfe6b25e8',
      '30fe7844-15f8-4c6b-b1d6-a642f43616b4', '3d635e50-7f62-444d-b302-8131aae56104',
      '0ce0209c-ee94-4ef0-9bea-5caad065b103', '4d7d8db6-d135-4db5-a718-11582822ff1a',
      '301734c8-1f4a-4dc7-ad19-7f24e0c4d4f6', 'd7eb1a8a-344d-463c-a660-9b4ef56a8b63',
      '0d843c9a-5c9d-4325-bae6-e44e0902e84a', 'b2a7d259-143a-4db7-9406-5a95e020d6b7',
      '157022dd-d527-460d-8abc-dca2f3310394', '948b8d17-35af-40cc-aa50-70a12b56e433',
      '87d70bed-3ed0-455c-a144-9fd955229125', '479eea59-8ad8-46aa-9456-29ab1b8f2cb2',
      '022e4ff8-e1af-43dc-b747-862ac83518d2', '9e4aeae9-8729-4b0f-9e84-5c1f4eeacc74',
      'b4b75916-a561-41e4-b3b9-5dfc2859028d', '46f29cfd-cb46-4ec7-b983-b483268bbe21',
      '407d6e31-ffbd-4e84-97ab-54b2ff396823', '13555453-55fc-4b13-8b32-b49dfcc9a8f0',
      '16db9568-bd11-494f-b712-28a2266ea3d0', '9a917ae9-8e47-4eaf-aaff-e82bfc9f7d6b',
      '3b4966d8-f38a-46cc-ba12-4f059b753241', 'de5c2828-29c5-4545-8367-502bb0ed1004',
      'c7c1cc61-27fa-4bfd-bdef-f3e04c527987', '8c2f443e-b329-42f2-a1cf-31c150be47e0',
      '6b74bbc2-128a-4206-82e1-eaec79579c51', '53400e38-644f-40bf-9968-b11f6682e7e5',
      '50846a75-cde7-44d0-878d-bedc39726f75', 'dbd205d8-339f-4432-aafc-a983ee553245',
      'f29a8628-ed1d-445a-9316-b4eb62fed4d7', '360dda05-d7a7-4ec3-8988-256bb6381cda',
      'dd13f25f-77a0-4a2c-83af-bb187b79a389', '24809752-2bcd-424f-aac1-5d909026b0c4',
      '3b6fda64-bb3f-4a77-bec0-7ba034d4540e', '61519ad0-2aea-4250-9a82-4dcdb93a569c',
      'd3af01d0-31b7-4fd9-af11-83b7bd594b12', 'efb8ddb1-b9a4-4833-a876-c8adc53850ca',
      'df11d649-9c87-4802-b602-ec97c3e868b1', '8aa1bd02-ee43-4e9d-b7f7-7ddc66b607f9',
      'c576bedb-b0f7-4458-a1e0-24071d6e1d36', 'c018eb43-ab13-401d-8743-53cdc93c9e65',
      '8d458ff0-0ed3-446d-bd87-cd23dae0a05f', 'ecfb8b5b-9e00-4faf-98f6-1e5d42140e3b',
      'ee179373-c30e-4ef2-9f32-75419e005cf4', '4203df44-c6be-49a5-93b8-a9047d438fe4',
      '351fbcd3-c1ec-416c-bed7-195fe5d1f41d', '63684250-6bd1-41e6-b270-ebfc5379a271',
      'f2aa92b3-ac1a-4afe-9dda-977c07ee8ee2', 'b0d888c3-f3d3-4e5c-9dce-bbcb4c15ac58',
      '862ab494-446d-43f6-b752-af51e938547e', '0990887a-6163-4c80-9c9e-468ea2598202',
      'd2bb3e13-9a98-45e0-8b3d-71b3b89e598f', '67aeb5af-38f7-4949-b59f-608896d20077',
      '24e24816-b56b-4c16-9e6d-0ae8afc8650f', 'efa1c292-bb43-420f-a229-1d1da9901b13',
      'c271feb1-dac7-4940-9ef0-cf4de3a6b1db', 'd609008d-6efb-4cd0-ab84-6c0de55198db',
      '40556c24-113c-49f9-9371-34931d4fba2d', 'c7091bfb-1ab2-4fbe-9d09-ae801a2585a2',
      '0457f007-211c-4dc3-844e-47d724fece51', 'a5e9c6b1-40b2-4763-862b-61811f58f2cb',
      'e118d556-2088-4cc2-b49a-82aad5974167', 'dd0b3f2a-d12e-4d58-ae4f-17b3ce7f95a4',
      'b43f7bd1-6472-4767-a175-d943f595e9ff', 'd3268d64-57ae-4060-937a-7980cdc5ffcd',
      '95085684-88ec-4d2a-8eba-a38268018193', 'ec533293-9f82-42ef-9ee1-a7b8c4710dc3',
      '41bf78e2-4791-4d2e-9459-22f401074964', 'cf9ee6f5-94ff-4097-a751-b8ed7c2a2e4b',
      '290e568c-16e2-4656-8e04-7298f5884c48', '9c0a19ac-3240-4e0e-9e1d-b49b74681ce0',
      '5134cb88-3e01-4861-a585-395201e803c9', '9324c6d7-3fdc-4614-8199-0c102f1b67c0',
      '6cd16a05-3496-4c8e-a915-b44ac950e241', 'f2a3349e-b223-4d54-b7a1-c09664d7d91b',
      'a60bbc2d-4d45-4896-b533-15c2cd54cf60', 'ed1698e5-4104-4b28-a0e7-71d710f2efb7',
      '985b1b37-40be-49aa-9df1-31aff97dfb1a', 'b2bc6b0a-b4f0-4ebf-829a-a2f28addae50',
      'd1cf1b0a-c3b8-45cc-93bc-792f820f3e92', '34192284-e4f1-4420-ab35-0eb02ce2e8d1',
      'a54d9ef4-a46a-418b-b6cc-f10b49a946ac', '56b4bef0-2bbb-4e2a-b434-36bfdb2f0f7e',
      'a6c16f12-0987-4690-87fe-336710f96398', '45a86c85-0bd3-4f3f-bcdc-2b27cd539288',
      'f16d63b1-c63a-42e0-80c9-aa18ec513da3', '583dc6e7-fabe-4d2a-b633-4ad5aea67f8c',
      'c87dcc5b-2846-4eca-b6c2-ecc7ea58bbc7', 'c81cafe4-2589-4207-b8ba-6abbc2e311c2',
      'fbd7bc95-9f42-4c2c-94f4-27fd78a7273c', '6bf8a6da-a8c3-45c4-8aa3-75649cd1772d',
      '110a98b6-ef01-44b2-a19c-9ab8db1dd6fa', 'f744e739-4b60-4f0e-95a5-42c9f2042f1a',
      'af8c78cf-7d82-44f0-a277-abe61896b015', 'a6006dcb-2c1a-4e66-8db0-d9deb6c93752',
      '784af6b8-2218-4e01-a066-b66eb2832d9d', '0c824653-a13b-4a4e-b907-660f1d8f8981',
      '5dc54855-a59b-47e3-867b-46eb1c789b23', '326a6c55-c963-42c2-bb8a-2591993aaaa2',
      'df2b2506-6a64-4f94-8b7c-171a373387a3', '20bae182-cdd9-4e73-a584-f337bbfa55df',
      '1b46a6a9-0e7d-4740-b101-b5bc28154f2c', '2d954abb-e11a-4f25-8f12-19a210c8d8e1',
      '07a34e07-8ee4-434b-8c09-6cb197451464', 'fe84e1e8-87b7-463b-99e4-2e492576ccbd',
      '4ac3f410-c8fd-4cf6-a065-405683c27499', '682564cb-62be-40ff-b5f6-0f53924d37e3',
      'febb3cef-56cc-4f40-b7c2-7c2663b0dc33', 'f1a772e9-bf5f-4bc9-96ea-7a45f38c8c41',
      'b8e437d3-d8db-48c3-91b5-ebd6ab29662f', '8cee7fce-00d0-45f1-81fe-b24bb7176679',
      'bbc41a1e-4bbe-4417-a40c-64635cc552e6', '58ec107b-68e2-4742-a497-d8d4af1523ee',
      'c1f31960-dee7-45ea-ac13-a4fea1c9235c', 'c2b6746a-5780-47c2-bee3-db18bfccd409',
      '8c9f9374-4398-4b3c-92b7-2c66dc4fcbd5', '757093f1-191a-4c71-9c28-92d476959e27',
      'ab6ffd82-1f4f-448a-a4a2-b0d8e861fc81', '95a070a7-086b-4f3d-a5ba-0a877f7fabf7',
      '30e340a8-776f-40d1-9d8c-d70d38bf0a5f', '5accb319-08ca-402a-9778-ad1f0608401b',
      '24d285d2-c1c9-4a9b-ab57-68fe292cf472', 'aa45a425-2c17-46de-8bc8-cbe080158c80',
      '3833df3d-dc81-4467-bdd9-16a7d99f7edb', 'b4ff7130-3055-4ed1-a878-8dfaca7191ac',
      'be4297a8-d763-42e2-a2cb-cab38f64cfe3', 'bedb175f-d70a-41c0-91f4-573f7b22f675',
      'e2ee5efe-d9cf-482c-8c73-a895185e3524', 'f9351b3e-e4de-4d94-b3bc-98786d35906b',
      'a7e8df51-db85-42c3-9ebe-207dcd736793', 'e3d59499-2840-4a2c-81fc-19f8f12f0a65',
      'f80e8b55-30d6-47fd-8f12-851016b0b0fa', '75f01e07-425c-46ac-bcfc-aba98f2b02be',
      'b18bc631-88a3-4c7a-89aa-14b3350aa48d', 'e6b87c27-0b3b-4851-8fb6-7c366a6af508',
      '66c74335-a160-4d1c-8165-8e49a3c03666', '8861a2f4-80be-4c62-8c1e-252efad26ccc',
      '6b6cee83-d69e-4a7a-b959-ee2de109d907', 'c1d277ee-1f7a-4dd1-b50a-40a8540d381b',
      '41d28e85-3a78-40a8-ad6f-aa22aa02795c', '3c77a951-188c-4c57-93b0-b7a194d4cc19',
      '059a9d92-4d36-40cb-84cc-408f9210821b', '53b45c45-9ca0-4ec9-b54d-90ffdfec0c38',
      '80134e57-9f6a-42e6-bd13-01875b5af715', 'a8049cb2-a13b-48f7-abe3-cf42f60368aa',
      'c9b4e8d8-34f7-474b-9233-db1d945d8fa0', '5aeb56d6-af68-4189-a8ec-1f8d8a0c9e02',
      'a73e8ebf-1c4d-4a05-acf7-8b834b1f3b85', '84f93536-ce17-41eb-9241-10d4b91fd97f',
      'b6a84ed4-e9d1-4df9-a293-938666fb512b', 'ac3c89bc-11bb-447a-b226-5a4a935e9653',
      '055bed5b-60ec-43e6-9110-137f2a36d65b', 'f44b2de7-97bc-4539-8515-f110feec479d',
      '166b9e2f-08a1-425f-9617-875d6ad3f834', '323cc961-5f66-4273-9d46-6097bc692269',
      '260f551d-e3c1-475e-b242-f17aad20ba2c', 'ee641ee3-b4b9-4a2f-8394-ebda81037fe4',
      '397c6457-0954-4cd2-995c-2fbeb6c72097', 'f4aafd07-9711-4850-a6b6-63efa2fe25c6',
      '74fd99d6-7bfc-40ae-b90f-1639c79294d3', '8fcc208a-30e9-49dc-b9de-51e6819a50b7',
      '6f157d6b-d974-4ec4-9c93-7e29bdaf0108', '2f256f48-8336-4846-9362-7349ae648dc5',
      '1c47637b-d4a1-432f-9810-b6e2daea0a43', '26d5159f-ca0f-4537-a97d-8ddda3c95f8a',
      '7b8358f3-c96a-4a17-87ab-9414866e18db', '089cc9f8-044f-40d6-aca9-af76bf35e4d2',
      '3b4f130b-0287-42ac-975d-ecd4f7377b0f', '4b0f6e44-14a3-4cd9-abab-e279af6f45c6',
      '58c394c6-19ba-48ad-9bc2-bc3efb4053b5', 'bb23dbbc-b679-4849-b1d8-63279cad50e2',
      '82a921b9-361a-4fd5-8db7-98961fdbf25a', 'feaad88c-c9f0-405a-a32f-4c9adfc0be6b',
      'b61ebedd-272f-4276-9eee-63bb9e1a4ad6', 'b8bbd940-ba64-474c-bfea-37e578fc2133',
      '3f38d579-66fa-4126-b269-3f4e1ee42b39', 'd83ac187-97cd-4ee0-a35f-5a1ffe6c7885',
      '35fff957-2ad7-40e6-b57a-30235da5c9a9', '57dc658f-bb87-452c-94fc-9bd6a72b0bcc',
      '5cd4f692-3b13-4728-9df3-debc682e42dd', '34ce3561-6cd1-43e8-8a5b-c1d468863166',
      'b7e851d4-1cbc-4bd1-a040-d21e68182d49', '2b6a9abb-9a75-4d1b-8cab-5dd862626349',
      '83e9c0b8-b6a7-4c19-8deb-2bca29b93139', '176dfa9b-725b-4813-9219-005041320db4',
      'da6b7c06-9bb4-45d0-b8c7-e7dadf77285c', '5af23884-b630-496c-b04e-b9db94250307',
      'd0966bbb-d634-419e-ba57-8cf6de4e98ee', 'ded067ce-6c9a-4b7d-93ca-b848b6977d45',
      'ea8ee8f7-b950-4cbb-95f7-c87da15d3e82', '8b73cf24-6579-4a9f-b7db-62317feb4d58',
      '02e82ef0-acb8-4366-8c83-4c3f2f69f7ea', 'ae2fae6b-0365-4a35-baee-144f2642cac4',
      'd3c0b2d1-4ab7-48e0-8ac4-fc6a104b03d3', '814e0cc2-ed93-4e72-af35-40a41cef21cd',
      '3d1e1b85-e0de-46a9-853f-4c3aa2e56271', 'c4c0c196-c6a7-41fd-bdf1-99002be6e841',
      '3893e3e7-24b7-45f3-90e8-e9ca6114e83f', '21faaa13-7447-420f-aa5f-7a2e128d06aa',
      '8a3533aa-d6d4-450c-8232-79e4851b6e11', '84b129ce-028e-4709-b410-ba9f18d29902',
      '6770c699-3b3c-4333-b70a-b7e7d839e38b', '5eaca290-9215-4b6d-907b-756aa17adbd0',
      '1fa707a3-63c4-488b-9558-f4d827fc9b12', 'cb29da71-7fe9-4270-83d1-ba58b8034fc6',
      'e4e6552b-35bf-4df8-8302-c9cc304d34f1', '1a2eb8d4-077e-4c0b-88d9-574872b45e4c',
      '34223626-fab9-48f5-82bd-0b0037745994', 'e3c665c2-f7b6-4be6-99a4-197af6f01bdf',
      '9d18a57c-ae79-434c-81f9-b20e9f0e21d3', 'da66d9bb-0d61-4a3a-afbd-792ac93ebddf',
      '00ab666c-afb8-476e-9872-6472b4e66b68', 'fecd1ca1-4240-42ab-ab94-bfb993a1da11',
      '81146c4b-a952-433f-98ed-1a125896d36f', '7fed6abf-3b16-41e3-8e9d-ec61382bce7e',
      '8013239e-0d75-4640-9ec1-2b9d856956a0', '63f5e202-6b13-491a-bdfc-9f13b7e4c036',
      'eb351488-1a8b-4521-831d-9d9d0d1dc208', 'b64eaa81-425f-446a-a521-9fdd9429f77b',
      'c6a176bc-6b30-4066-a7a4-b305ea1df716', '17b07883-4959-4037-9b80-dde9a06b80ae',
      'e0830d00-bcac-4084-893a-3b61de64ecf6', '82d8787f-ed86-41a0-b342-451c6064dc59',
      '879233b3-262c-4b19-a344-f4d595d08d26', 'c9b351d9-124b-4119-8be8-cb7c0f7e7994',
      '2bc2ade2-b154-4412-878c-98466a00ba7b', '75f6a3c6-ef3d-43e2-810c-560efaf92592',
      '2946b01f-4cda-4f69-9b3f-24aed26c4d15', '53db5d5c-fe90-47d1-8252-c05566bf03e7',
      '51ebb070-7d34-40b3-8520-55d2e318438a', '42ded059-c890-42e1-adb8-4b339d0879b5',
      '9580deeb-f28e-42e9-a877-1a6e1cea3dff', 'b00fe5cd-a973-4f16-93e1-b15b5ea1d7e2',
      'e7500981-9b13-4238-a855-52b91ed6244d', '0bee012c-efcb-4512-8c29-13d0a935cb48',
      '67556c68-b9a4-49e5-a3d6-7dc3c7b3e0ec', '3ca3dbf5-2390-409e-bd2c-c9f23a255e75',
      '66a5cdf8-3b0d-4d85-bdcc-68ae69205206', '136f8891-af6f-49c1-a69a-b4acd7116a3c',
      'f611477f-7b2e-4f36-81da-c6cdee27d7a1', '2b4ae7c4-ee25-4588-a0cd-bffb64a07f7f',
      'd72be89a-68d7-4ca8-bb97-1cb9be776001', '05180b17-aae0-4920-bfd0-96b062017f7d',
      '8acefded-803b-4cf3-9829-18bd5508f907', 'e0f51af7-56c6-4c0e-9f0a-6c34ec0ff98a',
      'ba79fc54-e083-40e7-ae36-fae72980f2d0', '28fba926-ef44-4874-a209-ac6680441822',
      '038aafca-cc30-47c6-9cbe-5c2cb52d8f04', 'ac48132f-2c65-4762-bb64-ed8f733a540d',
      'cdd98fba-4d2c-45d1-94b3-4b6b6077b58e', '9b7483ba-9c6f-4c67-bf48-549384ac66c7',
      'f486f8c2-ea21-4909-9af3-9cd2ff1638bc', 'ddf11c46-b071-4f00-82bc-c0ee74d78aa0',
      'd41622a7-9fe7-4f6c-9e19-62b431c9e36b', 'f2d36ddc-d20e-46c7-974c-fb7fc273db31',
      '24ea7a3f-4b4d-48a0-a256-6de2e8298d25', 'd4691f19-57be-44c4-afc2-5b5f82ec27b5',
      '3ff76631-5492-4693-8b20-73b5a6c8864d', 'a262930d-17eb-4053-840c-a8f9fb035a00',
      '36f6e239-816a-4508-9126-3b612741c26d', '35daf881-da7c-4779-a8e9-20a3985094e2',
      'd197d735-bb0d-4df5-9566-2325de86e98b', '89e13844-fa61-46bd-b99f-46423122de5a',
      'd3d33d63-101d-44fd-b6b9-4616a803225d', 'b1ea5c54-3b28-46c7-bc3a-8dab91d73d6a',
      '9d7b2444-3411-4252-aa07-fdd2b1aa60ad', 'b3aed23e-d863-4a87-8c38-309cf958f930',
      '7260c523-b2d5-4bdc-a831-96b4815dca00', '3aa0bbf3-e880-4a78-8a97-de9834b2eb18',
      'd36edb42-c422-41b4-b82f-594d8f3850be', '42bbf143-b184-4cf7-9afa-289468d9e36b',
      '3ea4837f-104a-44d3-a006-4bc586c0f829', 'dde1ede1-50cc-4ce5-9f4a-fdd3a8f1e3dc',
      '20a07a1c-ee7e-4958-a2c8-0db6e4fe0ce9', '77131a79-5ccb-4df8-afa4-14c71d4048b8',
      '6fb3efa6-3ede-43c5-bdea-0b64907a68de', 'c93a811e-1f47-43b6-aef9-c09338e43947',
      'c696b0e8-299f-4270-9218-25f973bc64b4', 'dc9199c5-f484-4da4-8bf4-de8319628d35',
      '92e8540c-3290-407d-9114-8458b9bd211a', '9f5be69d-028c-4020-8140-439e652e2343',
      'de4021b2-547f-4a26-a3ac-daba9c71edd0', '26eeb04e-ab64-45b2-bb12-7aff7a779f9a',
      '912427ec-befd-4a0f-b2ad-36af77198e06', '472c78e7-d814-4f61-9128-336d13fb9a0f',
      '9658fd5d-f8bf-4fd3-ba8d-6e40776039de', '0a11d9aa-1e9c-4f5e-94cf-faa6e796a855',
      '61d1a237-d70f-49b0-89ba-cea4d2526832', '83ff3fe4-b456-47ac-a5fa-1d152f900235',
      'ae4e4f1a-098a-4906-a2c9-78a68f4cdd14', '1bdf0668-a632-4290-ad94-c6269f9e924a',
      '622bf725-2471-4392-8f82-fcc9115a3745', '9187dcd7-7b5c-4633-ae73-af8200f1dfba',
      '2fb5e890-afaf-428a-8e28-a7c70bf8bdf1', '2ac8cb80-76be-4903-bcae-9f2c9b8b992f',
      'c2dc579e-47c5-4498-857a-2f765406b8a5', 'af533808-e79f-4f2b-994b-969dcf0e4f5d',
      'c6574c0d-ceca-4584-af55-d8e7e282ed8d', '942445c5-eb52-4128-98ca-0fbb5dd6aca4',
      'a173be18-19f9-4485-8143-0ac56b45b1eb', '077ff26a-4f7b-48d2-833e-1b9d31cb7615',
      '03ec3750-641a-4039-8b5d-74b485bde1ea', 'd9aa9a5c-e496-4da7-ab67-adabc1f37c72',
      '5034a63c-298d-4d16-bfd7-7510c09b78d3', 'f5abb4f1-ad31-4964-988c-14769501a8f7'])

    else:
        print("Error: Requested subject set %s is not defined"%(num))
        exit()

    np.random.seed(0)
    N     = len(orig_ids)
    perm  = np.random.permutation(N).astype(np.int32)
    split = int(np.floor(pct*N))

    np.random.seed(42)
    np.random.seed(10)

    train_ids = orig_ids[perm[:split]]
    test_ids  = orig_ids[perm[split:]]
    
    print("Samples in Train: %d. Samples in test: %d"%(np.sum(counts[perm[:split]]), np.sum(counts[perm[split:]])))
    print("Subjects in Train: %d. Subjects in test: %d"%(len(train_ids), len(test_ids)))

    return(train_ids, test_ids)

def get_ids(set="300"):

    """
    Returns a list of user IDs according to the subject set being used.

    Args:
        set (str): Which ID set to select.

    Returns:
        orig_ids (numpy.Array): An array of user IDs.
    """

    if(set=="200"):
        orig_ids=np.array(["00ab666c-afb8-476e-9872-6472b4e66b68",
        "02e82ef0-acb8-4366-8c83-4c3f2f69f7ea",
        "03996723-2411-4167-b14b-eb11dfc33124",
        "03c26210-7c9f-4bf2-b1c2-59d0bd64ffac",
        "03ec3750-641a-4039-8b5d-74b485bde1ea",
        "055bed5b-60ec-43e6-9110-137f2a36d65b",
        "059a9d92-4d36-40cb-84cc-408f9210821b",
        "07a34e07-8ee4-434b-8c09-6cb197451464",
        "0990887a-6163-4c80-9c9e-468ea2598202",
        "0a11d9aa-1e9c-4f5e-94cf-faa6e796a855",
        "0c824653-a13b-4a4e-b907-660f1d8f8981",
        "110a98b6-ef01-44b2-a19c-9ab8db1dd6fa",
        "13555453-55fc-4b13-8b32-b49dfcc9a8f0",
        "135c9c3b-a5cf-47a4-9fcf-4fc418c5eb96",
        "136f8891-af6f-49c1-a69a-b4acd7116a3c",
        "15d069af-7a36-4c33-8693-cc814845b9c3",
        "166b9e2f-08a1-425f-9617-875d6ad3f834",
        "16db9568-bd11-494f-b712-28a2266ea3d0",
        "176dfa9b-725b-4813-9219-005041320db4",
        "17b07883-4959-4037-9b80-dde9a06b80ae",
        "1b524925-07d8-42ea-8876-ada7298369ec",
        "1bdf0668-a632-4290-ad94-c6269f9e924a",
        "1c47637b-d4a1-432f-9810-b6e2daea0a43",
        "20a07a1c-ee7e-4958-a2c8-0db6e4fe0ce9",
        "24809752-2bcd-424f-aac1-5d909026b0c4",
        "24e24816-b56b-4c16-9e6d-0ae8afc8650f",
        "260f551d-e3c1-475e-b242-f17aad20ba2c",
        "26d5159f-ca0f-4537-a97d-8ddda3c95f8a",
        "26eeb04e-ab64-45b2-bb12-7aff7a779f9a",
        "2946b01f-4cda-4f69-9b3f-24aed26c4d15",
        "2b4ae7c4-ee25-4588-a0cd-bffb64a07f7f",
        "2bc2ade2-b154-4412-878c-98466a00ba7b",
        "2d954abb-e11a-4f25-8f12-19a210c8d8e1",
        "2f256f48-8336-4846-9362-7349ae648dc5",
        "323cc961-5f66-4273-9d46-6097bc692269",
        "326a6c55-c963-42c2-bb8a-2591993aaaa2",
        "34192284-e4f1-4420-ab35-0eb02ce2e8d1",
        "34223626-fab9-48f5-82bd-0b0037745994",
        "342d9d87-787d-4836-9f61-4d59ed9f3289",
        "34521ae7-012a-400f-8794-3d76ff4e70ab",
        "35daf881-da7c-4779-a8e9-20a3985094e2",
        "35fff957-2ad7-40e6-b57a-30235da5c9a9",
        "360dda05-d7a7-4ec3-8988-256bb6381cda",
        "36caadf6-9bd5-4bac-9f13-75d2f439b4de",
        "36f6e239-816a-4508-9126-3b612741c26d",
        "37733a30-f84c-416d-977f-ac3a5b2a68c4",
        "3833df3d-dc81-4467-bdd9-16a7d99f7edb",
        "3893e3e7-24b7-45f3-90e8-e9ca6114e83f",
        "3aa0bbf3-e880-4a78-8a97-de9834b2eb18",
        "3b4966d8-f38a-46cc-ba12-4f059b753241",
        "3b4f130b-0287-42ac-975d-ecd4f7377b0f",
        "3b9ff2e4-dfec-4022-8994-1a0c4db7227a",
        "3ca29402-0a0d-4f08-8f6c-5b57672809f6",
        "3d635e50-7f62-444d-b302-8131aae56104",
        "3ea4837f-104a-44d3-a006-4bc586c0f829",
        "3f38d579-66fa-4126-b269-3f4e1ee42b39",
        "407d6e31-ffbd-4e84-97ab-54b2ff396823",
        "41bf78e2-4791-4d2e-9459-22f401074964",
        "42ded059-c890-42e1-adb8-4b339d0879b5",
        "44e31af4-7a40-4124-8dc4-d14b12dd66c7",
        "45a86c85-0bd3-4f3f-bcdc-2b27cd539288",
        "472c78e7-d814-4f61-9128-336d13fb9a0f",
        "4ac3f410-c8fd-4cf6-a065-405683c27499",
        "4b0f6e44-14a3-4cd9-abab-e279af6f45c6",
        "4d7d8db6-d135-4db5-a718-11582822ff1a",
        "5034a63c-298d-4d16-bfd7-7510c09b78d3",
        "50846a75-cde7-44d0-878d-bedc39726f75",
        "5134cb88-3e01-4861-a585-395201e803c9",
        "51ebb070-7d34-40b3-8520-55d2e318438a",
        "53b45c45-9ca0-4ec9-b54d-90ffdfec0c38",
        "53db5d5c-fe90-47d1-8252-c05566bf03e7",
        "55675812-1eac-44e0-a57d-30a5a9ae083e",
        "56b8d410-341c-4e90-ad7c-3e8bf1cbb0b6",
        "583dc6e7-fabe-4d2a-b633-4ad5aea67f8c",
        "58ec107b-68e2-4742-a497-d8d4af1523ee",
        "5aeb56d6-af68-4189-a8ec-1f8d8a0c9e02",
        "5af23884-b630-496c-b04e-b9db94250307",
        "5b1ab5af-701b-4717-9c43-98ab90a89325",
        "5cd4f692-3b13-4728-9df3-debc682e42dd",
        "61519ad0-2aea-4250-9a82-4dcdb93a569c",
        "622bf725-2471-4392-8f82-fcc9115a3745",
        "63f5e202-6b13-491a-bdfc-9f13b7e4c036",
        "65df4c37-b8a5-443d-b488-4b92ecc891d0",
        "66a5cdf8-3b0d-4d85-bdcc-68ae69205206",
        "6770c699-3b3c-4333-b70a-b7e7d839e38b",
        "67aeb5af-38f7-4949-b59f-608896d20077",
        "682564cb-62be-40ff-b5f6-0f53924d37e3",
        "6b6cee83-d69e-4a7a-b959-ee2de109d907",
        "6b74bbc2-128a-4206-82e1-eaec79579c51",
        "7260c523-b2d5-4bdc-a831-96b4815dca00",
        "77131a79-5ccb-4df8-afa4-14c71d4048b8",
        "7b8358f3-c96a-4a17-87ab-9414866e18db",
        "805f3a7b-a197-4834-a4e4-a56da3dde6b1",
        "814e0cc2-ed93-4e72-af35-40a41cef21cd",
        "83e9c0b8-b6a7-4c19-8deb-2bca29b93139",
        "83ff3fe4-b456-47ac-a5fa-1d152f900235",
        "84f93536-ce17-41eb-9241-10d4b91fd97f",
        "862ab494-446d-43f6-b752-af51e938547e",
        "879233b3-262c-4b19-a344-f4d595d08d26",
        "8861a2f4-80be-4c62-8c1e-252efad26ccc",
        "8a3533aa-d6d4-450c-8232-79e4851b6e11",
        "8aa1bd02-ee43-4e9d-b7f7-7ddc66b607f9",
        "8acefded-803b-4cf3-9829-18bd5508f907",
        "8b73cf24-6579-4a9f-b7db-62317feb4d58",
        "8c2f443e-b329-42f2-a1cf-31c150be47e0",
        "8c9f9374-4398-4b3c-92b7-2c66dc4fcbd5",
        "8cee7fce-00d0-45f1-81fe-b24bb7176679",
        "8d458ff0-0ed3-446d-bd87-cd23dae0a05f",
        "8fcc208a-30e9-49dc-b9de-51e6819a50b7",
        "900b0dc1-b236-4364-bbab-0371af4eef84",
        "9187dcd7-7b5c-4633-ae73-af8200f1dfba",
        "9324c6d7-3fdc-4614-8199-0c102f1b67c0",
        "942445c5-eb52-4128-98ca-0fbb5dd6aca4",
        "948b8d17-35af-40cc-aa50-70a12b56e433",
        "94eb0755-56a3-4235-b759-8dc2ced70875",
        "95085684-88ec-4d2a-8eba-a38268018193",
        "9580deeb-f28e-42e9-a877-1a6e1cea3dff",
        "95a070a7-086b-4f3d-a5ba-0a877f7fabf7",
        "9658fd5d-f8bf-4fd3-ba8d-6e40776039de",
        "985b1b37-40be-49aa-9df1-31aff97dfb1a",
        "9a917ae9-8e47-4eaf-aaff-e82bfc9f7d6b",
        "9b7483ba-9c6f-4c67-bf48-549384ac66c7",
        "9d7b2444-3411-4252-aa07-fdd2b1aa60ad",
        "9e4aeae9-8729-4b0f-9e84-5c1f4eeacc74",
        "9f5be69d-028c-4020-8140-439e652e2343",
        "a173be18-19f9-4485-8143-0ac56b45b1eb",
        "a262930d-17eb-4053-840c-a8f9fb035a00",
        "a54d9ef4-a46a-418b-b6cc-f10b49a946ac",
        "a60bbc2d-4d45-4896-b533-15c2cd54cf60",
        "a6c16f12-0987-4690-87fe-336710f96398",
        "a73e8ebf-1c4d-4a05-acf7-8b834b1f3b85",
        "a8049cb2-a13b-48f7-abe3-cf42f60368aa",
        "aa45a425-2c17-46de-8bc8-cbe080158c80",
        "ab6ffd82-1f4f-448a-a4a2-b0d8e861fc81",
        "ac9b4778-0bb5-4384-b4ed-3a5738ba99a4",
        "ae4e4f1a-098a-4906-a2c9-78a68f4cdd14",
        "af8c78cf-7d82-44f0-a277-abe61896b015",
        "b00fe5cd-a973-4f16-93e1-b15b5ea1d7e2",
        "b0d888c3-f3d3-4e5c-9dce-bbcb4c15ac58",
        "b1ea5c54-3b28-46c7-bc3a-8dab91d73d6a",
        "b3aed23e-d863-4a87-8c38-309cf958f930",
        "b43f7bd1-6472-4767-a175-d943f595e9ff",
        "b8bbd940-ba64-474c-bfea-37e578fc2133",
        "ba79fc54-e083-40e7-ae36-fae72980f2d0",
        "bbc41a1e-4bbe-4417-a40c-64635cc552e6",
        "bedb175f-d70a-41c0-91f4-573f7b22f675",
        "c1f31960-dee7-45ea-ac13-a4fea1c9235c",
        "c271feb1-dac7-4940-9ef0-cf4de3a6b1db",
        "c2b6746a-5780-47c2-bee3-db18bfccd409",
        "c4c0c196-c6a7-41fd-bdf1-99002be6e841",
        "c5677eca-f00e-45af-ab0c-7388438c85e3",
        "c696b0e8-299f-4270-9218-25f973bc64b4",
        "c73670a9-16ca-43bb-b7b9-110558e31798",
        "c7c1cc61-27fa-4bfd-bdef-f3e04c527987",
        "c81cafe4-2589-4207-b8ba-6abbc2e311c2",
        "c93a811e-1f47-43b6-aef9-c09338e43947",
        "c9b351d9-124b-4119-8be8-cb7c0f7e7994",
        "c9b4e8d8-34f7-474b-9233-db1d945d8fa0",
        "cd6f425b-92fe-4bdf-b277-53a558fc7c27",
        "d1cf1b0a-c3b8-45cc-93bc-792f820f3e92",
        "d2bb3e13-9a98-45e0-8b3d-71b3b89e598f",
        "d3268d64-57ae-4060-937a-7980cdc5ffcd",
        "d36edb42-c422-41b4-b82f-594d8f3850be",
        "d3af01d0-31b7-4fd9-af11-83b7bd594b12",
        "d3d33d63-101d-44fd-b6b9-4616a803225d",
        "d41622a7-9fe7-4f6c-9e19-62b431c9e36b",
        "d4691f19-57be-44c4-afc2-5b5f82ec27b5",
        "d72be89a-68d7-4ca8-bb97-1cb9be776001",
        "d7eb1a8a-344d-463c-a660-9b4ef56a8b63",
        "d83ac187-97cd-4ee0-a35f-5a1ffe6c7885",
        "d9aa9a5c-e496-4da7-ab67-adabc1f37c72",
        "da66d9bb-0d61-4a3a-afbd-792ac93ebddf",
        "dc9199c5-f484-4da4-8bf4-de8319628d35",
        "dd0b3f2a-d12e-4d58-ae4f-17b3ce7f95a4",
        "dd13f25f-77a0-4a2c-83af-bb187b79a389",
        "dde1ede1-50cc-4ce5-9f4a-fdd3a8f1e3dc",
        "df11d649-9c87-4802-b602-ec97c3e868b1",
        "df2b2506-6a64-4f94-8b7c-171a373387a3",
        "e0830d00-bcac-4084-893a-3b61de64ecf6",
        "e0f51af7-56c6-4c0e-9f0a-6c34ec0ff98a",
        "e118d556-2088-4cc2-b49a-82aad5974167",
        "e2ee5efe-d9cf-482c-8c73-a895185e3524",
        "e3c665c2-f7b6-4be6-99a4-197af6f01bdf",
        "e3d59499-2840-4a2c-81fc-19f8f12f0a65",
        "e4e6552b-35bf-4df8-8302-c9cc304d34f1",
        "e6b87c27-0b3b-4851-8fb6-7c366a6af508",
        "ea8ee8f7-b950-4cbb-95f7-c87da15d3e82",
        "eb351488-1a8b-4521-831d-9d9d0d1dc208",
        "ecfb8b5b-9e00-4faf-98f6-1e5d42140e3b",
        "ee641ee3-b4b9-4a2f-8394-ebda81037fe4",
        "f16d63b1-c63a-42e0-80c9-aa18ec513da3",
        "f2a3349e-b223-4d54-b7a1-c09664d7d91b",
        "f44b2de7-97bc-4539-8515-f110feec479d",
        "f486f8c2-ea21-4909-9af3-9cd2ff1638bc",
        "f4aafd07-9711-4850-a6b6-63efa2fe25c6",
        "f5abb4f1-ad31-4964-988c-14769501a8f7",
        "f611477f-7b2e-4f36-81da-c6cdee27d7a1",
        "f744e739-4b60-4f0e-95a5-42c9f2042f1a",
        "f77f3c8b-49e6-44fe-92a1-c0b07bbea9e9",
        "f9351b3e-e4de-4d94-b3bc-98786d35906b",
        "fbd7bc95-9f42-4c2c-94f4-27fd78a7273c",
        "fe84e1e8-87b7-463b-99e4-2e492576ccbd",
        "febb3cef-56cc-4f40-b7c2-7c2663b0dc33"])


    elif(set=="300"):
        orig_ids=np.array(['15d069af-7a36-4c33-8693-cc814845b9c3', '65df4c37-b8a5-443d-b488-4b92ecc891d0',
      'b404aea2-e1a5-4dba-ad61-8132a3841dcf', 'cb8a3787-1e23-4265-bf3d-264bfe6b25e8',
      '30fe7844-15f8-4c6b-b1d6-a642f43616b4', '3d635e50-7f62-444d-b302-8131aae56104',
      '0ce0209c-ee94-4ef0-9bea-5caad065b103', '4d7d8db6-d135-4db5-a718-11582822ff1a',
      '301734c8-1f4a-4dc7-ad19-7f24e0c4d4f6', 'd7eb1a8a-344d-463c-a660-9b4ef56a8b63',
      '0d843c9a-5c9d-4325-bae6-e44e0902e84a', 'b2a7d259-143a-4db7-9406-5a95e020d6b7',
      '157022dd-d527-460d-8abc-dca2f3310394', '948b8d17-35af-40cc-aa50-70a12b56e433',
      '87d70bed-3ed0-455c-a144-9fd955229125', '479eea59-8ad8-46aa-9456-29ab1b8f2cb2',
      '022e4ff8-e1af-43dc-b747-862ac83518d2', '9e4aeae9-8729-4b0f-9e84-5c1f4eeacc74',
      'b4b75916-a561-41e4-b3b9-5dfc2859028d', '46f29cfd-cb46-4ec7-b983-b483268bbe21',
      '407d6e31-ffbd-4e84-97ab-54b2ff396823', '13555453-55fc-4b13-8b32-b49dfcc9a8f0',
      '16db9568-bd11-494f-b712-28a2266ea3d0', '9a917ae9-8e47-4eaf-aaff-e82bfc9f7d6b',
      '3b4966d8-f38a-46cc-ba12-4f059b753241', 'de5c2828-29c5-4545-8367-502bb0ed1004',
      'c7c1cc61-27fa-4bfd-bdef-f3e04c527987', '8c2f443e-b329-42f2-a1cf-31c150be47e0',
      '6b74bbc2-128a-4206-82e1-eaec79579c51', '53400e38-644f-40bf-9968-b11f6682e7e5',
      '50846a75-cde7-44d0-878d-bedc39726f75', 'dbd205d8-339f-4432-aafc-a983ee553245',
      'f29a8628-ed1d-445a-9316-b4eb62fed4d7', '360dda05-d7a7-4ec3-8988-256bb6381cda',
      'dd13f25f-77a0-4a2c-83af-bb187b79a389', '24809752-2bcd-424f-aac1-5d909026b0c4',
      '3b6fda64-bb3f-4a77-bec0-7ba034d4540e', '61519ad0-2aea-4250-9a82-4dcdb93a569c',
      'd3af01d0-31b7-4fd9-af11-83b7bd594b12', 'efb8ddb1-b9a4-4833-a876-c8adc53850ca',
      'df11d649-9c87-4802-b602-ec97c3e868b1', '8aa1bd02-ee43-4e9d-b7f7-7ddc66b607f9',
      'c576bedb-b0f7-4458-a1e0-24071d6e1d36', 'c018eb43-ab13-401d-8743-53cdc93c9e65',
      '8d458ff0-0ed3-446d-bd87-cd23dae0a05f', 'ecfb8b5b-9e00-4faf-98f6-1e5d42140e3b',
      'ee179373-c30e-4ef2-9f32-75419e005cf4', '4203df44-c6be-49a5-93b8-a9047d438fe4',
      '351fbcd3-c1ec-416c-bed7-195fe5d1f41d', '63684250-6bd1-41e6-b270-ebfc5379a271',
      'f2aa92b3-ac1a-4afe-9dda-977c07ee8ee2', 'b0d888c3-f3d3-4e5c-9dce-bbcb4c15ac58',
      '862ab494-446d-43f6-b752-af51e938547e', '0990887a-6163-4c80-9c9e-468ea2598202',
      'd2bb3e13-9a98-45e0-8b3d-71b3b89e598f', '67aeb5af-38f7-4949-b59f-608896d20077',
      '24e24816-b56b-4c16-9e6d-0ae8afc8650f', 'efa1c292-bb43-420f-a229-1d1da9901b13',
      'c271feb1-dac7-4940-9ef0-cf4de3a6b1db', 'd609008d-6efb-4cd0-ab84-6c0de55198db',
      '40556c24-113c-49f9-9371-34931d4fba2d', 'c7091bfb-1ab2-4fbe-9d09-ae801a2585a2',
      '0457f007-211c-4dc3-844e-47d724fece51', 'a5e9c6b1-40b2-4763-862b-61811f58f2cb',
      'e118d556-2088-4cc2-b49a-82aad5974167', 'dd0b3f2a-d12e-4d58-ae4f-17b3ce7f95a4',
      'b43f7bd1-6472-4767-a175-d943f595e9ff', 'd3268d64-57ae-4060-937a-7980cdc5ffcd',
      '95085684-88ec-4d2a-8eba-a38268018193', 'ec533293-9f82-42ef-9ee1-a7b8c4710dc3',
      '41bf78e2-4791-4d2e-9459-22f401074964', 'cf9ee6f5-94ff-4097-a751-b8ed7c2a2e4b',
      '290e568c-16e2-4656-8e04-7298f5884c48', '9c0a19ac-3240-4e0e-9e1d-b49b74681ce0',
      '5134cb88-3e01-4861-a585-395201e803c9', '9324c6d7-3fdc-4614-8199-0c102f1b67c0',
      '6cd16a05-3496-4c8e-a915-b44ac950e241', 'f2a3349e-b223-4d54-b7a1-c09664d7d91b',
      'a60bbc2d-4d45-4896-b533-15c2cd54cf60', 'ed1698e5-4104-4b28-a0e7-71d710f2efb7',
      '985b1b37-40be-49aa-9df1-31aff97dfb1a', 'b2bc6b0a-b4f0-4ebf-829a-a2f28addae50',
      'd1cf1b0a-c3b8-45cc-93bc-792f820f3e92', '34192284-e4f1-4420-ab35-0eb02ce2e8d1',
      'a54d9ef4-a46a-418b-b6cc-f10b49a946ac', '56b4bef0-2bbb-4e2a-b434-36bfdb2f0f7e',
      'a6c16f12-0987-4690-87fe-336710f96398', '45a86c85-0bd3-4f3f-bcdc-2b27cd539288',
      'f16d63b1-c63a-42e0-80c9-aa18ec513da3', '583dc6e7-fabe-4d2a-b633-4ad5aea67f8c',
      'c87dcc5b-2846-4eca-b6c2-ecc7ea58bbc7', 'c81cafe4-2589-4207-b8ba-6abbc2e311c2',
      'fbd7bc95-9f42-4c2c-94f4-27fd78a7273c', '6bf8a6da-a8c3-45c4-8aa3-75649cd1772d',
      '110a98b6-ef01-44b2-a19c-9ab8db1dd6fa', 'f744e739-4b60-4f0e-95a5-42c9f2042f1a',
      'af8c78cf-7d82-44f0-a277-abe61896b015', 'a6006dcb-2c1a-4e66-8db0-d9deb6c93752',
      '784af6b8-2218-4e01-a066-b66eb2832d9d', '0c824653-a13b-4a4e-b907-660f1d8f8981',
      '5dc54855-a59b-47e3-867b-46eb1c789b23', '326a6c55-c963-42c2-bb8a-2591993aaaa2',
      'df2b2506-6a64-4f94-8b7c-171a373387a3', '20bae182-cdd9-4e73-a584-f337bbfa55df',
      '1b46a6a9-0e7d-4740-b101-b5bc28154f2c', '2d954abb-e11a-4f25-8f12-19a210c8d8e1',
      '07a34e07-8ee4-434b-8c09-6cb197451464', 'fe84e1e8-87b7-463b-99e4-2e492576ccbd',
      '4ac3f410-c8fd-4cf6-a065-405683c27499', '682564cb-62be-40ff-b5f6-0f53924d37e3',
      'febb3cef-56cc-4f40-b7c2-7c2663b0dc33', 'f1a772e9-bf5f-4bc9-96ea-7a45f38c8c41',
      'b8e437d3-d8db-48c3-91b5-ebd6ab29662f', '8cee7fce-00d0-45f1-81fe-b24bb7176679',
      'bbc41a1e-4bbe-4417-a40c-64635cc552e6', '58ec107b-68e2-4742-a497-d8d4af1523ee',
      'c1f31960-dee7-45ea-ac13-a4fea1c9235c', 'c2b6746a-5780-47c2-bee3-db18bfccd409',
      '8c9f9374-4398-4b3c-92b7-2c66dc4fcbd5', '757093f1-191a-4c71-9c28-92d476959e27',
      'ab6ffd82-1f4f-448a-a4a2-b0d8e861fc81', '95a070a7-086b-4f3d-a5ba-0a877f7fabf7',
      '30e340a8-776f-40d1-9d8c-d70d38bf0a5f', '5accb319-08ca-402a-9778-ad1f0608401b',
      '24d285d2-c1c9-4a9b-ab57-68fe292cf472', 'aa45a425-2c17-46de-8bc8-cbe080158c80',
      '3833df3d-dc81-4467-bdd9-16a7d99f7edb', 'b4ff7130-3055-4ed1-a878-8dfaca7191ac',
      'be4297a8-d763-42e2-a2cb-cab38f64cfe3', 'bedb175f-d70a-41c0-91f4-573f7b22f675',
      'e2ee5efe-d9cf-482c-8c73-a895185e3524', 'f9351b3e-e4de-4d94-b3bc-98786d35906b',
      'a7e8df51-db85-42c3-9ebe-207dcd736793', 'e3d59499-2840-4a2c-81fc-19f8f12f0a65',
      'f80e8b55-30d6-47fd-8f12-851016b0b0fa', '75f01e07-425c-46ac-bcfc-aba98f2b02be',
      'b18bc631-88a3-4c7a-89aa-14b3350aa48d', 'e6b87c27-0b3b-4851-8fb6-7c366a6af508',
      '66c74335-a160-4d1c-8165-8e49a3c03666', '8861a2f4-80be-4c62-8c1e-252efad26ccc',
      '6b6cee83-d69e-4a7a-b959-ee2de109d907', 'c1d277ee-1f7a-4dd1-b50a-40a8540d381b',
      '41d28e85-3a78-40a8-ad6f-aa22aa02795c', '3c77a951-188c-4c57-93b0-b7a194d4cc19',
      '059a9d92-4d36-40cb-84cc-408f9210821b', '53b45c45-9ca0-4ec9-b54d-90ffdfec0c38',
      '80134e57-9f6a-42e6-bd13-01875b5af715', 'a8049cb2-a13b-48f7-abe3-cf42f60368aa',
      'c9b4e8d8-34f7-474b-9233-db1d945d8fa0', '5aeb56d6-af68-4189-a8ec-1f8d8a0c9e02',
      'a73e8ebf-1c4d-4a05-acf7-8b834b1f3b85', '84f93536-ce17-41eb-9241-10d4b91fd97f',
      'b6a84ed4-e9d1-4df9-a293-938666fb512b', 'ac3c89bc-11bb-447a-b226-5a4a935e9653',
      '055bed5b-60ec-43e6-9110-137f2a36d65b', 'f44b2de7-97bc-4539-8515-f110feec479d',
      '166b9e2f-08a1-425f-9617-875d6ad3f834', '323cc961-5f66-4273-9d46-6097bc692269',
      '260f551d-e3c1-475e-b242-f17aad20ba2c', 'ee641ee3-b4b9-4a2f-8394-ebda81037fe4',
      '397c6457-0954-4cd2-995c-2fbeb6c72097', 'f4aafd07-9711-4850-a6b6-63efa2fe25c6',
      '74fd99d6-7bfc-40ae-b90f-1639c79294d3', '8fcc208a-30e9-49dc-b9de-51e6819a50b7',
      '6f157d6b-d974-4ec4-9c93-7e29bdaf0108', '2f256f48-8336-4846-9362-7349ae648dc5',
      '1c47637b-d4a1-432f-9810-b6e2daea0a43', '26d5159f-ca0f-4537-a97d-8ddda3c95f8a',
      '7b8358f3-c96a-4a17-87ab-9414866e18db', '089cc9f8-044f-40d6-aca9-af76bf35e4d2',
      '3b4f130b-0287-42ac-975d-ecd4f7377b0f', '4b0f6e44-14a3-4cd9-abab-e279af6f45c6',
      '58c394c6-19ba-48ad-9bc2-bc3efb4053b5', 'bb23dbbc-b679-4849-b1d8-63279cad50e2',
      '82a921b9-361a-4fd5-8db7-98961fdbf25a', 'feaad88c-c9f0-405a-a32f-4c9adfc0be6b',
      'b61ebedd-272f-4276-9eee-63bb9e1a4ad6', 'b8bbd940-ba64-474c-bfea-37e578fc2133',
      '3f38d579-66fa-4126-b269-3f4e1ee42b39', 'd83ac187-97cd-4ee0-a35f-5a1ffe6c7885',
      '35fff957-2ad7-40e6-b57a-30235da5c9a9', '57dc658f-bb87-452c-94fc-9bd6a72b0bcc',
      '5cd4f692-3b13-4728-9df3-debc682e42dd', '34ce3561-6cd1-43e8-8a5b-c1d468863166',
      'b7e851d4-1cbc-4bd1-a040-d21e68182d49', '2b6a9abb-9a75-4d1b-8cab-5dd862626349',
      '83e9c0b8-b6a7-4c19-8deb-2bca29b93139', '176dfa9b-725b-4813-9219-005041320db4',
      'da6b7c06-9bb4-45d0-b8c7-e7dadf77285c', '5af23884-b630-496c-b04e-b9db94250307',
      'd0966bbb-d634-419e-ba57-8cf6de4e98ee', 'ded067ce-6c9a-4b7d-93ca-b848b6977d45',
      'ea8ee8f7-b950-4cbb-95f7-c87da15d3e82', '8b73cf24-6579-4a9f-b7db-62317feb4d58',
      '02e82ef0-acb8-4366-8c83-4c3f2f69f7ea', 'ae2fae6b-0365-4a35-baee-144f2642cac4',
      'd3c0b2d1-4ab7-48e0-8ac4-fc6a104b03d3', '814e0cc2-ed93-4e72-af35-40a41cef21cd',
      '3d1e1b85-e0de-46a9-853f-4c3aa2e56271', 'c4c0c196-c6a7-41fd-bdf1-99002be6e841',
      '3893e3e7-24b7-45f3-90e8-e9ca6114e83f', '21faaa13-7447-420f-aa5f-7a2e128d06aa',
      '8a3533aa-d6d4-450c-8232-79e4851b6e11', '84b129ce-028e-4709-b410-ba9f18d29902',
      '6770c699-3b3c-4333-b70a-b7e7d839e38b', '5eaca290-9215-4b6d-907b-756aa17adbd0',
      '1fa707a3-63c4-488b-9558-f4d827fc9b12', 'cb29da71-7fe9-4270-83d1-ba58b8034fc6',
      'e4e6552b-35bf-4df8-8302-c9cc304d34f1', '1a2eb8d4-077e-4c0b-88d9-574872b45e4c',
      '34223626-fab9-48f5-82bd-0b0037745994', 'e3c665c2-f7b6-4be6-99a4-197af6f01bdf',
      '9d18a57c-ae79-434c-81f9-b20e9f0e21d3', 'da66d9bb-0d61-4a3a-afbd-792ac93ebddf',
      '00ab666c-afb8-476e-9872-6472b4e66b68', 'fecd1ca1-4240-42ab-ab94-bfb993a1da11',
      '81146c4b-a952-433f-98ed-1a125896d36f', '7fed6abf-3b16-41e3-8e9d-ec61382bce7e',
      '8013239e-0d75-4640-9ec1-2b9d856956a0', '63f5e202-6b13-491a-bdfc-9f13b7e4c036',
      'eb351488-1a8b-4521-831d-9d9d0d1dc208', 'b64eaa81-425f-446a-a521-9fdd9429f77b',
      'c6a176bc-6b30-4066-a7a4-b305ea1df716', '17b07883-4959-4037-9b80-dde9a06b80ae',
      'e0830d00-bcac-4084-893a-3b61de64ecf6', '82d8787f-ed86-41a0-b342-451c6064dc59',
      '879233b3-262c-4b19-a344-f4d595d08d26', 'c9b351d9-124b-4119-8be8-cb7c0f7e7994',
      '2bc2ade2-b154-4412-878c-98466a00ba7b', '75f6a3c6-ef3d-43e2-810c-560efaf92592',
      '2946b01f-4cda-4f69-9b3f-24aed26c4d15', '53db5d5c-fe90-47d1-8252-c05566bf03e7',
      '51ebb070-7d34-40b3-8520-55d2e318438a', '42ded059-c890-42e1-adb8-4b339d0879b5',
      '9580deeb-f28e-42e9-a877-1a6e1cea3dff', 'b00fe5cd-a973-4f16-93e1-b15b5ea1d7e2',
      'e7500981-9b13-4238-a855-52b91ed6244d', '0bee012c-efcb-4512-8c29-13d0a935cb48',
      '67556c68-b9a4-49e5-a3d6-7dc3c7b3e0ec', '3ca3dbf5-2390-409e-bd2c-c9f23a255e75',
      '66a5cdf8-3b0d-4d85-bdcc-68ae69205206', '136f8891-af6f-49c1-a69a-b4acd7116a3c',
      'f611477f-7b2e-4f36-81da-c6cdee27d7a1', '2b4ae7c4-ee25-4588-a0cd-bffb64a07f7f',
      'd72be89a-68d7-4ca8-bb97-1cb9be776001', '05180b17-aae0-4920-bfd0-96b062017f7d',
      '8acefded-803b-4cf3-9829-18bd5508f907', 'e0f51af7-56c6-4c0e-9f0a-6c34ec0ff98a',
      'ba79fc54-e083-40e7-ae36-fae72980f2d0', '28fba926-ef44-4874-a209-ac6680441822',
      '038aafca-cc30-47c6-9cbe-5c2cb52d8f04', 'ac48132f-2c65-4762-bb64-ed8f733a540d',
      'cdd98fba-4d2c-45d1-94b3-4b6b6077b58e', '9b7483ba-9c6f-4c67-bf48-549384ac66c7',
      'f486f8c2-ea21-4909-9af3-9cd2ff1638bc', 'ddf11c46-b071-4f00-82bc-c0ee74d78aa0',
      'd41622a7-9fe7-4f6c-9e19-62b431c9e36b', 'f2d36ddc-d20e-46c7-974c-fb7fc273db31',
      '24ea7a3f-4b4d-48a0-a256-6de2e8298d25', 'd4691f19-57be-44c4-afc2-5b5f82ec27b5',
      '3ff76631-5492-4693-8b20-73b5a6c8864d', 'a262930d-17eb-4053-840c-a8f9fb035a00',
      '36f6e239-816a-4508-9126-3b612741c26d', '35daf881-da7c-4779-a8e9-20a3985094e2',
      'd197d735-bb0d-4df5-9566-2325de86e98b', '89e13844-fa61-46bd-b99f-46423122de5a',
      'd3d33d63-101d-44fd-b6b9-4616a803225d', 'b1ea5c54-3b28-46c7-bc3a-8dab91d73d6a',
      '9d7b2444-3411-4252-aa07-fdd2b1aa60ad', 'b3aed23e-d863-4a87-8c38-309cf958f930',
      '7260c523-b2d5-4bdc-a831-96b4815dca00', '3aa0bbf3-e880-4a78-8a97-de9834b2eb18',
      'd36edb42-c422-41b4-b82f-594d8f3850be', '42bbf143-b184-4cf7-9afa-289468d9e36b',
      '3ea4837f-104a-44d3-a006-4bc586c0f829', 'dde1ede1-50cc-4ce5-9f4a-fdd3a8f1e3dc',
      '20a07a1c-ee7e-4958-a2c8-0db6e4fe0ce9', '77131a79-5ccb-4df8-afa4-14c71d4048b8',
      '6fb3efa6-3ede-43c5-bdea-0b64907a68de', 'c93a811e-1f47-43b6-aef9-c09338e43947',
      'c696b0e8-299f-4270-9218-25f973bc64b4', 'dc9199c5-f484-4da4-8bf4-de8319628d35',
      '92e8540c-3290-407d-9114-8458b9bd211a', '9f5be69d-028c-4020-8140-439e652e2343',
      'de4021b2-547f-4a26-a3ac-daba9c71edd0', '26eeb04e-ab64-45b2-bb12-7aff7a779f9a',
      '912427ec-befd-4a0f-b2ad-36af77198e06', '472c78e7-d814-4f61-9128-336d13fb9a0f',
      '9658fd5d-f8bf-4fd3-ba8d-6e40776039de', '0a11d9aa-1e9c-4f5e-94cf-faa6e796a855',
      '61d1a237-d70f-49b0-89ba-cea4d2526832', '83ff3fe4-b456-47ac-a5fa-1d152f900235',
      'ae4e4f1a-098a-4906-a2c9-78a68f4cdd14', '1bdf0668-a632-4290-ad94-c6269f9e924a',
      '622bf725-2471-4392-8f82-fcc9115a3745', '9187dcd7-7b5c-4633-ae73-af8200f1dfba',
      '2fb5e890-afaf-428a-8e28-a7c70bf8bdf1', '2ac8cb80-76be-4903-bcae-9f2c9b8b992f',
      'c2dc579e-47c5-4498-857a-2f765406b8a5', 'af533808-e79f-4f2b-994b-969dcf0e4f5d',
      'c6574c0d-ceca-4584-af55-d8e7e282ed8d', '942445c5-eb52-4128-98ca-0fbb5dd6aca4',
      'a173be18-19f9-4485-8143-0ac56b45b1eb', '077ff26a-4f7b-48d2-833e-1b9d31cb7615',
      '03ec3750-641a-4039-8b5d-74b485bde1ea', 'd9aa9a5c-e496-4da7-ab67-adabc1f37c72',
      '5034a63c-298d-4d16-bfd7-7510c09b78d3', 'f5abb4f1-ad31-4964-988c-14769501a8f7'])

    else:
        print("Error: Requested subject set %s is not defined"%(num))
        exit()

    return(orig_ids)

def short_name_from_pkl(fname):
    """
    Extracts an indicator name from the name of a Qualtrics stream.

    Args:
        fname (str): Name of a file from which to extract the indicator name.

    Returns:
        short_name (str): Name of indicator extracted from fname.
    """

    short_name = fname[len("para_dumpdf_org.md2k.data_qualtrics.feature.v12."):-len(".pkl")]
    return(short_name)

def learn_model_get_results(pkl_dir, pkl_file, edd_directory, edd_name, save=False, results_dir="experiment_output"):
    """
    Primary function responsible for processing incoming data, learning a model and outputting predictions.

    Args:
        pkl_dir (str): Path to the data file for model training.
        pkl_file (str): Name of the master data file for model training.
        exp_parameters (dict): The parameters for model training.
        edd_directory (str): Directory where the specified EDD can be found.
        edd_name (str): Filename of the EDD to read in from edd_directory.
        save (bool): Whether or not to save the results of the model training.
        results_dir (str): Directory into which results should be saved.

    Returns:
        df_perf (pandas DataFrame): DataFrame representing the results of the model training.
    """

    np.random.seed(42)
    np.random.seed(10)

    edd = None
    with open(edd_directory + edd_name) as f:
        edd = json.load(f)

    if edd is not None:
        print("loaded EDD with target: {}".format(edd["target"]["name"]))
    else:
        print("EDD load failed: %s"%(edd_directory + edd_name))
        exit()

    if "exp-parameters" in edd:
        exp_parameters = edd["exp-parameters"]
        
    else:
        print("EDD missing required parameters")
        exit()

    # safety check for existing directory
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # INTERFACE TODO: the name of the experiment will come from the EDD 
    # short_name = short_name_from_pkl(pkl_file) 
    short_name = edd_name[:-len(".json")]


    print("Loaded summary file:",os.path.join(pkl_dir,pkl_file))
    out       = pickle.load( open(os.path.join(pkl_dir,pkl_file), "rb" ) )
    try:    
        df_raw    =  out["dataframe"]
        meta_data = out["metadata"]
    except:
        df_raw = out
    df_raw[df_raw==999] = np.nan

    #Correction to collect gats.status streams
    #Values are converted to strings during CSV generation
    edd_target_stream = edd["target"]["name"]
    if(edd_target_stream == "org.md2k.data_qualtrics.feature.v15.igtb.gats.status&value"):
        level1 = (1- np.isnan(df_raw["org.md2k.data_qualtrics.feature.v15.igtb.gats.status&value(current)"] ))
        level2 = 2*(1- np.isnan(df_raw["org.md2k.data_qualtrics.feature.v15.igtb.gats.status&value(past)"]))
        level3 = 3*(1- np.isnan(df_raw["org.md2k.data_qualtrics.feature.v15.igtb.gats.status&value(never)"]  ))          
        vals = (level1 + level2 + level3) 
        valmap=np.array([np.nan,1,2,3])
        vals = vals.apply(lambda x: valmap[x])
        df_raw["org.md2k.data_qualtrics.feature.v15.igtb.gats.status&value"]=vals
        
    #Deterime what streams are in the EDD    
    edd_marker_streams = ["target"] #add target to avoid dropping later
    for e in edd["marker-streams"]:
        s = e["name"]
        edd_marker_streams.append(s)

    #Check the set of streams in the master
    #This is qualtrics and markers combined
    master_streams = df_raw.columns

    #Copy the target stream to the target field
    
    print(edd_target_stream)
    
    if edd_target_stream in master_streams:
        df_raw["target"] = df_raw[edd_target_stream]
    else:
        print("Warning: The target specified in the EDD does not exist as a stream in the master summary")
        df_empty = pd.DataFrame()
        return(df_empty)

    #Drop all of the columns that are not listed as marker streams
    #in the EDD
    if(exp_parameters["experiment-type"]=="intake"):
        filter_streams = ["org.md2k.data_analysis.feature.phone.driving_total.day",
                            "org.md2k.data_analysis.feature.phone.bicycle_total.day",
                            "org.md2k.data_analysis.feature.phone.still_total.day",
                            "org.md2k.data_analysis.feature.phone.on_foot_total.day",
                            "org.md2k.data_analysis.feature.phone.tilting_total.day",
                            "org.md2k.data_analysis.feature.phone.walking_total.day",
                            "org.md2k.data_analysis.feature.phone.running_total.day",
                            "org.md2k.data_analysis.feature.phone.unknown_total.day"]
        master_streams = df_raw.columns
        cols_to_drop   = list(set(master_streams)-set(filter_streams+edd_marker_streams))
        edd_df_raw     = df_raw.drop(columns = cols_to_drop)
    else:
        edd_df_raw =df_raw
    
    #Filter qualtrics and derived streams out    
    master_streams = edd_df_raw.columns
    cols_to_drop=[]
    for s in(master_streams):
        if ("qualtrics" in s) or ("(" in s):
            cols_to_drop.append(s)
    edd_df_raw = edd_df_raw.drop(columns = cols_to_drop)       
    edd_df_streams = edd_df_raw.columns

    
    #Perform a stratified train-test split
    #Based on participant location codes
    all_ids = get_ids(set=exp_parameters["subject_set"])    
    umn_id = userid_map.perform_map(all_ids, "data/mperf_ids.txt")
    location = np.array([int(x[0]) for x in umn_id ]) #Get location inidcator    
    tr_ids,te_ids= train_test_split(all_ids, train_size=exp_parameters["train_test_split"], stratify=location, random_state=11) 

    X_tr, y_tr, Q_tr, G_tr, X_te, y_te, Q_te, G_te, MG, features, df_tr, df_te = raw_df_to_train(edd_df_raw.copy(), tr_ids, te_ids, exp_parameters)

    if exp_parameters["model"] == "lasso-ridge":
        from MLE.linear_regression_one import LinearRegressionOne
        model = LinearRegressionOne
        
    elif exp_parameters["model"] == "nn-regression":
       from MLE.nn_regression_one import NNRegressionOne
       model = NNRegressionOne        
    elif exp_parameters["model"] =="lr":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression
    else:
        raise ValueError('Invalid model!')

    # Estimate performance
    perf=trainTestPerformanceEstimator(short_name,model,features,exp_parameters["hyper_parameters"], exp_parameters["cv_folds"], exp_parameters["cv_type"])
    perf.estimate_performance(X_tr,y_tr,G_tr,X_te,y_te,G_te)
    df_perf,df_features = perf.report()   

    df_te["prediction"] = 0*df_te["target"]
    df_te["prediction"] = perf.yhat_test
    df_tr["prediction"] = 0*df_tr["target"]
    df_tr["prediction"] = perf.yhat_train

    df_te = df_te[["target","prediction"]]
    df_tr = df_tr[["target","prediction"]]

    #df_te_res=df_te["target"]
    #df_te_res["prediction"] = 0*df_te["target"]
    #df_te_res["prediction"] = perf.yhat_test
    
    #df_tr_res=df_tr["target"]
    #df_tr_res["prediction"] = 0*df_tr["target"]
    #df_tr_res["prediction"] = perf.yhat_train    

    exp_fields = ["Indicator","Master Summary"] + list(exp_parameters.keys())
    exp_vals   = [short_name, os.path.join(pkl_dir,pkl_file)] + list(exp_parameters.values())
    df_config = pd.DataFrame(data={"Experiment Parameter": exp_fields, "Value": exp_vals})

    #output={"df_config":df_config, "df_raw":df_raw, "df_tr":df_tr, "df_te":df_te, "df_perf":df_perf,"df_features":df_features,  "perf":perf}
    #output={"df_config":df_config, "df_raw":df_raw, "df_tr":df_tr_res, "df_te":df_te_res, "df_perf":df_perf,"df_features":df_features}

    output={"df_config":df_config, "df_tr":df_tr, "df_te":df_te, "df_perf":df_perf,"df_features":df_features}

    pickle.dump( output, open( results_dir + "%s-%s.pkl"%(exp_parameters["exp_name"],short_name), "wb" ), protocol=2 )

    if(save):
        data_frame_to_csv(df_te, "target", short_name, prefix="ground_truth_")
        data_frame_to_csv(df_te, "prediction", short_name, prefix="prediction_")
        
    return(df_perf)

def parallel_learn_worker(params):
    """
    Parallelizable function responsible for unpacking the experiment parameters pulled from the EDD
    and passing them to learn_model_get_results().

    Args:
        params (dict): The set of all parameters describing the setup for an experiment.

    Returns:
        output of learn_model_get_results() (pandas DataFrame): DataFrame representing the results of the model training.
    """
    
    np.random.seed(42)
    np.random.seed(10)

    #print("incoming params: {}".format(params))

    full_params = json.loads(params).copy()
    pkl_dir = full_params["pkl_dir"]
    pkl_file = full_params["pkl_file"]
    edd_directory = full_params["edd_directory"]
    edd_name = full_params["edd_name"]
    save = full_params["save"]
    results_dir = full_params["results_dir"]

    return learn_model_get_results(pkl_dir, pkl_file, edd_directory, edd_name, save=save, results_dir=results_dir)

def main(args):
    """
    Entry point to the model-training and prediction pipeline following summarization.  
    Responsible for checking and handling run-time arguments,
    packaging up the experiment parameters, setting parallelization (according to the "parallelism" parameter),
    starting execution of the pipeline, and finally collecting, concatenating and display the results of the model
    training.

    Args:
        args (argparse.Namespace): The set of arguments passed in at the command line.
    """
    
    np.random.seed(42)
    np.random.seed(10)

    no_spark = args.no_spark
    
    pd.set_option('display.width', 1000)

    # get data directory, handle if missing
    if args.data_file is None:
        print("Need a path to a master data file")
        exit()


    print("Using master file: %s"%(args.data_file))
    pkl_dir = os.path.dirname(args.data_file)
    pkl_file = os.path.basename(args.data_file)

    if not os.path.isdir(pkl_dir):
        print("data directory not found!")

        # TODO: throw an exception or something to indicate the directory doesn't exist
        os.makedirs(pkl_dir)

    # get EDD directory, handle if missing
    if args.edd_dir is not None:
        edd_directory = args.edd_dir

    if not os.path.isdir(edd_directory):
        print("edd directory not found: {}".format(edd_directory))

        # TODO: again -- handle this better than a silent loop over no content
        os.makedirs(edd_directory)

    # get EDD name, if any
    if args.edd_name is not None:
        edd_name = args.edd_name

    else:
        edd_name = None

    # set output directory
    mdd=os.path.splitext(pkl_file)[0]
    results_dir = "experiment_output/%s/"%(mdd)
    if not os.path.isdir(results_dir):
        try:
            os.makedirs(results_dir)
        except:
            pass

    full_params = {}
    full_params["pkl_dir"] = pkl_dir
    full_params["pkl_file"] = pkl_file
    full_params["edd_directory"] = edd_directory
    full_params["save"] = False
    full_params["results_dir"] = results_dir


    # loop through or parallelize a directory of EDDs
    if edd_name is None:
        all_jobs = []
        for edd_name in os.listdir(edd_directory):
            
            if(True):
            
                if(".json" in edd_name):
                    full_params_copy = full_params.copy()
                    full_params_copy["edd_name"] = edd_name
                    if no_spark:     
                        #try:               
                            df_results = parallel_learn_worker(json.dumps(full_params_copy))
                            print(df_results)
                        #except:
                        #    print("Error: EDD %s could not be processed"%(edd_name))
                            
                        
                    else:
                        # this will be the basis of the new jobs list
                        all_jobs.append(json.dumps(full_params_copy))

        if not no_spark:
            # derive job_list from all_jobs (above)
            job_list = sc.parallelize(all_jobs)
            job_map = job_list.map(parallel_learn_worker)
            df_results_list=job_map.collect()
            df_results = pd.concat(df_results_list)

    # run if EDD is specified at command line
    else:

        print ("edd_name: {}".format(edd_name))
        full_params["edd_name"] = edd_name
        # testing setup prior to parallelizing
        # this_result = parallel_learn_worker(full_params)
        if no_spark:

            df_results = parallel_learn_worker(json.dumps(full_params))
            print(df_results)
        else:
            # testing parallelization
            job_list = sc.parallelize([json.dumps(full_params)])
            job_map = job_list.map(parallel_learn_worker)
            df_results_list=job_map.collect()        
            df_results = pd.concat(df_results_list)
            print(df_results)
    
if __name__ == "__main__":
    """
    Start of execution.  Creates argparse.ArgumentParser() and parses the argument list 
    that main() uses to execute the pipeline.
    """
    
    parser = argparse.ArgumentParser(description="mPerf EMS")
    parser.add_argument("--data-file", help="path to data file")
    parser.add_argument("--edd-dir", help="path to directory containing EDDs")
    parser.add_argument("--edd-name", nargs='?', help="optional: single EDD filename")
    parser.add_argument("--no-spark", action='store_const', const=True, help="parallelism: 'multi' or 'none'")
    
    args = parser.parse_args()

    main(args)















    