from globals import *
from mvcs import *

import pandas as pd
import numpy as np
import os
import os.path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score
from mapie.metrics import classification_mean_width_score


# Function to fit multiple models.
def fit_models_3v(X_train, y_train, random_seed):
    classifiers = []
    
    colnames = list(X_train.columns)
    
    # V1
    selectedcols = [x for x in colnames if "v1_" in x]
    df = X_train.loc[:, selectedcols]
    model_v1 = RandomForestClassifier(n_estimators=NTREES, random_state=random_seed, n_jobs=NUMCORES).fit(df, y_train)
    classifiers.append(("v1",model_v1))
    
    # V2
    selectedcols = [x for x in colnames if "v2_" in x]
    df = X_train.loc[:, selectedcols]
    model_v2 = RandomForestClassifier(n_estimators=NTREES,random_state=random_seed, n_jobs=NUMCORES).fit(df, y_train)
    classifiers.append(("v2",model_v2))
    
    # V3
    selectedcols = [x for x in colnames if "v3_" in x]
    df = X_train.loc[:, selectedcols]
    model_v3 = RandomForestClassifier(n_estimators=NTREES,random_state=random_seed, n_jobs=NUMCORES).fit(df, y_train)
    classifiers.append(("v3",model_v3))
    
    # Aggregated
    model_agg = RandomForestClassifier(n_estimators=NTREES,random_state=random_seed, n_jobs=NUMCORES).fit(X_train, y_train)
    classifiers.append(("aggregated",model_agg))
    
    # MVCS
    # Get column indices of views.
    ind_v1 = [colnames.index(x) for x in colnames if "v1_" in x]
    ind_v2 = [colnames.index(x) for x in colnames if "v2_" in x]
    ind_v3 = [colnames.index(x) for x in colnames if "v3_" in x]

    model_mvcs = mvcs3(ind_v1, ind_v2, ind_v3, NTREES, NUMCORES, 10, random_seed)
    model_mvcs.fit(X_train, y_train)
    classifiers.append(("mvcs",model_mvcs))


    return classifiers

# Function to fit multiple models.
def fit_models_2v(X_train, y_train, random_seed):
    classifiers = []
    
    colnames = list(X_train.columns)
    
    # V1
    selectedcols = [x for x in colnames if "v1_" in x]
    df = X_train.loc[:, selectedcols]
    model_v1 = RandomForestClassifier(n_estimators=NTREES, random_state=random_seed, n_jobs=NUMCORES).fit(df, y_train)
    classifiers.append(("v1",model_v1))
    
    # V2
    selectedcols = [x for x in colnames if "v2_" in x]
    df = X_train.loc[:, selectedcols]
    model_v2 = RandomForestClassifier(n_estimators=NTREES,random_state=random_seed, n_jobs=NUMCORES).fit(df, y_train)
    classifiers.append(("v2",model_v2))
    
    # Aggregated
    model_agg = RandomForestClassifier(n_estimators=NTREES,random_state=random_seed, n_jobs=NUMCORES).fit(X_train, y_train)
    classifiers.append(("aggregated",model_agg))
    
    # MVCS
    # Get column indices of views.
    ind_v1 = [colnames.index(x) for x in colnames if "v1_" in x]
    ind_v2 = [colnames.index(x) for x in colnames if "v2_" in x]

    model_mvcs = mvcs2(ind_v1, ind_v2, NTREES, NUMCORES, 10, random_seed)
    model_mvcs.fit(X_train, y_train)
    classifiers.append(("mvcs",model_mvcs))
    
    return classifiers



# Function that saves a results data frame in the given folder.
def save_df(df, dataset_path, model_type, filename):
    
    os.chdir(".")
    #print("current dir is: %s" % (os.getcwd()))
    
    tmpdir = dataset_path+"results_"+model_type+"/"
    
    if os.path.isdir(tmpdir) == False:
        os.mkdir(tmpdir)
    
    df.to_csv(tmpdir+"results.csv", index=False)
	


# Compute p-values for each class as described in https://cml.rhul.ac.uk/cp.html
def compute_pvalues(calib_scores, test_scores):
    
    n = len(calib_scores)

    # Create array to store p-values
    pvalues = np.zeros(test_scores.shape)

    for r in range(test_scores.shape[0]):
        for c in range(test_scores.shape[1]):
            alpha_j = test_scores[r,c]
            pval = (sum(calib_scores >= alpha_j) + 1) / (n + 1)
            pvalues[r,c] = pval
    
    return(pvalues)

# Function that saves the classes order that correspond to the pvalues.
def save_classes_order(df, dataset_path):
    
    os.chdir(".")
    
    df.to_csv(dataset_path+"classes.csv", index=False, header=False)
