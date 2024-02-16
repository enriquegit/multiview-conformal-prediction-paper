import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

class mvcs3(BaseEstimator, ClassifierMixin):
    
    def __init__(self, ind_v1=None, ind_v2=None, ind_v3=None, nt=50, n_jobs=1, k = 10, intseed = 123):
        self.ind_v1 = ind_v1
        self.ind_v2 = ind_v2
        self.ind_v3 = ind_v3
        self.nt = nt
        self.n_jobs = n_jobs
        self.k = k
        self.intseed = intseed
        
    
    def fit(self, X, y):
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.uniquelabels_ = np.unique(y)
        self.uniquelabels_ = [str(x) for x in self.uniquelabels_ ]
        
        truelabels = []

        preds_v1 = []
        scores_v1 = None

        preds_v2 = []
        scores_v2 = None

        preds_v3 = []
        scores_v3 = None
        
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.intseed)

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):

            xtrain, xtest = X[train_index], X[test_index]
            ytrain, ytest = y[train_index], y[test_index]

            xtrainv1 = xtrain[:, self.ind_v1]
            xtestv1 = xtest[:, self.ind_v1]

            xtrainv2 = xtrain[:, self.ind_v2]
            xtestv2 = xtest[:, self.ind_v2]

            xtrainv3 = xtrain[:, self.ind_v3]
            xtestv3 = xtest[:, self.ind_v3]

            truelabels.append(ytest)

            # V1
            m_v1 = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(xtrainv1, ytrain)

            labels_v1 = m_v1.predict(xtestv1)
            preds_v1.append(labels_v1)
            raw_v1 = m_v1.predict_proba(xtestv1)
            if scores_v1 is not None:
                scores_v1 = np.vstack((scores_v1, raw_v1))
            else:
                scores_v1 = raw_v1

            # V2
            m_v2 = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(xtrainv2, ytrain)

            labels_v2 = m_v2.predict(xtestv2)
            preds_v2.append(labels_v2)
            raw_v2 = m_v2.predict_proba(xtestv2)
            if scores_v2 is not None:
                scores_v2 = np.vstack((scores_v2, raw_v2))
            else:
                scores_v2 = raw_v2


            # V3
            m_v3 = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(xtrainv3, ytrain)

            labels_v3 = m_v3.predict(xtestv3)
            preds_v3.append(labels_v3)
            raw_v3 = m_v3.predict_proba(xtestv3)
            if scores_v3 is not None:
                scores_v3 = np.vstack((scores_v3, raw_v3))
            else:
                scores_v3 = raw_v3
                

        truelabels = np.concatenate(truelabels).ravel()
        preds_v1 = np.concatenate(preds_v1).ravel()
        preds_v2 = np.concatenate(preds_v2).ravel()
        preds_v3 = np.concatenate(preds_v3).ravel()
        

        # Build first-level learners with all data.
        self.m_v1_ = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(X[:, self.ind_v1], y)

        self.m_v2_ = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(X[:, self.ind_v2], y)

        self.m_v3_ = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(X[:, self.ind_v3], y)
        
        # Construct meta-features

        # Average scores
        avgscores = (scores_v1 + scores_v2 + scores_v3) / 3

        df = pd.DataFrame(avgscores)

        # Change column names to strings.
        df.columns = df.columns.astype(str)

        df = df.assign(preds_v1=preds_v1)
        df = df.assign(preds_v2=preds_v2)
        df = df.assign(preds_v3=preds_v3)
        
        # One-hot encode preds
        categories = []
        categories.append(self.uniquelabels_)
        categories.append(self.uniquelabels_)
        categories.append(self.uniquelabels_)

        self.enc_ = OneHotEncoder(categories=categories,
                                  drop='first',
                                  sparse_output=False).set_output(transform="pandas")

        predcols = df[['preds_v1','preds_v2','preds_v3']]

        ohe_predcols = self.enc_.fit_transform(predcols)

        # Remove preds cols and concatenate encoded cols.
        metaX = pd.concat([df, ohe_predcols],axis=1).drop(columns=['preds_v1','preds_v2','preds_v3'])
        metaY = truelabels

        # Train metalearner
        self.m_meta_ = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(metaX, metaY)

        
        # Return the classifier
        return self
    
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        #X = check_array(X)
        
        xtestv1 = X.values[:, self.ind_v1]
        xtestv2 = X.values[:, self.ind_v2]
        xtestv3 = X.values[:, self.ind_v3]

        preds_v1 = self.m_v1_.predict(xtestv1)
        scores_v1 = self.m_v1_.predict_proba(xtestv1)

        preds_v2 = self.m_v2_.predict(xtestv2)
        scores_v2 = self.m_v2_.predict_proba(xtestv2)

        preds_v3 = self.m_v3_.predict(xtestv3)
        scores_v3 = self.m_v3_.predict_proba(xtestv3)

        # Average scores
        avgscores = (scores_v1 + scores_v2 + scores_v3) / 3

        tmp = pd.DataFrame(avgscores)

        # Change column names to strings.
        tmp.columns = tmp.columns.astype(str)

        tmp = tmp.assign(preds_v1=preds_v1)
        tmp = tmp.assign(preds_v2=preds_v2)
        tmp = tmp.assign(preds_v3=preds_v3)

        predcols = tmp[['preds_v1','preds_v2','preds_v3']]

        ohe_predcols = self.enc_.fit_transform(predcols)

        # Remove preds cols and concatenate encoded cols.
        metaX = pd.concat([tmp, ohe_predcols],axis=1).drop(columns=['preds_v1','preds_v2','preds_v3'])

        predictions = self.m_meta_.predict(metaX)
        #scores = m_meta.predict_proba(metaX)
        
        return predictions
    
    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        #X = check_array(X)
        
        xtestv1 = X.values[:, self.ind_v1]
        xtestv2 = X.values[:, self.ind_v2]
        xtestv3 = X.values[:, self.ind_v3]

        preds_v1 = self.m_v1_.predict(xtestv1)
        scores_v1 = self.m_v1_.predict_proba(xtestv1)

        preds_v2 = self.m_v2_.predict(xtestv2)
        scores_v2 = self.m_v2_.predict_proba(xtestv2)

        preds_v3 = self.m_v3_.predict(xtestv3)
        scores_v3 = self.m_v3_.predict_proba(xtestv3)

        # Average scores
        avgscores = (scores_v1 + scores_v2 + scores_v3) / 3

        tmp = pd.DataFrame(avgscores)

        # Change column names to strings.
        tmp.columns = tmp.columns.astype(str)

        tmp = tmp.assign(preds_v1=preds_v1)
        tmp = tmp.assign(preds_v2=preds_v2)
        tmp = tmp.assign(preds_v3=preds_v3)

        predcols = tmp[['preds_v1','preds_v2','preds_v3']]

        ohe_predcols = self.enc_.fit_transform(predcols)

        # Remove preds cols and concatenate encoded cols.
        metaX = pd.concat([tmp, ohe_predcols],axis=1).drop(columns=['preds_v1','preds_v2','preds_v3'])

        scores = self.m_meta_.predict_proba(metaX)
        
        return scores
    
    
class mvcs2(BaseEstimator, ClassifierMixin):
    
    def __init__(self, ind_v1=None, ind_v2=None, nt=50, n_jobs=1, k = 10, intseed = 123):
        self.ind_v1 = ind_v1
        self.ind_v2 = ind_v2
        
        self.nt = nt
        self.n_jobs = n_jobs
        self.k = k
        self.intseed = intseed
        
    
    def fit(self, X, y):
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.uniquelabels_ = np.unique(y)
        self.uniquelabels_ = [str(x) for x in self.uniquelabels_ ]
        
        truelabels = []

        preds_v1 = []
        scores_v1 = None

        preds_v2 = []
        scores_v2 = None

        
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.intseed)

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):

            xtrain, xtest = X[train_index], X[test_index]
            ytrain, ytest = y[train_index], y[test_index]

            xtrainv1 = xtrain[:, self.ind_v1]
            xtestv1 = xtest[:, self.ind_v1]

            xtrainv2 = xtrain[:, self.ind_v2]
            xtestv2 = xtest[:, self.ind_v2]

        
            truelabels.append(ytest)

            # V1
            m_v1 = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(xtrainv1, ytrain)

            labels_v1 = m_v1.predict(xtestv1)
            preds_v1.append(labels_v1)
            raw_v1 = m_v1.predict_proba(xtestv1)
            if scores_v1 is not None:
                scores_v1 = np.vstack((scores_v1, raw_v1))
            else:
                scores_v1 = raw_v1

            # V2
            m_v2 = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(xtrainv2, ytrain)

            labels_v2 = m_v2.predict(xtestv2)
            preds_v2.append(labels_v2)
            raw_v2 = m_v2.predict_proba(xtestv2)
            if scores_v2 is not None:
                scores_v2 = np.vstack((scores_v2, raw_v2))
            else:
                scores_v2 = raw_v2
        

        truelabels = np.concatenate(truelabels).ravel()
        preds_v1 = np.concatenate(preds_v1).ravel()
        preds_v2 = np.concatenate(preds_v2).ravel()
        

        # Build first-level learners with all data.
        self.m_v1_ = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(X[:, self.ind_v1], y)

        self.m_v2_ = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(X[:, self.ind_v2], y)

        
        # Construct meta-features

        # Average scores
        avgscores = (scores_v1 + scores_v2) / 2

        df = pd.DataFrame(avgscores)

        # Change column names to strings.
        df.columns = df.columns.astype(str)

        df = df.assign(preds_v1=preds_v1)
        df = df.assign(preds_v2=preds_v2)
        
        # One-hot encode preds
        categories = []
        categories.append(self.uniquelabels_)
        categories.append(self.uniquelabels_)

        self.enc_ = OneHotEncoder(categories=categories,
                                  drop='first',
                                  sparse_output=False).set_output(transform="pandas")

        predcols = df[['preds_v1','preds_v2']]

        ohe_predcols = self.enc_.fit_transform(predcols)

        # Remove preds cols and concatenate encoded cols.
        metaX = pd.concat([df, ohe_predcols],axis=1).drop(columns=['preds_v1','preds_v2'])
        metaY = truelabels

        # Train metalearner
        self.m_meta_ = RandomForestClassifier(n_estimators=self.nt,
                                              random_state=self.intseed,
                                          n_jobs=self.n_jobs).fit(metaX, metaY)

        
        # Return the classifier
        return self
    
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        #X = check_array(X)
        
        xtestv1 = X.values[:, self.ind_v1]
        xtestv2 = X.values[:, self.ind_v2]

        preds_v1 = self.m_v1_.predict(xtestv1)
        scores_v1 = self.m_v1_.predict_proba(xtestv1)

        preds_v2 = self.m_v2_.predict(xtestv2)
        scores_v2 = self.m_v2_.predict_proba(xtestv2)

        # Average scores
        avgscores = (scores_v1 + scores_v2) / 2

        tmp = pd.DataFrame(avgscores)

        # Change column names to strings.
        tmp.columns = tmp.columns.astype(str)

        tmp = tmp.assign(preds_v1=preds_v1)
        tmp = tmp.assign(preds_v2=preds_v2)

        predcols = tmp[['preds_v1','preds_v2']]

        ohe_predcols = self.enc_.fit_transform(predcols)

        # Remove preds cols and concatenate encoded cols.
        metaX = pd.concat([tmp, ohe_predcols],axis=1).drop(columns=['preds_v1','preds_v2'])

        predictions = self.m_meta_.predict(metaX)
        #scores = m_meta.predict_proba(metaX)
        
        return predictions
    
    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        #X = check_array(X)
        
        xtestv1 = X.values[:, self.ind_v1]
        xtestv2 = X.values[:, self.ind_v2]

        preds_v1 = self.m_v1_.predict(xtestv1)
        scores_v1 = self.m_v1_.predict_proba(xtestv1)

        preds_v2 = self.m_v2_.predict(xtestv2)
        scores_v2 = self.m_v2_.predict_proba(xtestv2)


        # Average scores
        avgscores = (scores_v1 + scores_v2) / 2

        tmp = pd.DataFrame(avgscores)

        # Change column names to strings.
        tmp.columns = tmp.columns.astype(str)

        tmp = tmp.assign(preds_v1=preds_v1)
        tmp = tmp.assign(preds_v2=preds_v2)

        predcols = tmp[['preds_v1','preds_v2']]

        ohe_predcols = self.enc_.fit_transform(predcols)

        # Remove preds cols and concatenate encoded cols.
        metaX = pd.concat([tmp, ohe_predcols],axis=1).drop(columns=['preds_v1','preds_v2'])

        scores = self.m_meta_.predict_proba(metaX)
        
        return scores
    
        