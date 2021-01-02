import numpy as np #for linear algebra
import pandas as pd #for chopping, processing
import csv #for opening csv files
#%matplotlib inline 
#import matplotlib.pyplot as plt #for plotting the graphs
from sklearn.linear_model import LogisticRegression #for logistic regression
from sklearn.pipeline import Pipeline #to assemble steps for cross validation
from sklearn.preprocessing import PolynomialFeatures #for all the polynomial features
from sklearn import svm #for Support Vector Machines
from sklearn.neighbors import NearestNeighbors #for nearest neighbor classifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier #for decision tree classifier
from sklearn.naive_bayes import GaussianNB  #for naive bayes classifier
from scipy import stats #for statistical info
from sklearn.model_selection import train_test_split # to split the data in train and test
from sklearn.model_selection import KFold # for cross validation
#from sklearn.grid_search import GridSearchCV  # for tuning parameters
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier  #for k-neighbor classifier
from sklearn import metrics  # for checking the accuracy 
from time import time
from azureml.core.run import Run
path_url='https://raw.githubusercontent.com/hammad-alt/nd00333-capstone/master/data/train.csv'
from azureml.data.dataset_factory import TabularDatasetFactory

from azureml.core import Workspace, Dataset, Experiment
run=Run.get_context()
ds = Dataset.Tabular.from_delimited_files(path =path_url)
def clean_data(data):
    df=data.to_pandas_dataframe()
    print(df.head())
    print("length_of_dataframe",len(df))
    # Number of cases with active results
    active = len(df[df['Activity']==1])
    print("Number of active cases",active)
    #Number of non-active cases
    nactive = len(df[df['Activity']==0])
    print("Number of non-active cases",nactive)
    x_df=df.drop('Activity',axis=1)
   
    y_df=df['Activity']
    return x_df,y_df
def train_classifier(clf, X_train, Y_train):
    start = time()
    model=clf.fit(X_train, Y_train)
    end = time()
    print ("Trained model in {:.4f} seconds".format(end - start))
    return model
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
import argparse
import joblib
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    #parser.add_argument('--gamma', type=int, default=1, help="Maximum number of iterations to converge")
    #parser.add_argument('--kernel', type=str, default='sigmoid', help="Specifies the kernel type to be used in the algorithm")
    parser.add_argument('--coef0', type=int, default=0, help="Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’")

    args = parser.parse_args()
    run.log("Regularization Strength:", np.float(args.C))
    #run.log("Kernel:", str(args.kernel))
    run.log("coef0:", np.int(args.coef0))

    x, y = clean_data(ds)
    # TODO: Split data into train and test sets.
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=403,shuffle=True)
    clf=SVC(C=args.C,coef0=args.coef0)
    #clf=SVC(C=0.1,kernel='sigmoid',coef0=1)
    model=train_classifier(clf, x_train, y_train)
    
    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model,'outputs/model.joblib')
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
if __name__ == '__main__':
    main() 
