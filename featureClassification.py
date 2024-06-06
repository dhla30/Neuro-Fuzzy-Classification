import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
from scipy import stats
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import time

def findPRFC(predicted, actual, display=True) :
    f1 = f1_score(predicted, actual, average="macro")
    pre = precision_score(predicted, actual, average="macro")
    acc = accuracy_score(predicted, actual)
    rec = recall_score(predicted, actual, average="macro")
    conf_matrix = confusion_matrix(predicted, actual);
    diff_arr = np.array(predicted) - np.array(actual)
    idx = np.where(diff_arr == 0)[0]
    
    if(display) :
        print("Test f1 score : %s "% f1)
        print("Test Precision score : %s "% pre)
        print("Test accuracy score : %s "% acc)
        print("Test Recall score : %s "% rec)
        print('Confusion matrix')
        print(conf_matrix)
    return idx
    
def classifyDLAndFindCorrect(features, classes, test, test_classes, method, display=True) :
    t1 = time.time()
    
    Y = classes.astype(np.int8)
    X = np.asarray(features)
    X = X.reshape((X.shape[0], X.shape[1]))
    
    Y_test = test_classes.astype(np.int8)
    X_test = np.asarray(test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

    scaler = StandardScaler()  
    scaler.fit(X)
    X_train = scaler.transform(X)  
    X_Test = scaler.transform(X_test)
    
    if(method == 1) :
        neigh1 = KNeighborsClassifier(n_neighbors=1)
        neigh1.fit(X_train, Y)
        y_pred1 = neigh1.predict(X_Test)  
        return findPRFC(test_classes, y_pred1, display)
    elif(method == 2) :
        neigh2 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        neigh2.fit(X_train, Y)
        y_pred2 = neigh2.predict(X_Test)  
        return findPRFC(test_classes, y_pred2, display)
    elif(method == 3) :
        neigh3 = LinearSVC(random_state=0, tol=1e-5)
        neigh3.fit(X_train, Y)
        y_pred3 = neigh3.predict(X_Test)  
        return findPRFC(test_classes, y_pred3, display)
    elif(method == 4) :
        logisticRegressionClassifier = LogisticRegression(random_state=0,multi_class='auto',solver='lbfgs',max_iter=1000)
        logisticRegressionClassifier.fit(X_train,Y)
        y_pred_lrc = logisticRegressionClassifier.predict(X_test)
        return findPRFC(test_classes, y_pred_lrc, display)
    elif(method == 5) :
        randomForestClassifier = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        randomForestClassifier.fit(X_train,Y)
        y_pred_rfc = randomForestClassifier.predict(X_test)
        return findPRFC(test_classes, y_pred_rfc, display)
