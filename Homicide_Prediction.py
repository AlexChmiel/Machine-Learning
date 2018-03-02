# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:11:30 2017

@author: Aleksander Chmiel
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn import cross_validation
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


def DimensionalityReduction(data,target):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    logR = linear_model.LogisticRegression()
    scores = cross_validation.cross_val_score(logR, data, target,cv=10)
    print("Original accuracy", np.mean(scores))
    
    pca = decomposition.PCA(n_components=2)
    pca.fit(data)
    data = pca.transform(data)
    print("Explainedd variance:", np.sum(pca.explained_variance_ratio_))
    
    logR = linear_model.LogisticRegression()
    scores = cross_validation.cross_val_score(logR, data, target,cv=10)
    print(np.mean(scores))
    

def RegressionScoring(data,target):
    selector = SelectPercentile(f_regression, percentile=25)
    selector.fit(data,target)
    headers = data.dtypes.index
    for n,s in zip(headers,selector.scores_):
        print("F score:",s,"for feature",n)

def runClassifiers(data, target):
    # Naive bayes 87%, Log R 90%, when guessnig black or white race 
    nearestN = KNeighborsClassifier()
    scores = model_selection.cross_val_score(nearestN, data, target, cv=10)
    print("KNN: ",scores)
    
    randomForest = RandomForestClassifier()
    scores = model_selection.cross_val_score(randomForest, data, target, cv=10)
    print ("RForest : ",scores.mean())
    
    
    rbfSvm = SVC()
    scores = model_selection.cross_val_score(rbfSvm, data, target, cv=10)
    print ("SVC : ", scores.mean())
    

    nBayes = naive_bayes.GaussianNB()
    scores = model_selection.cross_val_score(nBayes, data, target, cv=10)
    print ("Naive Bayes : ",scores.mean())
    
    logR = linear_model.LogisticRegression()
    scores = model_selection.cross_val_score(logR, data, target, cv=10)
    print ("Log R : ",scores.mean())
    
    decisionTree = tree.DecisionTreeClassifier()
    scores = model_selection.cross_val_score(decisionTree, data, target, cv=10)
    print ("Tree : ", scores.mean())
    
    

def mapStringToInt(df, columnName):
    # Map string values to int
    temp_set = set(df[columnName])
    i = 0
    temp_dict = {}
    for x in temp_set:
        temp_dict[x] = i
        i+=1
    
    df[columnName] = df[columnName].map(temp_dict)
    
    return df


def performPreProcessing(homicideDF):
    
    # Drop rows where crime is not solved and where data makes no sense (e.g. age 0)
    # I didn't want to use the mean instead as I already have 630k of data and I can afford to drop it for better accuracy
    
    homicideDF = homicideDF.query("Crime_Solved != 'No'")
    homicideDF = homicideDF.query("Perpetrator_Age != 0")
    homicideDF = homicideDF.query("Victim_Age != 0")
    homicideDF = homicideDF.query("Perpetrator_Race != 'Unknown'")
    homicideDF = homicideDF.query("Perpetrator_Race == 'Black' | Perpetrator_Race == 'White'")
    homicideDF = homicideDF.query("Victim_Race == 'Black' | Victim_Race == 'White'")
    homicideDF = homicideDF.query("Victim_Sex == 'Female' | Victim_Sex == 'Male'")
    # Drop unrelated features
    homicideDF = homicideDF.drop(['Agency_Name'], axis=1)
    #homicideDF = homicideDF.drop(['Record_Source'], axis=1)
    #homicideDF = homicideDF.drop(['Crime_Type'], axis=1)
    homicideDF = homicideDF.drop(['Perpetrator_Ethnicity'], axis=1)
    homicideDF = homicideDF.drop(['Victim_Ethnicity'], axis=1)
    
    #imputer = preprocessing.Imputer(missing_values = 'NaN', strategy='mean', axis= 0)
    #imputer.fit(homicideDF[['Perpetrator_Race']])
    #homicideDF['Perpetrator_Race']= imputer.transform(homicideDF[['Perpetrator_Race']])
    
    homicideDF['Victim_Sex'] = homicideDF['Victim_Sex'].map({'Female': 0, 'Male':1}).astype(int)
    homicideDF['Perpetrator_Sex'] = homicideDF['Perpetrator_Sex'].map({'Female': 0, 'Male':1}).astype(int)
    homicideDF['Perpetrator_Race'] = homicideDF['Perpetrator_Race'].map({'White': 0, 'Black':1}).astype(int)
    homicideDF['Victim_Race'] = homicideDF['Victim_Race'].map({'White': 0, 'Black':1}).astype(int)
    homicideDF['Crime_Solved'] = homicideDF['Crime_Solved'].map({'No': 0, 'Yes':1}).astype(int)
    #homicideDF = homicideDF.drop(['Victim_Race'], axis=1)
    
    homicideDF = mapStringToInt(homicideDF, 'Weapon')
    homicideDF = mapStringToInt(homicideDF, 'Month')
    homicideDF = mapStringToInt(homicideDF, 'City')
    homicideDF = mapStringToInt(homicideDF, 'State')
    homicideDF = mapStringToInt(homicideDF, 'Relationship')
    homicideDF = mapStringToInt(homicideDF, 'Agency_Code')
    homicideDF = mapStringToInt(homicideDF, 'Agency_Type')
    homicideDF = mapStringToInt(homicideDF,'Crime_Type')
    homicideDF = mapStringToInt(homicideDF,'Record_Source')
    
    
    return homicideDF
    
    
def main():
    homicideDF = pd.read_csv('homicide.csv', low_memory=False)
    
    homicideDF = homicideDF[:90000]
    homicideDF = homicideDF.rename(columns=lambda x: x.strip().replace(' ','_'))
    
    homicideDF = performPreProcessing(homicideDF)

    label_test = homicideDF['Perpetrator_Race']
    homicideDF = homicideDF.drop(['Perpetrator_Race'], axis=1)
    
    #runClassifiers(homicideDF, label_test)
    #DimensionalityReduction(homicideDF, label_test)
    #RegressionScoring(homicideDF,label_test)
    
    allResults = []
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    
    for train_index, test_index in kf.split(homicideDF):
        
        clf =  linear_model.LogisticRegression(penalty = 'l2')
        
        clf.fit(homicideDF.iloc[train_index], label_test.iloc[train_index])
        
        results = clf.predict(homicideDF.iloc[test_index])
        
        #print(results [results != label_test.iloc[test_index]])
        
        allResults.append(metrics.accuracy_score(results, label_test.iloc[test_index]))
        
    print("Accuracy is,", np.mean(allResults))
    
    scaler = StandardScaler()
    homicideDF=  scaler.fit_transform(homicideDF)
    
main()