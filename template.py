# C:\Users\manav\Desktop\Manav\college\Year 3\Prog. for DA\Final project python3  
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

"""
Created on a sunny day

@author: ?????
@id: ????
@Cohort: ???
"""



def Task1():
    data = pd.read_csv('humanDetails.csv')
    data.head()
    #remove records with unknown work-class
    data = data[data[' workclass'] != ' ?']
    #clean age column
    data['age'] = data['age'].str.replace('s', '').astype(int)
    #create income column
    data['Income'] = data['Income'].apply(lambda x: ' >50K' if x == '>50K' else '<=50K')   
    #split the data into x and y
    x = data[['native-country', ' workclass', 'age']]
    y = data['Income']
    #split the data into train and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)   
    workclass=data[' workclass'].unique()
    labels={}
    for w in range(len(workclass)):
        labels[workclass[w]]=w
    data[' workclass']=data[' workclass'].map(lambda x: labels[x])


    #create empty dictionary to store results
    results = {}

    #loop through each country 
    for country in data['native-country'].unique():
        #create a new dataframe for each country
        df = data.loc[data['native-country'] == country]
        if len(df)>1:
            #split the data into x and y
            x = df[[' workclass', 'age']]
            y = df['Income']
            
            #split the data into train and test set
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)
            
            #perform k-fold cross validation
            kf = KFold(n_splits=5)
            scores = []
            depths = []
            
            #loop through range of depths
            for depth in range(1, 20):
                #create decision tree classifier
                clf = DecisionTreeClassifier(max_depth=depth)
                
                #loop through each fold
                for train_index, test_index in kf.split(x):
                    #split the data into train and test
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    
                    #fit the classifier
                    clf.fit(x_train, y_train)
                    
                    #evaluate the classifier
                    score = accuracy_score(y_test, clf.predict(x_test))
                    scores.append(score)
                    depths.append(depth)
            #find the best depth for the classifier
            best_depth = depths[np.argmax(scores)]
            
            #store the results in the dictionary
            results[country] = {'depth': best_depth, 'score': np.max(scores)}
        else:
            pass
        

    #find countries that have a score gap larger than 20%
    overfit_countries = [c for c in results.keys() if results[c]['score'] - np.mean(scores) > 0.2]


    #plot the scores
    plt.figure(figsize=(20,10))
    plt.bar(range(len(results)), [x['depth'] for x in results.values()], align='center')
    for i in range(len(results)):
        # plt.text(i,.4,'score',rotat)
        plt.annotate("Score:{}".format(list(results.values())[i]['score']),xy=(i-.3,0.4),rotation=90,color='white')
    plt.xticks(range(len(results)), list(results.keys()))
    plt.title('Accuracy scores of countries with overfitting')
    plt.xlabel('Country')
    plt.ylabel('Depth')
    plt.xticks(rotation=90)
    plt.show()




def Task2():
    data = pd.read_csv('humanDetails.csv')
    # a) Fill the unknown cells in the Occupation column with the Occupation with the highest frequency
    occupation_unique, occupation_count = np.unique(data['occupation '], return_counts=True)
    data['occupation '] = np.where(data['occupation '] == ' ?', occupation_unique[occupation_count.argmax()], data['occupation '])
    # b) Clean the values in age column
    data['age'] = data['age'].str.replace('s', '')
    data['age'] = pd.to_numeric(data['age'], errors='coerce')
    # c) Remove Other-relative from the dataset
    data = data[data['relationship'] != ' Other-relative']
    # d) Remove any values in hours-per-week mentioned only once
    hours_unique, hours_count = np.unique(data['hours-per-week'], return_counts=True)
    hours_discarded = [hours_unique[i] for i in range(len(hours_unique)) if hours_count[i] == 1]
    for i in hours_discarded:
        data = data[data['hours-per-week'] != i]
    # e) Convert the income attribute to two unique values
    data['Income'] = np.where(data['Income'] == ' <=50K', 0, 1)
    cols=['occupation ','relationship']
    for c in cols:
        workclass=data[c].unique()
        labels={}
        for w in range(len(workclass)):
            labels[workclass[w]]=w
        data[c]=data[c].map(lambda x: labels[x])
    # Split the data into features and target
    X = data[['hours-per-week', 'occupation ', 'age', 'relationship']].values
    y = data['Income'].values
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    # Applying Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculating the Decision Tree accuracy score
    dt_score = accuracy_score(y_test, y_pred)
    print('Decision Tree score:', dt_score)
    # Applying K Nearest Neighbour Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Calculating the K Nearest Neighbour accuracy score
    knn_score = accuracy_score(y_test, y_pred)
    print('K Nearest Neighbour score:', knn_score)
    # Applying K-Fold Cross Validation with 5 folds
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, X, y, cv=5)

    # Printing the scores
    print('K-Fold Cross Validation scores:', scores)
    print('K-Fold Cross Validation mean scores:', np.mean(scores))
    # Visualizing the results
    sns.barplot(x=['Decision Tree','K Nearest Neighbour','K-Fold Cross Validation'], y=[dt_score, knn_score, scores.mean()])

# The results show that the Decision Tree classifier has the highest accuracy score with an accuracy
#  of 0.79. The K Nearest Neighbour classifier has a same accuracy score of 0.79. The K-Fold Cross
#  Validation has an average accuracy score of 0.78. This suggests that the Decision Tree classifier
#  and KNN are the most effective at predicting income based on the given features.
   
    

def Task3():
    #importing the dataset
    data = pd.read_csv('humanDetails.csv')
    #cleaning the age column
    data['age'] = data['age'].apply(lambda x: int(x[:-1]))
    #selecting the required columns
    X = data[['age','fnlwgt','education-num','hours-per-week']]
    #clustering the data into two clusters
    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(X)
    labels = kmeans.labels_
    #reducing the number of features to two
    X_new = kmeans.transform(X)
    #visualizing the results
    plt.figure(figsize=(10,5))

    #plotting the first visualization
    plt.subplot(1, 2, 1)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=data['Income'].map({' <=50K':'green', ' >50K':'red'}))
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Income based visualization')
    #plotting the second visualization
    plt.subplot(1, 2, 2)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=labels)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Cluster based visualization')

    plt.show()
# The first visualization shows that most individuals with a low income and high income are not 
# clustered together, while they are spread out. The second visualization shows that the clusters
#  created by the K-Means algorithm are not necessarily related to the income of the individuals.
#  This suggests that the other features in our dataset, such as age, fnlwgt, education-num and 
#  hours-per-week, are more important in determining the clusters than income.
    
    


    
    
