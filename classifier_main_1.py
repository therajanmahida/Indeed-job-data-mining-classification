"""
# -*- coding: utf-8 -*-
Created on Mon Nov 30 20:53:52 2020
@author: rajan mahida

"""

import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB


def classify(test_file):


    train_data = pd.read_csv("ads_combined.csv", encoding='UTF-8')
    train_df = pd.DataFrame(train_data)
    train_df.columns = ['Text', 'Job_Title']



    test_data = pd.read_csv(test_file, encoding='UTF-8')
    test_df = pd.DataFrame(test_data)
    test_df.columns = ["Text"]

    

    X_train = train_df["Text"]
    y_train = train_df["Job_Title"]
    X_test = test_df["Text"]


    count_vectorizer = CountVectorizer()
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_test_counts = count_vectorizer.transform(X_test)

    
    lr_clf = LogisticRegression(solver='liblinear', max_iter=10000)
    dt_clf = DecisionTreeClassifier()
    mnb_clf = MultinomialNB()
    mlp_clf = MLPClassifier(max_iter=50)


    predictors = [('lr_clf', lr_clf),
                  ('dt_clf', dt_clf),
                  ('mnb_clf', mnb_clf),
                  ('mlp_clf', mlp_clf)]

    
    
    VT = VotingClassifier(predictors)
    VT.fit(X_train_counts, y_train)

  

    predictions = VT.predict(X_test_counts)

   
    output = open( 'output.csv', 'w', encoding='UTF-8' )
    writer = csv.writer( output, lineterminator='\n' )
    writer.writerow(['predictions'])

    for prediction in predictions:
        writer.writerow([prediction])


classify("ads_combined - only desc.csv")
