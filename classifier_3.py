# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:51:11 2020

@author: rajan
"""




import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


  
def normalize_sentence(sentence):
    # remove ' and .
    sentence = sentence.replace("'", "")
    sentence = sentence.replace(".", "")
    # change , for space
    sentence = sentence.replace(", ", " ")
    sentence = sentence.replace(",", " ")
    sentence = sentence.replace("/", " ")
    sentence = sentence.lower()
    sentence = sentence.partition("&")[0]
    sentence = sentence.partition(" and ")[0]
    return sentence


# Load tsv file into dataframe
data = pd.read_csv('ads_princeton_datascientist.csv', sep='\t')

positions_categories = pd.Categorical(data["Classification"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["Position"],
    positions_categories.codes,
    train_size=0.85,
    stratify=positions_categories.codes
)

X_train = [normalize_sentence(x) for x in X_train]
X_test = [normalize_sentence(x) for x in X_test]

with open('data_sets/x_train.pkl', 'wb') as file:
    pickle.dump(X_train, file)

with open('data_sets/x_test.pkl', 'wb') as file:
    pickle.dump(X_test, file)

with open('data_sets/y_train.pkl', 'wb') as file:
    pickle.dump(y_train, file)

with open('data_sets/y_test.pkl', 'wb') as file:
    pickle.dump(y_test, file)

with open('data_sets/positions_categories.pkl', 'wb') as file:
    pickle.dump(positions_categories, file)