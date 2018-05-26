from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import urllib.parse

import matplotlib.pyplot as plt

def loadFile(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    with open(filepath,'r') as f:
        data = f.readlines()
    data = list(set(data))
    

badQueries = loadFile('badqueries.txt')
validQueries = loadFile('goodqueries.txt')

badQueries = list(set(badQueries))
validQueries = list(set(validQueries))
allQueries = badQueries + validQueries
yBad = [1 for i in range(0, len(badQueries))]  #labels, 1 for malicious and 0 for clean
yGood = [0 for i in range(0, len(validQueries))]
y = yBad + yGood
queries = allQueries

vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,4)) #converting data to vectors
X = vectorizer.fit_transform(queries)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43) #splitting data

badCount = len(badQueries)
validCount = len(validQueries)

lgs = LogisticRegression(class_weight={1: 2 * validCount / badCount, 0: 1.0}) # class_weight='balanced')
lgs.fit(X_train, y_train) #training our model

# Evaluation #


predicted = lgs.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, (lgs.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)

