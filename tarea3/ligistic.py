# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:53:59 2020

@author: iscca
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import scipy.io
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
sns.set()

def clasificar(datos,clases):
    label1 = np.where(clases==1)
    label2 = np.where(clases==2)
    result1 = datos[label1[0],:]
    result2 = datos[label2[0],:]    
    
    return result1, result2

mat = scipy.io.loadmat('datos_wdbc.mat')
trn = mat['trn']
clas = trn['y'][0,0]
xc = trn['xc'][0,0]
xd = trn['xd'][0,0]

continuos = pd.DataFrame(data=xc)



model = linear_model.LogisticRegression()
model.fit(continuos,clas)

predictions = model.predict(continuos)

number = model.score(continuos,clas)
print('Model score:')

name = 'Logistic Regression'
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(continuos, clas, test_size=.30, random_state=seed)
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))