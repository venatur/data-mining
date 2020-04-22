# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:07:28 2020

@author: iscca
"""

import numpy as np
from dann_class import DANN
import scipy.io
from scipy import stats
from numpy import linalg as LA
from sklearn.model_selection import train_test_split


def sanear(datos):
    D, V = LA.eig(datos)
    V[V<0] =.001
    tras = np.transpose(V)
    op = (D*V)
    ops = np.dot(op,tras)
    return ops

x = np.zeros(shape=[30,30])

mat = scipy.io.loadmat('Data_Synthetic_1500x100x3C.mat')
trn = mat['trn']
clas = trn['y'][0,0]
xc = trn['xc'][0,0]

C_train, C_test, cl_train, cl_test = train_test_split(xc, clas, test_size=.20, shuffle=False)



danny = DANN()

cosa = danny.fit(C_train,clas,neighborhood_size=11)

predict = danny.predict(C_test[0,:])

xc[0,:]