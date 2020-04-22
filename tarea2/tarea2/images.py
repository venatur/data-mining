# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:51:36 2020

@author: iscca
"""
import io
from PIL import Image 
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from numpy import linalg as LA
import math
def medias(data):

    x = []
    for i in range(data.shape[1]):
        x.append(np.mean(data[:,i]))
        
    return x

def cambiar(data):
    for item in range(len(data)):
        if item[i]==255:
            item[i] = 1

    return data

def LDC(data,median,inversa,priori):
    c= -.5
    dt = data
    media_t = np.transpose(median)
    s1 = np.dot(median,inversa)
    
    suma1 = np.dot(s1,dt)
    
    suma2 = c * np.dot(s1, media_t)
    
    suma3 = math.log(priori)
    
    r = suma1 + suma2 +suma3
    return r

def sanear(datos):
    D, V = LA.eig(datos)
    datos[datos<0] =.001
    tras = np.transpose(V)
    op = (D*V)
    ops = np.dot(op,tras)
    return ops

def clasificar(datos,clases):
    label1 = np.where(clases==0)
    label2 = np.where(clases==1)
    result1 = datos[label1[0],:]
    result2 = datos[label2[0],:]    
    
    return result1, result2

def mod_class(mod1,mod2):
    mod = []
    for j in range(len(mod1)):
        if mod1[j]> mod2[j]:
            mod.append(0)
        else:
            mod.append(1)
            
    return mod

def tasas(mod, cl_test):
    aciertos = []

    for ind in range(len(mod)):
        if mod[ind] == cl_test[ind]:
            aciertos.append(1)
    tasa = (sum(aciertos)/len(cl_test))*100
    return tasa

newl = []
newt = []
clases = [0,0,0,0,1,1,1,1]
for name in glob.glob('circulos_cuadrados/*.jpg'):
    print(name)
    im = Image.open(name)
    thresh = 200
    fn = lambda x : 1 if x > thresh else 0
    im = im.convert('L').point(fn)
        
    tama = np.size(im)
    dimen = tama[0]*tama[1]
    data = np.asarray(im)
    newv = np.matrix.flatten(data,'C')
    newl.append(newv)
    
        
        #im.show()


test = 'circulos_cuadrados/test/*.jpg'
v =3
for name in glob.glob(test):
    print(name)
    im = Image.open(name)
    thresh = 200
    fn = lambda x : 1 if x > thresh else 0
    im = im.convert('L').point(fn)
        
    tama = np.size(im)
    dimen = tama[0]*tama[1]
    data = np.asarray(im)
    newv = np.matrix.flatten(data,'C')
  
    newt.append(newv)

    #im.show()

newl = np.asmatrix(newl)
newt = np.asmatrix(newt)

pca = PCA(n_components=8)
pca = pca.fit(newl)
cosa =pca.components_

mult1 = np.dot(cosa,newl.T)
mult2 = np.dot(cosa,newt.T)

clases = np.array([0,0,0,0,1,1,1,1])

class1 = np.asmatrix(np.where(clases == 0))
class2 = np.asmatrix(np.where(clases == 1))

datos_c1, datos_c2 = clasificar(mult1, clases)

m1 = medias(datos_c1)
m2 = medias(datos_c2)


LDC1 = []
LDC2 = []

sigma = np.cov(newl)
saneado = sanear(sigma)
saneado = np.linalg.inv(saneado)
c, r = class1.shape
c2, r2 = class2.shape
p_clas1 = r/len(clases)
p_clas2 = r2/len(clases)


for i in range(mult2.shape[1]):
    
    Test = mult2[:, i]
    sumaL1 = LDC(Test,m1,saneado,p_clas1)
    LDC1.append(sumaL1)
    sumaL2 = LDC(Test,m2,saneado,p_clas2)
    LDC2.append(sumaL2)

LDC_T = mod_class(LDC1,LDC2)
tasa_LDC = tasas(LDC_T,clases)*v


print("Tasa de reconocimiento\n",tasa_LDC)



