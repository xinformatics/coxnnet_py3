from cox_nnet import *
import numpy
import sklearn
import sklearn.model_selection
import pandas as pd

#################################################################### PBC ##########################
# d_path = 'PBC/'

# # load data
# x = numpy.loadtxt(fname=d_path+"x.csv",delimiter=",",skiprows=0)
# ytime = numpy.loadtxt(fname=d_path+"ytime.csv",delimiter=",",skiprows=0)
# ystatus = numpy.loadtxt(fname=d_path+"ystatus.csv",delimiter=",",skiprows=0)

# filename = 'PBC_model.pkl'
# model    = loadModel(filename)

# print ('model load succesful!')

# x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test =  sklearn.model_selection.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = 1)

# print('Running variable importance')

# varimp = varImportance(model, x_train, ytime_train, ystatus_train)

# print(varimp)

# print('finished!')
##################################################################### 

d_path = 'ssBRCA/'

# load data
x = numpy.loadtxt(fname=d_path+"x.csv",delimiter=",",skiprows=0)
ytime = numpy.loadtxt(fname=d_path+"ytime.csv",delimiter=",",skiprows=0)
ystatus = numpy.loadtxt(fname=d_path+"ystatus.csv",delimiter=",",skiprows=0)

filename = 'ssBRCA_cindex_0.669.pkl'
model    = loadModel(filename)

print ('model load succesful!')

x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test =  sklearn.model_selection.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = 1)

print('Running variable importance')

varimp = varImportance(model, x_train, ytime_train, ystatus_train)

print(varimp)

print('finished!')
