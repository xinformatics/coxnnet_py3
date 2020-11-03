#works for 

# pandas==0.25.1
# numpy==1.17.2
# scikit-learn==0.21.3
# Theano==1.0.4
# tqdm==4.36.1
# pickle==4.0 / alternatively use dill bcz pickle may not be able to serialize lambda objects 
######################
#machine specs: 
#OS -  Ubuntu 18.04.3 LTS
#Python 3.7.4


####################
from cox_nnet import *
import numpy
import sklearn
import sklearn.model_selection
import pandas as pd


d_path = "LUAD/mrna/"

# load for mirna data
x = numpy.loadtxt(fname=d_path+"data.csv",delimiter=",",skiprows=0)
ytime = numpy.loadtxt(fname=d_path+"ytime.csv",delimiter=",",skiprows=0)
ystatus = numpy.loadtxt(fname=d_path+"ystatus.csv",delimiter=",",skiprows=0)
#########    ROWS should be equal (no. of patients), in data files columns should be the expression values  ######################
#print ('Data Loaded')

############## with L2 search and model train ########################################################################


# split into test/train sets

x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = \
    sklearn.model_selection.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = 1)

# split training into optimization and validation sets
x_opt, x_validation, ytime_opt, ytime_validation, ystatus_opt, ystatus_validation = \
    sklearn.model_selection.train_test_split(x_train, ytime_train, ystatus_train, test_size = 0.2, random_state = 1)

# set parameters
model_params = dict(node_map = None, input_split = None)
search_params = dict(method = "nesterov", learning_rate=0.01, momentum=0.9, 
    max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
    rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0)


# ############################ regularixation parameter search
# cv_params = dict(cv_metric = "cindex", L2_range = numpy.arange(-1,2,0.33))

# #print ('Finding L2 parameter')
# print ('Finding L2 parameter')
# #profile log likelihood to determine lambda parameter
# likelihoods, L2_reg_params = L2Profile(x_opt,ytime_opt,ystatus_opt,
#     x_validation,ytime_validation,ystatus_validation,
#     model_params, search_params, cv_params, verbose=False)

# #numpy.savetxt(d_path+"LUAD_cindex.csv", likelihoods, delimiter=",")

# #build model based on optimal lambda parameter
# L2_reg = L2_reg_params[numpy.argmax(likelihoods)]
#############################################################################

L2_reg = 0.67 # for mrna data
#L2_reg = 1 # for mrna_clinical data
#L2_reg = -1 # for meth data
#L2_reg = -0.67 # for meth_clinical data
#L2_reg = -1.5 # for mirna data
#L2_reg = -1 # for mirna_clinical data
#L2_reg = 1.31 # for mrna_meth data
#L2_reg = 1.31 # for mrna_meth_clinical data
#L2_reg = _ # for mrna_mirna data
#L2_reg = _ # for mrna_mirna_clinical data
#L2_reg = 1.3 # for all omics only data
#L2_reg = 0.3 # for all omics + clinical data
############################################################################


print('best L2 value is: ', numpy.exp(L2_reg))
model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
print ('Training Cox NN')
model, cost_iter = trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)

theta = model.predictNewData(x_test)

print ('c-index', CIndex(model, x_test, ytime_test, ystatus_test))

#numpy.savetxt(d_path+"LUAD_theta.csv", theta, delimiter=",")
#numpy.savetxt(d_path+"LUAD_ytime_test.csv", ytime_test, delimiter=",")
#numpy.savetxt(d_path+"LUAD_ystatus_test.csv", ystatus_test, delimiter=",")


########## saving model ###########

#saveModel(model, 'LUAD_model.pkl')

######### loading model ############

########## var importance ############

print('Running variable importance')

varImportance(model, x_train, ytime_train, ystatus_train)

print('finished!')


############################################## multiple BATCH RUNs bELOW

# list_of_cindex = []

# for i in range(1,16):

# 	x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = \
# 	    sklearn.model_selection.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = i)

# 	# split training into optimization and validation sets
# 	x_opt, x_validation, ytime_opt, ytime_validation, ystatus_opt, ystatus_validation = \
# 	    sklearn.model_selection.train_test_split(x_train, ytime_train, ystatus_train, test_size = 0.2, random_state = 1)

# 	# set parameters
# 	model_params = dict(node_map = None, input_split = None)
# 	search_params = dict(method = "nesterov", learning_rate=0.01, momentum=0.9, 
# 	    max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
# 	    rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0)



# 	#L2_reg = 0.67 # for mrna data
# 	#L2_reg = 1.0 # for mrna_clinical data
# 	#L2_reg = -1 # for meth data
# 	#L2_reg = -0.67 # for meth_clinical data
# 	#L2_reg = -1.5 # for mirna data
# 	#L2_reg = -1 # for mirna_clinical data
# 	#L2_reg = 1.31 # for mrna_meth data
# 	#L2_reg = 1.31 # for mrna_meth_clinical data
# 	L2_reg = 1.31 # for mrna_mirna data
# 	#L2_reg = 1.31 # for mrna_mirna_clinical data
# 	#L2_reg = 1.3 # for all omics only data
# 	#L2_reg = 0.3 # for all omics_clinical data

# 	print('best L2 value is: ', numpy.exp(L2_reg))
# 	model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
# 	print ('Training Cox NN')
# 	model, cost_iter = trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)

# 	#theta = model.predictNewData(x_test)

# 	#print ('c_index: ',  concordance_index(ytime_test, theta, ystatus_test))
# 	list_of_cindex.append(CIndex(model, x_test, ytime_test, ystatus_test))
# 	print ('random_state_', i ,' c-index ', CIndex(model, x_test, ytime_test, ystatus_test))

# print (list_of_cindex)