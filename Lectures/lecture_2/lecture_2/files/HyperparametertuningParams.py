# Hyperparametertuning parameters for regressors

def linear():
    dictHPT = {
              "normalize"         : [True, False]
            }
    return dictHPT

def lasso():
    dictHPT = {
              "alpha"             : [0.01,0.02, 0.05, 0.1],
              "normalize"         : [True, False],
              "max_iter"          : [5,10,50,100,None],
              "selection"         : ['cyclic', 'random']
             }
    return dictHPT
    
def ridge():
    dictHPT = {
              "alpha"             : [0.01,0.1,1,10,15,20,50],
              "normalize"         : [True, False],
              "solver"            : ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
              "max_iter"          : [10,50,100,None]
             }
    return dictHPT

def dtTree ():
    dictHPT = {
              "min_samples_split" : [5,10,20,30],
              "max_leaf_nodes"    : [5,10,100,150],
              "criterion"         : ['mse', 'friedman_mse', 'mae', 'poisson'],
              "max_depth"         : [2,4,8,12]
             }
    return dictHPT

def rForest():
    dictHPT = {
              'bootstrap'         : [True,False],
              'max_depth'         : [2,10,20,50],
              'min_samples_leaf'  : [2, 5],
              'min_samples_split' : [5, 10, 20,30],
              'n_estimators'      : [50,100, 300, 500]
             }
    return dictHPT

def XGB():
    dictHPT = {
              "n_estimators"      : [50],
              "learning_rate"     : [0.01,0.05,0.1] ,
              "max_depth"         : [ 5,15],
              "min_child_weight"  : [ 3,5,8 ],
              "gamma"             : [ 0.0,0.5]
             }
    return dictHPT

def Ada():
    dictHPT = {
              "n_estimators"      : [50, 100,300],
              "learning_rate"     : [0.005,0.01,0.05,0.1],
              "loss"              : ['linear', 'square', 'exponential'],
              "base_estimator__max_depth"          : [2,4,8,12,30],
              "base_estimator__min_samples_split"  : [2,4,6]
             }
    return dictHPT

def KNNregr():
    dictHPT = {
              "n_neighbors"       : [2,5,12,30],
              "leaf_size"         : [3,15,30,40],
              "p"                 : [1,2],
              "weights"           : ['uniform', 'distance'],
              "metric"            : ['euclidean','manhattan','minkowski'] 
             }
    return dictHPT


