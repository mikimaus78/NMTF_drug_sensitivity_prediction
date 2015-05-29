"""
Simply use the row, column, and overall average as the predictions for 
missing values.

Performance: 
    Row average: 
        {'R^2': 0.029644933669548325,   'MSE': 11.348631455868887,  'Rp': 0.17923432717792037}.
    Column average: 
        {'R^2': 0.6924921858635702,     'MSE': 3.5957738645991015,  'Rp': 0.83221287015222334}.
    Overall average: 
        {'R^2': -7.272791048196225e-05, 'MSE': 11.696021184479459,  'Rp': -1.4527554718208073e-17}.
    
Standardised:
    Row average: 
        {'R^2': 0.13962335040720558,    'MSE': 0.86040636569319207, 'Rp': 0.37456165576957895}.
    Column average: 
        {'R^2': -0.004933197333114059,  'MSE': 1.0048378932278417,  'Rp': -0.091742344925145242}.
    Overall average: 
        {'R^2': -0.0004141644759969143, 'MSE': 1.0001484747287068,  'Rp': -3.6538130267781734e-18}.

"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")
import numpy, itertools, random
import ml_helpers.code.mask as mask
import ml_helpers.code.statistics as statistics
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger


# Settings
standardised = False


# Method for predicting on all folds. Return a list of MSE's. f is a function
# that runs the predictor (row/column/overall average) on the fold.
def run_cross_validation(X,M,no_folds,f):
    (I,J) = M.shape
    folds_M = mask.compute_folds(I,J,no_folds,M)
    Ms = mask.compute_Ms(folds_M)
    assert_no_empty_rows_columns(Ms)
    
    performances = []
    for fold in range(0,no_folds):
        print "Fold %s." % (fold+1)
        M_training = Ms[fold]
        M_test = folds_M[fold]
        assert numpy.array_equal(M_training+M_test, M)
        
        performance = f(X,M_training,M_test)
        performances.append(performance)
        
    return performances
    
# Test the folds to ensure none has an entirely empty row or column.
def assert_no_empty_rows_columns(Ms):
    for M in Ms:
        (I,J) = M.shape
        OmegaI = [[] for i in range(0,I)]
        OmegaJ = [[] for j in range(0,J)]
        for i,j in itertools.product(range(0,I),range(0,J)):
            if M[i,j]:
                OmegaI[i].append(j)
                OmegaJ[j].append(i)
        for i,omega_i in enumerate(OmegaI):
            assert len(omega_i) != 0, "Fully unobserved row in M, row %s." % i
        for j,omega_j in enumerate(OmegaJ):
            assert len(omega_j) != 0, "Fully unobserved column in M, column %s." % j    
    
    
# The functions f for the row/column/overall average    
def f_row(X,M_training,M_test):
    # Compute the row averages of M_training
    (I,J) = M_training.shape
    omega_I = mask.nonzero_row_indices(M_training)
    averages = [sum([X[i][j] for j in omega_I[i]])/float(len(omega_I[i])) for i in range(0,I)]
    
    X_pred = numpy.array([[averages[i] for j in range(0,J)] for i in range(0,I)])
    
    MSE = statistics.MSE(X,X_pred,M_test)
    R2 = statistics.R2(X,X_pred,M_test)
    Rp = statistics.Rp(X,X_pred,M_test)
    return {'MSE':MSE,'R^2':R2,'Rp':Rp}

def f_column(X,M_training,M_test):
    # Compute the row averages of M_training
    (I,J) = M_training.shape
    omega_J = mask.nonzero_column_indices(M_training)
    averages = [sum([X[i][j] for i in omega_J[j]])/float(len(omega_J[j])) for j in range(0,J)]
    
    X_pred = numpy.array([[averages[j] for j in range(0,J)] for i in range(0,I)])
    
    MSE = statistics.MSE(X,X_pred,M_test)
    R2 = statistics.R2(X,X_pred,M_test)
    Rp = statistics.Rp(X,X_pred,M_test)
    return {'MSE':MSE,'R^2':R2,'Rp':Rp}
    
def f_overall(X,M_training,M_test):
    # Compute the row averages of M_training
    (I,J) = M_training.shape
    omega = mask.nonzero_indices(M_training)
    average = sum([X[i][j] for (i,j) in omega])/len(omega)
                
    X_pred = numpy.array([[average for j in range(0,J)] for i in range(0,I)])
    
    MSE = statistics.MSE(X,X_pred,M_test)
    R2 = statistics.R2(X,X_pred,M_test)
    Rp = statistics.Rp(X,X_pred,M_test)
    return {'MSE':MSE,'R^2':R2,'Rp':Rp}
    
    
# Run the cross-validation
random.seed(0)    

(X,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
no_folds = 5

row_performances = run_cross_validation(X,M,no_folds,f_row)
column_performances = run_cross_validation(X,M,no_folds,f_column)
overall_performances = run_cross_validation(X,M,no_folds,f_overall)

# Find average performances
def find_averages(performances):
    averages = {'MSE':0.0,'R^2':0.0,'Rp':0.0}
    n = float(len(performances))
    for performance in performances:
        for name in performance:
            averages[name] += performance[name]
    for name,value in averages.iteritems():
        averages[name] /= n
    return averages
            
row_averages = find_averages(row_performances)
column_averages = find_averages(column_performances)
overall_averages = find_averages(overall_performances)

print "Row performances: %s." % (row_performances)
print "Row average: %s." % (row_averages)
print "Column performances: %s." % (column_performances)
print "Column average: %s." % (column_averages)
print "Overall performances: %s." % (overall_performances)
print "Overall average: %s." % (overall_averages)