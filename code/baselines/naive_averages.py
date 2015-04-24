"""
Simply use the row, column, and overall average as the predictions for 
missing values.

Performance: 
    Row average:        11.3361332734
    Column average:     3.59137002826
    Overall average:    11.6961916386
    
Standardised:
    Row average:        0.859709985449
    Column average:     1.00370736585
    Overall average:    1.00003145726
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")
import numpy, itertools
import ml_helpers.code.mask as mask
import ml_helpers.code.statistics as statistics
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger


# Settings
standardised = True
seed = 42


# Method for predicting on all folds. Return a list of MSE's. f is a function
# that runs the predictor (row/column/overall average) on the fold.
def run_cross_validation(X,M,no_folds,seed,f):
    (I,J) = M.shape
    folds_M = mask.compute_folds(I,J,no_folds,seed,M)
    Ms = mask.compute_Ms(folds_M)
    assert_no_empty_rows_columns(Ms)
    
    MSEs = []
    for fold in range(0,no_folds):
        print "Fold %s." % (fold+1)
        M_training = Ms[fold]
        M_test = folds_M[fold]
        assert numpy.array_equal(M_training+M_test, M)
        
        MSE = f(X,M_training,M_test)
        MSEs.append(MSE)
        
    return MSEs
    
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
    return MSE

def f_column(X,M_training,M_test):
    # Compute the row averages of M_training
    (I,J) = M_training.shape
    omega_J = mask.nonzero_column_indices(M_training)
    averages = [sum([X[i][j] for i in omega_J[j]])/float(len(omega_J[j])) for j in range(0,J)]
                
    X_pred = numpy.array([[averages[j] for j in range(0,J)] for i in range(0,I)])
    MSE = statistics.MSE(X,X_pred,M_test)
    return MSE
    
def f_overall(X,M_training,M_test):
    # Compute the row averages of M_training
    (I,J) = M_training.shape
    omega = mask.nonzero_indices(M_training)
    average = sum([X[i][j] for (i,j) in omega])/len(omega)
                
    X_pred = numpy.array([[average for j in range(0,J)] for i in range(0,I)])
    MSE = statistics.MSE(X,X_pred,M_test)
    return MSE
    
    
if __name__ == "__main__":
    (X,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
    no_folds = 5
    
    row_MSEs = run_cross_validation(X,M,no_folds,seed,f_row)
    column_MSEs = run_cross_validation(X,M,no_folds,seed,f_column)
    overall_MSEs = run_cross_validation(X,M,no_folds,seed,f_overall)
    
    print "Row average: %s." % (sum(row_MSEs)/float(len(row_MSEs)))
    print "Column average: %s." % (sum(column_MSEs)/float(len(column_MSEs)))
    print "Overall average: %s." % (sum(overall_MSEs)/float(len(overall_MSEs)))