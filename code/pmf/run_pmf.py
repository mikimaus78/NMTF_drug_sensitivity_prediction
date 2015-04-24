"""
Run the variational Probabilistic Matrix Factorisation algorithm on the 
Sanger drug sensitivity dataset, and see what the performance is like for
cross-validation (5 folds).

Performances
Non-negative values:
            5 iterations    10 iterations   20 iterations   100 iterations
K = 1   ->  3.14341020912   3.14237116191   3.14237132098   -
K = 5   ->  2.90927310451   2.52532203189   2.49676575818   2.40232972398
K = 10  ->  2.84899158313   2.3464385451    2.27234725768   2.23122158042
K = 50  ->  3.13621751419   2.4142635075    2.20830485965   2.16682080088

Standardised, non-negative Sanger dataset:
            5 iterations    10 iterations   20 iterations   100 iterations
K = 1   ->  0.870348893168  0.870348894795  -               -
K = 5   ->  0.7936101907    0.621832605327  0.608217217781  0.532586943708
K = 10  ->  0.808557759748  0.502871452276                  0.472640568664

Standardised Sanger dataset:
            5 iterations    10 iterations   20 iterations   100 iterations
K = 1   ->  0.836889624042  0.822344257119  -               -
K = 5   ->  0.648011959179  0.547163374917  0.520122238173  0.520143401769                
K = 10  ->  0.538111699046  
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/libraries")#("/home/tab43/Documents/Projects/libraries/")
import numpy, itertools, matplotlib.pyplot as plt
from variational_pmf.code.variational_pmf import VariationalPMF
import ml_helpers.code.mask as mask
import ml_helpers.code.statistics as statistics
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger


# Settings
standardised = True
negative = False
updates = 1
seed = 42
no_folds = 5
iterations = 100
K = 10



# Run PMF on different folds    
def run_cross_validation(X,M,no_folds,K):
    (I,J) = M.shape
    folds_M = mask.compute_folds(I,J,no_folds,seed,M)
    Ms = mask.compute_Ms(folds_M)
    assert_no_empty_rows_columns(Ms)
    
    MSEs = []
    #for fold in range(0,no_folds):
    for fold in [0]:
        print "Fold %s for k=%s." % (fold+1,K)
        M_training = Ms[fold]
        M_test = folds_M[fold]
        assert numpy.array_equal(M_training+M_test,M)
        
        MSE = run_PMF(X,M_training,M_test,K)
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
    

# Run NMTF on the training data X, with known values <M_training> and test entries <M_test>.
def run_PMF(X,M_training,M_test,K):
    pmf = VariationalPMF(X,M_training,K)
    pmf.initialize()
    pmf.run(iterations,updates)
    X_pred = pmf.predicted_X
    
    # Calculate MSE of predictions.
    MSE = statistics.MSE(X,X_pred,M_test)
    
    print "Performance on test set: MSE=%s." % MSE  
    return MSE



if __name__ == "__main__":
    """ Load in data. """
    (X,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
    
    if negative:
        data = X
    else:
        data = X_min
    MSEs = run_cross_validation(data,M,no_folds,K)
    print sum(MSEs)/float(len(MSEs))