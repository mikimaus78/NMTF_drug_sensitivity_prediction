"""
Load in the drug sensitivity dataset.
Take out 10% of the values.
Use Penalized Non-Negative Matrix Tri-Factorisation (I-divergence implementation) 
to predict the missing values.

Performance (MSE) on X_min dataset:
  
  
  
  
Observations:
 -  We want to pick alpha and beta in such a way that the I-divergence cost
    it brings does not massively outweigh the normal I-divergence (after convergence - ~2200).
    e.g. for K=L=1:             Start                           End    
    alpha=1, beta=1             1539465.09943, 68265.1008543    1119.41437184, 2237.95301079
    alpha=0.01, beta=0.01       15394.6509943 682.651008543     1043.89637122 1735.35712107
    alpha=0.0001, beta=0.002    153.946509943 136.530201709     148.233811672 137.125113322
  
"""


# Settings
standardised = True
use_kmeans = True
S_first = True
iterations = 100
updates = 1
K = 10
L = 5

# Multiply the [0,1] kernels by this to reward/punish clustering more/less
# 622 cell lines, 139 drugs, I-div=2200, values are normally ~1 in the kernels
alpha = 0.01
beta = 0.1



import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/drug_sensitivity/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")
import numpy, itertools, random
from pnmtf_i_div.code.pnmtf import PNMTF
import ml_helpers.code.mask as mask
import ml_helpers.code.statistics as statistics
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger, load_kernels



# Try each of the values for k in the list <k_values>, and return the performances.
def try_different_k(X,M,no_folds,K_L_values,C1,C2,iterations,updates):
    mean_MSEs = []
    mean_i_divs = []
    for K,L in K_L_values:
        print "Running cross-validation with value K=%s, L=%s." % (K,L)
        MSEs,i_divs = run_cross_validation(X,M,no_folds,K,L,C1,C2,iterations,updates)
        mean_MSEs.append(sum(MSEs) / float(len(MSEs)))
        mean_i_divs.append(sum(i_divs) / float(len(i_divs)))
        
    return (mean_MSEs,mean_i_divs)
    
# Run NMTF on different folds    
def run_cross_validation(X,M,no_folds,K,L,C1,C2,iterations,updates):
    (I,J) = M.shape
    folds_M = mask.compute_folds(I,J,no_folds,M)
    Ms = mask.compute_Ms(folds_M)
    assert_no_empty_rows_columns(Ms)
    
    MSEs = []
    i_divs = []
    #for fold in range(0,no_folds):
    for fold in [0]:
        print "Fold %s for k=%s,l=%s." % (fold+1,K,L)
        M_training = Ms[fold]
        M_test = folds_M[fold]
        assert numpy.array_equal(M_training+M_test, M)
        
        (MSE,i_div) = run_NMTF(X,M_training,M_test,K,L,C1,C2,iterations,updates)
        MSEs.append(MSE)
        i_divs.append(i_div)
        
    return (MSEs,i_divs)

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
def run_NMTF(X,M_training,M_test,K,L,C1,C2,iterations,updates):
    nmtf = PNMTF(X,M_training,K,L,C1,C2)
    nmtf.initialise(use_kmeans)
    nmtf.run(iterations,updates,S_first)
    X_pred = nmtf.R_pred
    
    # Calculate MSE of predictions.
    MSE = statistics.MSE(X,X_pred,M_test)
    i_div = statistics.i_div(X,X_pred,M_test)
    
    print "Performance on test set: MSE=%s, I-div=%s." % (MSE,i_div)    
    return (MSE,i_div)



location_kernels = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/data/kernels/"

     
if __name__ == "__main__":
    random.seed(0)    
    
    """ Load in data. """
    (X,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
    
    C1 = load_kernels(location_kernels,["copy_variation","gene_expression","mutation"]) #cell lines
    C2 = load_kernels(location_kernels,["1d2d_descriptors","PubChem_fingerprints","targets"]) #drugs
        
    # Do kernel-1 to give rewards rather than punishments, but ensure the diagonals are still 0's
    C1 = numpy.subtract(C1,1)
    C2 = numpy.subtract(C2,1)
    for c1 in C1:
        numpy.fill_diagonal(c1,0)
    for c2 in C2:
        numpy.fill_diagonal(c2,0)
    
    # Multiply the kernels by alpha, beta
    C1 = numpy.multiply(C1,alpha)
    C2 = numpy.multiply(C2,beta)
    
    # We can also treat negative values as missing values. 
    X_filtered = numpy.array([[0 if v < 0 else v for v in row] for row in X])
    M_filtered = numpy.array([[0 if (v < 0 or not M[i][j]) else 1 for j,v in enumerate(row)] for i,row in enumerate(X)])
    
    """ Run NMTF cross-validation for different K's, L's """
    no_folds = 5
    
    (MSEs,i_divs) = run_cross_validation(X_min,M,no_folds,K,L,C1,C2,iterations,updates)
    print sum(MSEs)/float(len(MSEs))