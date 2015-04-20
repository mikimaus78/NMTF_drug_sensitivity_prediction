"""
Load in the drug sensitivity dataset.
Take out 10% of the values.
Use Non-Negative Matrix Tri-Factorisation (I-divergence implementation) to 
predict the missing values.

Performance (MSE) on X_min dataset:
    
Standard initialisation:
- K=L=1  -> 3.14475
- K=L=5  -> 3.14420747411
- K=L=10 -> 3.14418096438

Kmeans initialisation (S to all 1's):
            5 iterations    10 iterations   20 iterations   100 iterations  1000 iterations
- K=L=1  -> 3.14475         -               -               -
- K=L=5  -> 3.08460959049   3.0532899867    3.03879916012   3.02687957842   2.88608055993
- K=L=10 -> 3.11872624757   

Kmeans initialisation (S randomly from [0,1]):
            5 iterations    10 iterations   20 iterations   100 iterations
- K=L=1  -> 3.14475227179   -               -               -
- K=L=5  -> 14.2546620532   7.99169398269   4.80329188879   3.09846431001 (1st fold only)

S before F and G, Kmeans initialisation (S to all 1's):
            5 iterations    10 iterations   20 iterations   100 iterations  1000 iterations
- K=L=1  -> 3.14475227179   -               -               -
- K=L=5  -> 3.87658046398   3.39345862811   3.11654697979   3.02607177499   2.85277569876

            2000 iterations 5000 iterations 10000 iterations
- K=L=2  -> 
- K=L=3  -> 

Findings:

Using the K-means initialisation offers a good performance
increase over the regular initialisation; and running more iterations will
give a better fit as well.

Using Kmeans with S initialised randomly to [0,1] gives very slow convergence:
only after <> iterations do we reach the same performance as initialising S 
to all 1's.

When doing the updates to F and G before S, G ~= [0,2], and F ~= [0,20+].
When doing the updates to S before F and G, G ~= [0,1.5], and F ~= [0,2.3].
So doing S first gives better cluster indicators.
The price for that is slower convergence initially (but same performance at 100 iterations).


Standardised Sanger dataset:
    
S before F and G, Kmeans initialisation (S to all 1's):
            5 iterations    10 iterations   20 iterations   100 iterations  500 iterations  1000 iterations
- K=L=1  -> 0.870371549574  0.870371549574  -               -               -               -
- K=L=5  -> 0.778403335661  0.740796354643  0.707632116702  0.667259433036  0.616872929724  0.59421880759
- K=L=10 -> 0.836234871516  0.80792697128   0.763092721476  

            1000 iterations 2000 iterations 5000 iterations 10000 iterations
- K=L=2  ->   
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/libraries/")
import numpy, itertools, matplotlib.pyplot as plt
from nmtf_i_div.code.nmtf import NMTF
import ml_helpers.code.mask as mask
import ml_helpers.code.statistics as statistics
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger


# Settings
standardised = True
use_kmeans = True
seed_kmeans = 1
S_first = True
iterations = 2000
K = 2
L = 2


# Try each of the values for k in the list <k_values>, and return the performances.
def try_different_k(X,M,no_folds,K_L_values,seed,iterations,updates):
    mean_MSEs = []
    mean_i_divs = []
    for K,L in K_L_values:
        print "Running cross-validation with value K=%s, L=%s." % (K,L)
        MSEs,i_divs = run_cross_validation(X,M,no_folds,K,L,seed,iterations,updates)
        mean_MSEs.append(sum(MSEs) / float(len(MSEs)))
        mean_i_divs.append(sum(i_divs) / float(len(i_divs)))
        
    return (mean_MSEs,mean_i_divs)
    
# Run NMTF on different folds    
def run_cross_validation(X,M,no_folds,K,L,seed,iterations,updates):
    (I,J) = M.shape
    folds_M = mask.compute_folds(I,J,no_folds,seed,M)
    Ms = mask.compute_Ms(folds_M)
    assert_no_empty_rows_columns(Ms)
    
    MSEs = []
    i_divs = []
    for fold in range(0,no_folds):
        print "Fold %s for k=%s,l=%s." % (fold+1,K,L)
        M_training = Ms[fold]
        M_test = folds_M[fold]
        assert numpy.array_equal(M_training+M_test, M)
        
        (MSE,i_div) = run_NMTF(X,M_training,M_test,K,L,iterations,updates)
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
def run_NMTF(X,M_training,M_test,K,L,iterations,updates):
    nmtf = NMTF(X,M_training,K,L)
    nmtf.initialise(use_kmeans,seed_kmeans)
    nmtf.run(iterations,updates,S_first)
    X_pred = nmtf.R_pred
    
    # Calculate MSE of predictions.
    MSE = statistics.MSE(X,X_pred,M_test)
    i_div = statistics.i_div(X,X_pred,M_test)
    
    print "Performance on test set: MSE=%s, I-div=%s." % (MSE,i_div)    
    return (MSE,i_div)

     
if __name__ == "__main__":
    """ Load in data. """
    (X,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised)
    
    # We can also treat negative values as missing values. 
    X_filtered = numpy.array([[0 if v < 0 else v for v in row] for row in X])
    M_filtered = numpy.array([[0 if (v < 0 or not M[i][j]) else 1 for j,v in enumerate(row)] for i,row in enumerate(X)])
    
    """ Run NMTF cross-validation for different K's """
    no_folds = 5
    seed = 0
    updates = 1
    
    (MSEs,i_divs) = run_cross_validation(X_min,M,no_folds,K,L,seed,iterations,updates)
    print sum(MSEs)/float(len(MSEs))
    #print run_cross_validation(X_filtered,M_filtered,no_folds,10,seed,iterations,updates)
    
    #K_L_values = itertools.product(range(1,2+1),range(1,2+1))
    #(MSEs,i_divs) = try_different_k(X_min,M,no_folds,K_L_values,seed,iterations,updates)
    #(MSEs_filtered,i_divs_filtered) = try_different_k(X_filtered,M_filtered,no_folds,K_L_values,seed,iterations,updates)
    #print MSEs,i_divs   
    
    """
    Note: MSE and i_div for X_min and X_filtered stay roughly the same for 
          different values of K (MSE of ~3.15 and ~2.1 resp.).
    """    
    
    '''
    """ Plot the performances with varying K's """
    fig = plt.figure(0)
    plt.plot(K_L_values,MSEs)
    plt.title("MSE for X_min")
    
    fig = plt.figure(1)
    plt.plot(Ks,i_divs)
    plt.title("I-div for X_min")
    
    fig = plt.figure(2)
    plt.plot(Ks,MSEs_filtered)
    plt.title("MSE for X_filtered")
    
    fig = plt.figure(3)
    plt.plot(Ks,i_divs_filtered)
    plt.title("I-div for X_filtered")
    
    plt.show()
    '''