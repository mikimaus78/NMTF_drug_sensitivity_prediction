"""
Load in the drug sensitivity dataset.
Take out 10% of the values.
Use Non-Negative Matrix Factorisation (I-divergence implementation) to 
predict the missing values.
"""

import numpy, sys, itertools, math, matplotlib.pyplot as plt
sys.path.append("/home/tab43/Documents/Projects/libraries/")
from nmf_i_div.code.nmf import NMF
import ml_helpers.code.mask as mask
import ml_helpers.code.statistics as statistics

dataset_location = "/home/tab43/Documents/Projects/drug_sensitivity/NMTF_drug_sensitivity_prediction/data/Sanger_drug_sensivitiy/"
file_name = "ic50_excl_empty.txt"


# Try each of the values for k in the list <k_values>, and return the performances.
def try_different_k(X,M,no_folds,K_values,seed,iterations,updates):
    mean_MSEs = []
    mean_i_divs = []
    for K in K_values:
        print "Running cross-validation with value K=%s." % K
        MSEs,i_divs = run_cross_validation(X,M,no_folds,K,seed,iterations,updates)
        mean_MSEs.append(sum(MSEs) / float(len(MSEs)))
        mean_i_divs.append(sum(i_divs) / float(len(i_divs)))
        
    return (mean_MSEs,mean_i_divs)
    
# Run NMF on different folds    
def run_cross_validation(X,M,no_folds,K,seed,iterations,updates):
    folds_M = mask.compute_folds(I,J,no_folds,seed,M)
    Ms = mask.compute_Ms(folds_M)
    assert_no_empty_rows_columns(Ms)
    
    MSEs = []
    i_divs = []
    for fold in range(0,no_folds):
        print "Fold %s for k=%s." % (fold+1,K)
        M_training = Ms[fold]
        M_test = folds_M[fold]
        assert numpy.array_equal(M_training+M_test, M)
        
        (MSE,i_div) = run_NMF(X,M_training,M_test,K,iterations,updates)
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
    

# Run NMF on the training data X, with known values <M_training> and test entries <M_test>.
def run_NMF(X,M_training,M_test,K,iterations,updates):
    nmf = NMF(X,M_training,K)
    nmf.initialise()
    nmf.run(iterations,updates,True)
    X_pred = nmf.V_pred
    
    # Calculate MSE of predictions.
    MSE = statistics.MSE(X,X_pred,M_test)
    i_div = statistics.i_div(X,X_pred,M_test)
    
    print "Performance on test set: MSE=%s, I-div=%s." % (MSE,i_div)    
    return (MSE,i_div)

     
if __name__ == "__main__":
    """ Load in data. We get a masked array, and set masked values to 0. """
    data = numpy.genfromtxt(dataset_location+file_name,dtype=str,delimiter="\t",usemask=True)
    
    drug_names = data[0,3:]
    cell_lines = data[1:,0]
    cancer_types = data[1:,1]
    tissues = data[1:,2]
    
    X = data[1:,3:] #numpy.array(data[1:,3:],dtype='f')
    M = mask.calc_inverse_M(numpy.array(X.mask,dtype=float))
    (I,J) = X.shape # 2200 drugs, 60 cancer cell lines
    
    """ For missing values we place 0. Our method requires non-negative values so we transform the values.
        Exponential transform gives horrible performance, so we simply subtract the min from all values. """
    X = numpy.array([[v if v else '0' for v in row] for row in X],dtype=float) #set missing values to 0
    minimum = X.min()-1
    X_min = numpy.array([[v-minimum if v else '0' for v in row] for row in X],dtype=float)
    
    # We can also treat negative values as missing values. 
    X_filtered = numpy.array([[0 if v < 0 else v for v in row] for row in X])
    M_filtered = numpy.array([[0 if (v < 0 or not M[i][j]) else 1 for j,v in enumerate(row)] for i,row in enumerate(X)])
    
    
    """ Run NMF cross-validation for different K's """
    no_folds = 5
    seed = 0
    iterations = 5
    updates = 1
    Ks = range(1,50+1)
    #print run_cross_validation(X_min,M,no_folds,10,seed,iterations,updates)
    #print run_cross_validation(X_filtered,M_filtered,no_folds,10,seed,iterations,updates)
    
    (MSEs,i_divs) = try_different_k(X_min,M,no_folds,Ks,seed,iterations,updates)
    (MSEs_filtered,i_divs_filtered) = try_different_k(X_filtered,M_filtered,no_folds,Ks,seed,iterations,updates)
    
    """
    Note: MSE and i_div for X_min and X_filtered stay roughly the same for 
          different values of K (MSE of ~3.15 and ~2.1 resp.).
    TODO: Try different initialisation to see whether we get a better fit.
    TODO: Could try preprocessing (e.g. all values to a range [0,10]). Shouldn't make much of a difference.
    """    
    
    """ Plot the performances with varying K's """
    fig = plt.figure(0)
    plt.plot(Ks,MSEs)
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