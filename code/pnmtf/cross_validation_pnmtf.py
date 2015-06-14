"""
Run the cross-validation parameter search for the PNMTF class, on the Sanger
dataset. We fix the value of K and L, and search for alpha and beta.

Since we want to find co-clusters of significantly higher/lower drug sensitivity
values, we should use the unstandardised Sanger dataset.

For alpha we use the values [0,0.01,0.1,1,10,100].
For beta we use the values [0,0.01,0.1,1,10,100].
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/drug_sensitivity/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")

import numpy, itertools, random
from pnmtf_i_div.code.pnmtf import PNMTF
from ml_helpers.code.matrix_cross_validation import MatrixCrossValidation
from ml_helpers.code.parallel_matrix_cross_validation import ParallelMatrixCrossValidation
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger, load_kernels, negate_Sanger


# Settings
standardised = False
train_config = {
    'max_iterations' : 10000,
    'updates' : 1,
    'epsilon_stop' : 0.0001,#0.00001,#
    'stop_validation' : True,
    'Kmeans' : True,
    'S_random' : True,
    'S_first' : True,
    'M_test' : None,
    'Rpred_Skl' : False
}
K = 20
L = 30
alpha_range = [1,10,100]#[0, 0.01, 0.1, 1, 10, 100]
beta_range = [0, 0.01, 0.1, 1, 10, 100]
no_folds = 5

random.seed(1)   
numpy.random.seed(1)

output_file = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/cross_validation_pnmtf/crossval_pnmtf_%s.txt" % ("std" if standardised else "notstd")
location_kernels = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/data/kernels/"

# Load in the Sanger dataset
(X,_,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
X_min = negate_Sanger(X,M)

# Load in the kernels
C1 = load_kernels(location_kernels,["copy_variation","gene_expression","mutation"]) #cell lines
C2 = load_kernels(location_kernels,["1d2d_descriptors","PubChem_fingerprints","targets"]) #drugs
        
#C1 = numpy.subtract(C1,1) #Give rewards for clustering similar things (negative values)
#C2 = numpy.subtract(C2,1)
# Ensure the diagonals are always 0's
for c1 in C1:
    numpy.fill_diagonal(c1,0)
for c2 in C2:
    numpy.fill_diagonal(c2,0)
    
# Construct the parameter search
parameter_search = [{'K':K,'L':L,'C1':numpy.multiply(C1,alpha),'C2':numpy.multiply(C2,beta)} for (alpha,beta) in itertools.product(alpha_range,beta_range)]

# Run the cross-validation framework
random.seed(0)
crossval = ParallelMatrixCrossValidation(
    method=PNMTF,
    X=X_min,
    M=M,
    K=no_folds,
    parameter_search=parameter_search,
    train_config=train_config,
    file_performance=output_file,
    P=5
)
crossval.run()
crossval.find_best_parameters(evaluation_criterion='MSE',low_better=True)