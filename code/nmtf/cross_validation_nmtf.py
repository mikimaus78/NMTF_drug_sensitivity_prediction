"""
Run the cross-validation parameter search for the NMTF class, on the Sanger
dataset.

Since we want to find co-clusters of significantly higher/lower drug sensitivity
values, we should use the unstandardised Sanger dataset.
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/drug_sensitivity/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")

import numpy, itertools, random
from nmtf_i_div.code.nmtf import NMTF
from ml_helpers.code.matrix_cross_validation import MatrixCrossValidation
from ml_helpers.code.parallel_matrix_cross_validation import ParallelMatrixCrossValidation
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger


# Settings
standardised = False
train_config = {
    'max_iterations' : 10000,
    'updates' : 1,
    'epsilon_stop' : 0.00001,
    'Kmeans' : True,
    'S_first' : True,
    'M_test' : None
}
K_range = [10,20,30,40,50,60,70,80]#[5,10,15,20,25,30,35,40,45,50]
L_range = [5,10,15,20,25,30]#[5,10,15,20,25]
no_folds = 5
output_file = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/crossval_nmtf_%s.txt" % ("std" if standardised else "notstd")

# Construct the parameter search
parameter_search = [{'K':K,'L':L} for (K,L) in itertools.product(K_range,L_range)]

# Load in the Sanger dataset
(_,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)

# Run the cross-validation framework
random.seed(0)
crossval = ParallelMatrixCrossValidation(
    method=NMTF,
    X=X_min,
    M=M,
    K=no_folds,
    parameter_search=parameter_search,
    train_config=train_config,
    file_performance=output_file,
    P=4
)
crossval.run()
crossval.find_best_parameters(evaluation_criterion='MSE',low_better=True)