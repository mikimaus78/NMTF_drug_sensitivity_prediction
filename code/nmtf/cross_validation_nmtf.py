"""
Run the cross-validation parameter search for the NMTF class, on the Sanger
dataset.
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/drug_sensitivity/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")

import numpy, itertools, random
from nmtf_i_div.code.nmtf import NMTF
from ml_helpers.code.matrix_cross_validation import MatrixCrossValidation
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger


# Settings
standardised = True
train_config = {
    'max_iterations' : 10000,
    'updates' : 1,
    'epsilon_stop' : 0.00001,
    'Kmeans' : True,
    'S_first' : True,
    'M_test' : None
}
K_range = [1,2,3]#[1,5,10,15,20,25,30,35,40,45,50]
L_range = [1,2]#[1,5,10,15,20,25]
no_folds = 5
output_file = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/crossval_nmtf.txt"

# Construct the parameter search
parameter_search = [{'K':K,'L':L} for (K,L) in itertools.product(K_range,L_range)]

# Load in the Sanger dataset
(_,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)

# Run the cross-validation framework
crossval = MatrixCrossValidation(
    method=NMTF,
    X=X_min,
    M=M,
    K=no_folds,
    parameter_search=parameter_search,
    train_config=train_config,
    file_performance=output_file)
crossval.run()
crossval.find_best_parameters(evaluation_criterion='MSE',low_better=True)