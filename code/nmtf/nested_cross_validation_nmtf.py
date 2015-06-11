"""
Run the nested cross-validation for the NMTF class, on the Sanger dataset.

Since we want to find co-clusters of significantly higher/lower drug sensitivity
values, we should use the unstandardised Sanger dataset.
"""

import sys
sys.path.append("/home/thomas/Documenten/PhD/")#("/home/tab43/Documents/Projects/drug_sensitivity/")
sys.path.append("/home/thomas/Documenten/PhD/libraries/")#("/home/tab43/Documents/Projects/libraries/")

import numpy, itertools, random
from nmtf_i_div.code.nmtf import NMTF
from ml_helpers.code.matrix_nested_cross_validation import MatrixNestedCrossValidation
from NMTF_drug_sensitivity_prediction.code.helpers.load_data import load_Sanger, negate_Sanger


# Settings
standardised = False
train_config = {
    'max_iterations' : 10000,
    'updates' : 1,
    'epsilon_stop' : 0.0,#0.0001,
    'stop_validation' : True,
    'Kmeans' : True,
    'S_random' : True,
    'S_first' : True,
    'M_test' : None,
    'Rpred_Skl' : False
}
K_range = [10,20,30]#[10,20,30,40]#
L_range = [15,20,25,30]#[5,10,15,20,25,30,35,40]#
no_folds = 5
output_file = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/nested_cross_validation_nmtf/results.txt"
files_nested_performances = [
    "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/results/nested_cross_validation_nmtf/fold_%s.txt" % fold for fold in range(1,no_folds+1)]

# Construct the parameter search
parameter_search = [{'K':K,'L':L} for (K,L) in itertools.product(K_range,L_range)]

# Load in the Sanger dataset
(X,_,M,drug_names,cell_lines,cancer_types,tissues) = load_Sanger(standardised=standardised)
X_min = negate_Sanger(X,M)

# Run the cross-validation framework
random.seed(0)
nested_crossval = MatrixNestedCrossValidation(
    method=NMTF,
    X=X_min,
    M=M,
    K=no_folds,
    parameter_search=parameter_search,
    train_config=train_config,
    file_performance=output_file,
    files_nested_performances=files_nested_performances
)
nested_crossval.run()