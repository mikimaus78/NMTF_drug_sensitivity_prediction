"""
Script for standardising the drug sensitivity values for each cancer cell 
line to mean 0 and standard deviation 1: v_new = (v_old - mean) / std
We store these values in a file
of the format:
    cell_line_name      mean        std
So that new predictions can be transformed back using:
    pred_value * std + mean
"""

import load_data
import numpy

location_folder_Sanger = "/home/thomas/Dropbox/Biological databases/Sanger_drug_sensivitity/"
original_Sanger = location_folder_Sanger+"ic50_excl_empty_filtered_cell_lines_drugs.txt"
new_Sanger = location_folder_Sanger+"ic50_excl_empty_filtered_cell_lines_drugs_standardised.txt"


def standardise_Sanger(location):
    (X,_,M,drug_names,cell_lines,cancer_types,tissues) = load_data.load_Sanger()
    
    mean_columns = []
    std_columns = []
    (no_rows,no_columns) = X.shape
    for column in range(0,no_columns):
        known_values = numpy.array([X[row,column] for row in range(0,no_rows) if M[row,column]])
        mean_columns.append(known_values.mean())
        std_columns.append(known_values.std())
        
    X_standardised = X - [mean_columns for i in range(0,no_rows)]
    X_standardised = X_standardised / [std_columns for i in range(0,no_rows)]
    
    load_data.store_Sanger(
        location=new_Sanger,
        X=X_standardised,
        M=M,
        drug_names=drug_names,
        cell_lines=cell_lines,
        cancer_types=cancer_types,
        tissues=tissues
    )
    
    
if __name__ == "__main__":
    standardise_Sanger(original_Sanger)