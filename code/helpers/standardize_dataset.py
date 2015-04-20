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

location_folder_Sanger = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/data/Sanger_drug_sensivity/"
original_Sanger = location_folder_Sanger+"ic50_excl_empty.txt"
new_Sanger = location_folder_Sanger+"ic50_excl_empty_standardised.txt"


def standardise_Sanger(location):
    (X,_,M,drug_names,cell_lines,cancer_types,tissues) = load_data.load_Sanger()
    
    mean_columns = numpy.mean(X,axis=0) #column-wise mean, std
    std_columns = numpy.std(X,axis=0)
    
    X_standardised = X - [mean_columns for i in range(0,len(X))]
    X_standardised = X_standardised / [std_columns for i in range(0,len(X))]
    
    fout = open(new_Sanger,'w')
    first_line = "Cell Line\tCancer Type\tTissue\t" + "\t".join(drug_names) + "\n"
    fout.write(first_line)    
    for (cell_line,cancer_type,tissue,values) in zip(cell_lines,cancer_types,tissues,X_standardised):
        line = cell_line + "\t" + cancer_type + "\t" + tissue + "\t" + "\t".join([str(v) for v in values]) + "\n"
        fout.write(line)
    fout.close()
    
    
if __name__ == "__main__":
    standardise_Sanger(original_Sanger)