"""
Construct the kernels used in "Integrative and Personalized QSAR Analysis in 
Cancer by Kernelized Bayesian Matrix Factorization".
"""
import numpy

location_cell_line_features = "/home/tab43/Dropbox/Biological databases/Sanger_drug_sensivitiy/cell_line_features/"
name_cell_line_features = "en_input_w5.csv"

output_kernel_location = "/home/tab43/Documents/Projects/drug_sensitivity/NMTF_drug_sensitivity_prediction/data/"

gene_expression_kernel_name = "gene_expression"
copy_variation_kernel_name = "copy_variation"
cancer_gene_mutation_kernel_name = "mutation"



"""
We get the following features:
- Gene expression
- Copy variation
- Cancer gene mutation
First 13321 rows are gene expression values, next 426 are copy number profiles, 
final 82 are cancer gene mutations.
We also remove any rows with the same values for all cancer cell lines.
"""
def constuct_cell_line_kernels():
    features = numpy.loadtxt(location_cell_line_features+name_cell_line_features,dtype=str,delimiter=",")
    print features
    
    
if __name__ == "__main__":
    constuct_cell_line_kernels()