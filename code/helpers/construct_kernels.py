"""
Construct the kernels used in "Integrative and Personalized QSAR Analysis in 
Cancer by Kernelized Bayesian Matrix Factorization".

Also construct a new dataset for the drug sensitivity values, consisting of 
only those cancer cell lines for which we have features available.
There are 622 such cell lines.
"""
import numpy
import load_data

location_cell_line_features = "/home/tab43/Dropbox/Biological databases/Sanger_drug_sensivitity/cell_line_features/"
name_cell_line_features = "en_input_w5.csv"

output_kernel_location = "/home/tab43/Documents/Projects/drug_sensitivity/NMTF_drug_sensitivity_prediction/data/kernels/"

output_Sanger_location = "/home/tab43/Dropbox/Biological databases/Sanger_drug_sensivitity/"
name_Sanger_filtered = "ic50_excl_empty_features.txt"

output_cell_line_features_location = "/home/tab43/Dropbox/Biological databases/Sanger_drug_sensivitity/cell_line_features/"

gene_expression_kernel_name = "gene_expression"
copy_variation_kernel_name = "copy_variation"
cancer_gene_mutation_kernel_name = "mutation"

gene_start_index = 0
copy_number_start_index = gene_start_index+13321
cancer_gene_start_index = copy_number_start_index+426
tissue_subtype_start_index = cancer_gene_start_index+71


"""
HELPER METHODS
"""
# Method for removing features (columns) with only the same value.
def remove_homogenous_features(matrix,names):
    columns_to_remove = []
    for i,column in enumerate(matrix.T):
        if len(numpy.unique(column)) == 1:
            columns_to_remove.append(i)
            print "Removed %sth feature, name %s." % (i,names[i])
    new_matrix = numpy.delete(matrix,columns_to_remove,axis=1)
    new_names = numpy.delete(names,columns_to_remove)
    return (new_matrix,new_names)

# Method for sorting the matrix rows based on the names (alphabetically).
def sort_matrix_rows(matrix,names):
    name_to_row = {}
    for name,row in zip(names,matrix):
        name_to_row[name] = row
    sorted_names = sorted(names)
    new_matrix = numpy.array([name_to_row[name] for name in sorted_names])
    return new_matrix

# Method for computing the intersection of two lists
def intersection(a,b):
    return sorted(list(set(a) & set(b)))

# Method for only keeping rows with the specified names
def remove_matrix_rows(matrix,all_names,keep_names):
    # Compute indices of rows to remove
    rows_to_remove = [i for i,name in enumerate(all_names) if name not in keep_names]
    new_matrix = numpy.delete(matrix,rows_to_remove,axis=0)
    return new_matrix

# Method for storing the feature matrices for cell lines, first row the feature
# names, and then each row first the cell line names, then all the feature values
def store_cell_line_features(location,matrix,cell_line_names,feature_names):
    fout = open(location,'w')
    line = "\t".join(feature_names) + "\n"
    fout.write(line)
    for cell_line_name,row in zip(cell_line_names,matrix):
        line = cell_line_name + "\t" + "\t".join([str(v) for v in row]) + "\n"
        fout.write(line)
    fout.close()
    

"""
KERNEL METHODS
Methods for computing the kernels - specifically Jaccard coefficient and Gaussian.
"""



"""
CELL LINE KERNELS

There are 623 cell lines.
We get the following features:
- Gene expression
- Copy variation
- Cancer gene mutation
First 13321 rows are gene expression values, next 426 are copy number profiles, 
next 71 are cancer gene mutations, and final 13 are tissue sub-types.
We also remove any rows with the same values for all cancer cell lines.

Then we compute the overlap with the Sanger drug sensitivity dataset, and remove
any cell lines in both that set and our feature sets that do not appear in both.
These are then stored in a folder.

Finally, we end up with the following kernels:

Feature type            Number of features    Kernel
Gene expression         13321                 Gaussian
Copy number             426                   Gaussian
Cancer gene mutations   71                    Jaccard
"""
def construct_filtered_feature_matrices_cell_lines():
    # Load in the feature sets
    lines = open(location_cell_line_features+name_cell_line_features,'r').readlines()
    lines = [line.split("\n")[0].split(",") for line in lines]
    features = numpy.array(lines[1:])
    cell_line_names = lines[0]
    
    
    # Extract the features, and feature names
    gene_expression_names = features[gene_start_index:copy_number_start_index,0]
    gene_expression_values = numpy.array(features[gene_start_index:copy_number_start_index,1:],dtype=float).T
    
    copy_number_names = features[copy_number_start_index:cancer_gene_start_index,0]
    copy_number_values = numpy.array(features[copy_number_start_index:cancer_gene_start_index,1:],dtype=float).T
    
    cancer_gene_names = features[cancer_gene_start_index:tissue_subtype_start_index,0]
    cancer_gene_values = numpy.array(features[cancer_gene_start_index:tissue_subtype_start_index,1:],dtype=float).T
        
    tissue_subtype_names = features[tissue_subtype_start_index:,0]
    tissue_subtype_values = numpy.array(features[tissue_subtype_start_index:,1:],dtype=float).T
    
    
    # Filter out rows for all the same values - which is none in this dataset (but we still check)
    (gene_expression_values,gene_expression_names) = remove_homogenous_features(gene_expression_values,gene_expression_names)
    (copy_number_values,copy_number_names) = remove_homogenous_features(copy_number_values,copy_number_names)
    (cancer_gene_values,cancer_gene_names) = remove_homogenous_features(cancer_gene_values,cancer_gene_names)
    
    
    # Reorder the rows in the datasets so that they are in alphabetical order of cell line name
    gene_expression_values = sort_matrix_rows(gene_expression_values,cell_line_names)
    copy_number_values = sort_matrix_rows(copy_number_values,cell_line_names)
    cancer_gene_values = sort_matrix_rows(cancer_gene_values,cell_line_names)
    
    # Also reorder the names
    sorted_cell_line_names = sorted(cell_line_names)
    
    
    # Load in the Sanger dataset, and the names of the cell lines etc. Then sort it on cell line name alphabetically
    (X,_,M,Sanger_drug_names,Sanger_cell_lines,Sanger_cancer_types,Sanger_tissues) = load_data.load_Sanger()
    X_sorted = sort_matrix_rows(X,Sanger_cell_lines)
    M_sorted = sort_matrix_rows(M,Sanger_cell_lines)
    
    (sorted_Sanger_cell_lines,sorted_Sanger_cancer_types,sorted_Sanger_tissues) = \
        (numpy.array(t) for t in zip(*sorted(zip(Sanger_cell_lines,Sanger_cancer_types,Sanger_tissues))))
    
    
    # Filter out the cell lines that are in common between the features and the drug sensivitity dataset
    overlap_cell_lines = intersection(sorted_Sanger_cell_lines,sorted_cell_line_names)
    
    X_filtered = remove_matrix_rows(X_sorted,sorted_Sanger_cell_lines,overlap_cell_lines)
    M_filtered = remove_matrix_rows(M_sorted,sorted_Sanger_cell_lines,overlap_cell_lines)
    
    Sanger_cancer_types_filtered = remove_matrix_rows(sorted_Sanger_cancer_types,Sanger_cell_lines,overlap_cell_lines)
    Sanger_tissues_filtered = remove_matrix_rows(sorted_Sanger_tissues,Sanger_cell_lines,overlap_cell_lines)
    
    gene_expression_values_filtered = remove_matrix_rows(gene_expression_values,sorted_cell_line_names,overlap_cell_lines)
    copy_number_values_filtered = remove_matrix_rows(copy_number_values,sorted_cell_line_names,overlap_cell_lines)
    cancer_gene_values_filtered = remove_matrix_rows(cancer_gene_values,sorted_cell_line_names,overlap_cell_lines)
    
    
    # Store the new X, as well as the feature matrices
    load_data.store_Sanger(
        location=output_Sanger_location+name_Sanger_filtered,
        X=X_filtered,
        M=M_filtered,
        drug_names=Sanger_drug_names,
        cell_lines=overlap_cell_lines,
        cancer_types=Sanger_cancer_types_filtered,
        tissues=Sanger_tissues_filtered)
    
    store_cell_line_features(   # gene expression values
        location=output_cell_line_features_location+gene_expression_kernel_name,
        matrix=gene_expression_values_filtered,
        cell_line_names=overlap_cell_lines,
        feature_names=gene_expression_names)
    store_cell_line_features(   # copy number values
        location=output_cell_line_features_location+copy_variation_kernel_name,
        matrix=copy_number_values_filtered,
        cell_line_names=overlap_cell_lines,
        feature_names=copy_number_names)
    store_cell_line_features(   # cancer gene mutation values
        location=output_cell_line_features_location+cancer_gene_mutation_kernel_name,
        matrix=cancer_gene_values_filtered,
        cell_line_names=overlap_cell_lines,
        feature_names=cancer_gene_names)


#def constuct_cell_line_kernels():


    
if __name__ == "__main__":
    #constuct_cell_line_kernels()
    pass

