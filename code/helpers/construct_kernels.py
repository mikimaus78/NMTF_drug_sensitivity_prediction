"""
Construct the kernels used in "Integrative and Personalized QSAR Analysis in 
Cancer by Kernelized Bayesian Matrix Factorization".

Also construct a new dataset for the drug sensitivity values, consisting of 
only those cancer cell lines for which we have features available.
There are 622 such cell lines.
"""
import numpy, math
import load_data

output_kernel_location = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/data/kernels/"#"/home/tab43/Documents/Projects/drug_sensitivity/NMTF_drug_sensitivity_prediction/data/kernels/"
output_Sanger_location = "/home/tab43/Dropbox/Biological databases/Sanger_drug_sensivitity/"
name_Sanger_filtered = "ic50_excl_empty_features.txt"


"""
HELPER METHODS
"""
# Method for loading in a file where the first line is names of the columns, 
# the first column is names of rows, and values are of the specified datatype.
def read_file_with_names(location,datatype=float):
    lines = open(location,'r').readlines()
    lines = [line.split("\n")[0].split("\t") for line in lines]
    names_columns = lines[0]
    values = numpy.array(lines[1:])
    names_rows = values[:,0]
    values = numpy.array(values[:,1:],dtype=datatype)
    return (names_columns,names_rows,values)

# Read csv file, comma separated. First row is "Name" + feature names, first 
# column is drug/cell line names. Return (names_columns,names_rows,values) tuple.
def read_csv_file(location,delim=','):
    lines = numpy.array([line.split("\n")[0].split(delim) for line in open(location,'r').readlines()])
    feature_names = lines[0,1:]
    row_names = lines[1:,0]
    values = lines[1:,1:]
    values = numpy.array(lines[1:,1:],dtype=float)
    return (feature_names,row_names,values)

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

# Method for standardising the columns of a matrix
def standardise_columns(matrix):
    mean_columns = numpy.mean(matrix,axis=0) #column-wise mean, std
    std_columns = numpy.std(matrix,axis=0)
    matrix_standardised = ( matrix - [mean_columns for i in range(0,len(matrix))] ) / [std_columns for i in range(0,len(matrix))]
    return matrix_standardised

# Method for storing the feature matrices for cell lines/drugs, first row the 
# feature names, and then each row first the cell line/drug names, then all the 
# feature values.
def store_features(location,matrix,row_names,feature_names):
    fout = open(location,'w')
    line = "\t".join(feature_names) + "\n"
    fout.write(line)
    for row_name,row in zip(row_names,matrix):
        line = row_name + "\t" + "\t".join([str(v) for v in row]) + "\n"
        fout.write(line)
    fout.close()
    
# Method for storing the kernels - first row is cancer cell line/drug names, then kernel values
def store_kernel(location,matrix,names):
    fout = open(location,'w')
    line = "\t".join(names) + "\n"
    fout.write(line)
    for row in matrix:
        line = "\t".join([str(v) for v in row]) + "\n"
        fout.write(line)
    fout.close()
    
    
    

"""
KERNEL METHODS
Methods for computing the kernels - specifically Jaccard coefficient and Gaussian.

Jaccard(a1,a2) = |a1 and a2| / |a1 or a2|
We take 1-Jaccard(a1,a2) as we want our kernels to measure dissimilarity.
"""
# Jaccard coefficient
def jaccard_kernel(values):
    print "Computing Jaccard kernel..."
    # Rows are data points
    no_points = numpy.array(values).shape[0]
    kernel = numpy.zeros((no_points,no_points))
    for i in range(0,no_points):
        print "Data point %s." % i
        data_point_1 = values[i]
        for j in range(i,no_points):
            data_point_2 = values[j]
            sim = jaccard(data_point_1,data_point_2)
            kernel[i,j] = 1 - sim
            kernel[j,i] = 1 - sim
    assert numpy.array_equal(kernel,kernel.T), "Constructed kernel but did not become symmetrical!"
    return kernel
    
def jaccard(a1,a2):
    a_and = [1 for (v1,v2) in zip(a1,a2) if v1 == 1 and v2 == 1]
    a_or = [1 for (v1,v2) in zip(a1,a2) if v1 == 1 or v2 == 1]
    return 1 if len(a_or) == 0 else len(a_and) / float(len(a_or))


# Gaussian kernel
def gaussian_kernel(values): 
    print "Computing Gaussian kernel..."
    (no_points,no_features) = numpy.array(values).shape
    sigma_2 = math.sqrt( no_features / 4. ) # The KMF paper used sigma^2 = sqrt(D/4) for D = no. features
    kernel = numpy.zeros((no_points,no_points))
    for i in range(0,no_points):
        print "Data point %s." % i
        data_point_1 = values[i]
        for j in range(i,no_points):
            data_point_2 = values[j]
            sim = gaussian(data_point_1,data_point_2,sigma_2)
            kernel[i,j] = 1 - sim
            kernel[j,i] = 1 - sim
    assert numpy.array_equal(kernel,kernel.T), "Constructed kernel but did not become symmetrical!"
    return kernel

def gaussian(a1,a2,sigma_2):
    distance = sum([(x1-x2)**2 for x1,x2 in zip(a1,a2)])
    return math.exp( -distance / (2.*sigma_2) )


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
location_cell_line_features = "/home/tab43/Dropbox/Biological databases/Sanger_drug_sensivitity/cell_line_features/"
name_cell_line_features = "en_input_w5.csv"

output_cell_line_features_location = "/home/tab43/Dropbox/Biological databases/Sanger_drug_sensivitity/cell_line_features/"

gene_expression_kernel_name = "gene_expression"
copy_variation_kernel_name = "copy_variation"
cancer_gene_mutation_kernel_name = "mutation"

gene_start_index = 0
copy_number_start_index = gene_start_index+13321
cancer_gene_start_index = copy_number_start_index+426
tissue_subtype_start_index = cancer_gene_start_index+71


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
    
    store_features(   # gene expression values
        location=output_cell_line_features_location+gene_expression_kernel_name,
        matrix=gene_expression_values_filtered,
        row_names=overlap_cell_lines,
        feature_names=gene_expression_names)
    store_features(   # copy number values
        location=output_cell_line_features_location+copy_variation_kernel_name,
        matrix=copy_number_values_filtered,
        row_names=overlap_cell_lines,
        feature_names=copy_number_names)
    store_features(   # cancer gene mutation values
        location=output_cell_line_features_location+cancer_gene_mutation_kernel_name,
        matrix=cancer_gene_values_filtered,
        row_names=overlap_cell_lines,
        feature_names=cancer_gene_names)


"""
Standardise the feature matrices for the Gaussian kernel by feature, to mean 0 
and std 1.
"""
def standardise_features_cell_lines():
    # Load in the non-standardised features
    (gene_expression_names,cell_line_names,gene_expression_values) = read_file_with_names(output_cell_line_features_location+gene_expression_kernel_name,datatype=float)
    (copy_number_names,cell_line_names,copy_number_values) = read_file_with_names(output_cell_line_features_location+copy_variation_kernel_name,datatype=float)
    
    # Rows are cell lines, columns features, so we standardise the columns
    gene_expression_values_standardised = standardise_columns(gene_expression_values) 
    copy_number_values_standardised = standardise_columns(copy_number_values)    
    
    # Write the standardised features
    store_features(   # gene expression values
        location=output_cell_line_features_location+gene_expression_kernel_name+"_std",
        matrix=gene_expression_values_standardised,
        row_names=cell_line_names,
        feature_names=gene_expression_names)
    store_features(   # copy number values
        location=output_cell_line_features_location+copy_variation_kernel_name+"_std",
        matrix=copy_number_values_standardised,
        row_names=cell_line_names,
        feature_names=copy_number_names)
    
    
""" Use the earlier created feature matrices to construct feature kernels. """
def constuct_cell_line_kernels():
    # Gene expression kernel
    (gene_expression_names,cell_line_names,gene_expression_values) = read_file_with_names(output_cell_line_features_location+gene_expression_kernel_name+"_std",datatype=float)
    gene_expression_kernel = gaussian_kernel(gene_expression_values)  
    store_kernel(output_kernel_location+gene_expression_kernel_name,gene_expression_kernel,cell_line_names)  
    
    # Copy number kernel
    (copy_number_names,cell_line_names,copy_number_values) = read_file_with_names(output_cell_line_features_location+copy_variation_kernel_name+"_std",datatype=float)
    copy_number_kernel = gaussian_kernel(copy_number_values)
    store_kernel(output_kernel_location+copy_variation_kernel_name,copy_number_kernel,cell_line_names)
    
    # Cancer gene mutation kernel
    (cancer_gene_names,cell_line_names,cancer_gene_values) = read_file_with_names(output_cell_line_features_location+cancer_gene_mutation_kernel_name,datatype=float)
    cancer_gene_kernel = jaccard_kernel(cancer_gene_values)
    store_kernel(output_kernel_location+cancer_gene_mutation_kernel_name,cancer_gene_kernel,cell_line_names)

    
    
"""
DRUG KERNELS

There are 140 drugs.
After filtering out one drug (84691) whose 1D and 2D descriptors did not 
process properly in PaDeL, we have 139 drugs left.

We get the following features:
- PubChem fingerprints
- 1&2D descriptors (PaDeL)
- Drug targets (Sanger)
- Vsurf (PaDeL 2D -> LigPrep -> Molecular Operating Environment Software)
- GRIND/GRIND2 (PaDeL 2D -> Pentacle)

We end up with the following kernels:

Feature type            Number of features      Kernel
PubChem                 _                       Jaccard
1&2D descriptors        _                       Gaussian
Drug targets            _                       Jaccard
Vsurf                   _                       Gaussian
GRIND/GRIND2            _                       Gaussian
"""
location_drug_features = "/home/thomas/Dropbox/Biological databases/Sanger_drug_sensivitity/drug_features/"#"/home/tab43/Dropbox/Biological databases/Sanger_drug_sensivitity/drug_features/"

descriptors_1d2d_kernel_name = "1d2d_descriptors"
targets_kernel_name = "targets"
pubchem_fp_kernel_name = "PubChem_fingerprints"


# Read in the drug_target file and convert it into a binary matrix
def construct_binary_drug_target_matrix():
    # Read in the lists of drug targets, tab separated, and the list of PubChem ids
    pubchem_ids = [pc_id.split("\n")[0] for pc_id in open(location_drug_features+"drug_pubchem_ids").readlines()]
    targets = [target.split("\n")[0].split("\t") for target in open(location_drug_features+"targets/drug_targets").readlines()]
    targets = [[target for target in target_list if target != ''] for target_list in targets] #take out the empty targets (target name = '')
    feature_names = sorted(list(set([x for sublist in targets for x in sublist]))) 
    
    # Sort the feature names, and make a binary matrix using that order
    no_features = len(feature_names)
    no_drugs = len(pubchem_ids)    
    binary_matrix = numpy.zeros((no_drugs,no_features))
    for i,target_list in enumerate(targets):
        for target in target_list:
            id_in_matrix = feature_names.index(target)
            binary_matrix[i,id_in_matrix] = 1
    
    # Store the matrix. First row is feature names, first column drug ids.
    store_features(
        location=location_drug_features+"targets/drug_targets_binary",
        matrix=binary_matrix,
        row_names=pubchem_ids,
        feature_names=feature_names
    )
    
    
# Standardise the features for the Gaussian kernel: 1&2D descriptors
def standardise_features_drugs():
    location_1d2d_descriptors = location_drug_features+"1d2d/drug_1d2d_filtered.csv"
    (descriptor_names,drug_ids,values_1d2d) = read_csv_file(location_1d2d_descriptors,"\t")
    
    # Check features for constant values across all drugs, and then standardise them
    (values_1d2d,descriptor_names) = remove_homogenous_features(values_1d2d,descriptor_names)
    new_values_1d2d = standardise_columns(values_1d2d)
    
    store_features(
        location=location_drug_features+"1d2d/drug_1d2d_filtered_std",
        matrix=new_values_1d2d,
        row_names=drug_ids,
        feature_names=descriptor_names
    )


# Construct the kernels from the features
def construct_drug_kernels():
    location_1d2d_descriptors = location_drug_features+"1d2d/drug_1d2d_filtered_std"
    location_PubChem_fingerprints = location_drug_features+"fingerprints/drug_PubChem_fingerprints_filtered.csv"
    location_targets = location_drug_features+"targets/drug_targets_binary_filtered"
    
    (descriptor_names,drug_ids,values_1d2d) = read_file_with_names(location_1d2d_descriptors)
    (fingerprint_names,_,values_fingerprints) = read_csv_file(location_PubChem_fingerprints)
    (target_names,_,values_targets) = read_file_with_names(location_targets)
    
    # Also check columns for the same values across all drugs, and remove those
    (values_1d2d,descriptor_names) = remove_homogenous_features(values_1d2d,descriptor_names)
    (values_fingerprints,fingerprint_names) = remove_homogenous_features(values_fingerprints,fingerprint_names)
    (values_targets,target_names) = remove_homogenous_features(values_targets,target_names)
    
    # Then construct the kernels
    descriptors_1d2d_kernel = gaussian_kernel(values_1d2d)  
    store_kernel(output_kernel_location+descriptors_1d2d_kernel_name,descriptors_1d2d_kernel,drug_ids)
    
    fingerprints_kernel = jaccard_kernel(values_fingerprints)  
    store_kernel(output_kernel_location+pubchem_fp_kernel_name,fingerprints_kernel,drug_ids)
    
    targets_kernel = jaccard_kernel(values_targets)  
    store_kernel(output_kernel_location+targets_kernel_name,targets_kernel,drug_ids)
    



if __name__ == "__main__":
    construct_drug_kernels()
    pass

