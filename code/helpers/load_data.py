"""
Helper function for reading in the Sanger dataset, splitting into data X,
mask M, drug names, cancer cell line names.
We exclude two lines from the dataset because on those cell lines only two 
drugs were tested.
Returns:
    X               Drug sensitivity values (original)
    X_min           Drug sensitivity values, minus (the lowest value in the dataset + 1)
    M               Mask of known vs unknown values
    drug_names      List of drug names
    cell_lines      List of which cell lines they are
    cancer_types    List of the cancer types of the cell lines
    tissues         List of tissue types of the cell lines
    
Also have a helper for storing it back into a file.
"""

import numpy, sys
sys.path.append("/home/thomas/Documenten/PhD/libraries/")
#sys.path.append("/home/tab43/Documents/Projects/libraries/")
import ml_helpers.code.mask as mask

#location_Sanger = "/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/data/Sanger_drug_sensitivity/"
location_Sanger = "/home/tab43/Documents/Projects/drug_sensitivity/NMTF_drug_sensitivity_prediction/data/Sanger_drug_sensitivity/"
file_name = "ic50_excl_empty.txt"
file_name_standardised = "ic50_excl_empty_standardised.txt"

def load_Sanger(standardised=False):
    """ Load in data. We get a masked array, and set masked values to 0. """
    fin = location_Sanger + (file_name if not standardised else file_name_standardised)
    data = numpy.genfromtxt(fin,dtype=str,delimiter="\t",usemask=True)
    
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
    
    return (X,X_min,M,drug_names,cell_lines,cancer_types,tissues)
    
    
def store_Sanger(location,X,M,drug_names,cell_lines,cancer_types,tissues):
    ''' Store the data X. First line is drug names, then comes the data.
        For the data, first column is cell line name, second is cancer type, 
        third is tissue, then follows the drug sensitivity values.
        For missing values we store nothing '''
    fout = open(location,'w')
    fout.write("Cell Line\tCancer Type\tTissue\t" + "\t".join(drug_names) + "\n")
    
    for i,(cell_line,cancer_type,tissue,row) in enumerate(zip(cell_lines,cancer_types,tissues,X)):
        line = cell_line+"\t"+cancer_type+"\t"+tissue+"\t"
        data = [str(val) if M[i][j] else ""for (j,val) in enumerate(row)]
        line += "\t".join(data) + "\n"
        fout.write(line)
    fout.close()
