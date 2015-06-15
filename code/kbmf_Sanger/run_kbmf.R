source("kbmf_regression_train.R")
source("kbmf_regression_test.R")

set.seed(1606)

Px <- 3
Nx <- 622
Pz <- 3
Nz <- 139

# Load in the drug sensitivity values
folder_drug_sensitivity <- '/home/thomas/Dropbox/Biological databases/Sanger_drug_sensivitity/'
name_drug_sensitivity <- 'ic50_excl_empty_filtered_cell_lines_drugs.txt'
Y <- as.matrix(read.table(paste(folder_drug_sensitivity,name_drug_sensitivity,sep=''),
				header=TRUE,
				sep='\t',
				colClasses=c(rep("NULL",3), rep("numeric",139))))

# Load in the kernels - X = cancer cell lines, Z = drugs
folder_kernels <- '/home/thomas/Documenten/PhD/NMTF_drug_sensitivity_prediction/data/kernels/'

kernel_copy_variation <- as.matrix(read.table(paste(folder_kernels,'copy_variation',sep=''),header=TRUE))
kernel_gene_expression <- as.matrix(read.table(paste(folder_kernels,'gene_expression',sep=''),header=TRUE))
kernel_mutation <- as.matrix(read.table(paste(folder_kernels,'mutation',sep=''),header=TRUE))

kernel_1d2d <- as.matrix(read.table(paste(folder_kernels,'1d2d_descriptors',sep=''),header=TRUE))
kernel_fingerprints<- as.matrix(read.table(paste(folder_kernels,'PubChem_fingerprints',sep=''),header=TRUE))
kernel_targets <- as.matrix(read.table(paste(folder_kernels,'targets',sep=''),header=TRUE))

Kx <- array(0, c(Nx, Nx, Px))
Kx[,, 1] <- kernel_copy_variation
Kx[,, 2] <- kernel_gene_expression
Kx[,, 3] <- kernel_mutation

Kz <- array(0, c(Nz, Nz, Pz))
Kz[,, 1] <- kernel_1d2d
Kz[,, 2] <- kernel_fingerprints
Kz[,, 3] <- kernel_targets

state <- kbmf_regression_train(Kx, Kz, Y, 5)
prediction <- kbmf_regression_test(Kx, Kz, state)

print(prediction$Y$mu)

print(sprintf("MSE = %.4f", mean((prediction$Y$mu - Y)^2, na.rm=TRUE )))
# 200 iterations: "MSE = 2.0170"
# 1000 iterations: "MSE = "

print("kernel weights on X")
print(state$ex$mu)

print("kernel weights on Z")
print(state$ez$mu)
