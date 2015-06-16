# Try different values for the latent dimension R, using K-fold cross-validation, and return the performances of the predictions for different values of R.

setwd("../kbmf/")
source("kbmf_regression_train.R")
source("kbmf_regression_test.R")

split_dataset <- function(Y, K) {
	nrows <- nrow(Y)
	ncols <- ncol(Y)
	entries <- which(!is.na(Y)) # gives indices of !NA entries

	# Split dataset
	random_entries <- sample(entries, replace = FALSE)
	split_entries <- split(random_entries, 1:K) 

	# Create folds and add to list
	folds <- list()
	for (i in seq(split_entries)) {
		# For each index, set that entry to the value in entries
		new_fold = matrix(data=NA, nrow=nrows, ncol=ncols)
		entries_fold = split_entries[[i]]
		for (j in seq(entries_fold)) {
			index = entries_fold[j]
			new_fold[index] = Y[index]
		}
		folds[[i]] = new_fold
	}
	return(folds)
}

create_train_test_sets <- function(folds,Y) {
	nrows <- nrow(Y)
	ncols <- ncol(Y)

	# Use the folds to create the training and test sets
	K <- length(folds)
	train_sets <- list()
	test_sets <- list()

	for (i in seq(folds)) {
		# Test set is this fold, training is the other four - so set these entries to NA
		fold = folds[[i]]
		test_sets[[i]] = fold

		train_sets[[i]] = matrix(data=Y, nrow=nrows, ncol=ncols) #copy over Y, and set values to NA
		entries <- which(!is.na(fold))
		for (j in seq(entries)) {
			index = entries[j]
			train_sets[[i]][index] = NA
		}
	}

	return(list(train_sets,test_sets))
}

kbmf_cross_validation <- function(Kx, Kz, Y, R_values, K) {
	# For each value R in R_values, split the data into K folds, and use K-1 folds to train and test on the remaining folds
	for (i in seq(R_values)) {
		sets = create_train_test_sets(split_dataset(Y, K), Y)
		training_sets = sets[[1]]
		test_sets = sets[[2]]

		# For each fold, test the performance
		for (f in seq(training_sets)) {
			train = training_sets[[f]]
			test = test_sets[[f]]

			state <- kbmf_regression_train(Kx, Kz, train, 20)
			prediction <- kbmf_regression_test(Kx, Kz, state)$Y$mu

			MSE = mean((prediction - test)^2, na.rm=TRUE )
			print(MSE)
		}
	}


}


