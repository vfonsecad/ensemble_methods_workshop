# -----------------------------------------------------------------------

# -------------       DATA SETS       ----------------------

# -----------------------------------------------------------------------


rm(list = ls())

# --- libraries

library(data.table)
library(caret)

# --- read datasets with fread function


credit <- fread("data/credit-data/credit-data.csv", sep = ";") # Yvar: Sale_MF (binary)

breast <- fread("data/breast-cancer-data/breast-cancer-data.csv", sep = ";") # Yvar: diagnosis (binary)

spectra_train <- fread("data/spectroscopy-data/spectra_train.csv", sep = ";") # Yvar named "Y" (continuous)
spectra_test <- fread("data/spectroscopy-data/spectra_test.csv", sep = ";")


digits_71_train <- fread("data/digits-data/digits_train.csv", sep = ";") # Yvar: one1_seven0 (binary)
digits_71_test <- fread("data/digits-data/digits_test.csv", sep = ";") # Yvar: one1_seven0

data_sets <- list(credit = credit,
                 breast = breast)


# - Check for missing data in each data set (spectral data has no missings)

lapply(data_sets, function(x) sum(is.na(x)))


# - Data splitting

#   Based on response variable for credit, breast and suicide

credit_trainIndex <- createDataPartition(credit[["Sale_MF"]], p = .8, 
                                  list = FALSE, 
                                  times = 1)

breast_trainIndex <- createDataPartition(breast[["diagnosis"]], p = .8, 
                                         list = FALSE, 
                                         times = 1)



credit_train <- credit[credit_trainIndex[,1]]
credit_test <- credit[-credit_trainIndex[,1]]

breast_train <- breast[breast_trainIndex[,1]]
breast_test <- breast[-breast_trainIndex[,1]]

