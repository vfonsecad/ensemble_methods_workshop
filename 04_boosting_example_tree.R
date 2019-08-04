# -----------------------------------------------------------------------

# -------------       BOOSTING - DECISION TREES      --------------------

# -----------------------------------------------------------------------


rm(list = ls())

# --- Libraries ---

library(data.table)
library(caret)
library(ggplot2)
library(rpart.plot)
library(adabag)



# --- Read data sets ---

source("01_read_data.R")

# --- Select one current dataset for work

current_dataset_train <- breast_train
current_dataset_test <- breast_test
current_response_var <- "diagnosis"

current_dataset_train[[current_response_var]] <- as.factor(current_dataset_train[[current_response_var]])
current_dataset_test[[current_response_var]] <- as.factor(current_dataset_test[[current_response_var]])

yvar <- current_dataset_train[[current_response_var]]
yvar_test <- current_dataset_test[[current_response_var]]



# --------------------------------- CART MODEL TRAINING --------------------------


my_formula <- as.formula(paste0(current_response_var, " ~ ."))

# --- Model 0: This model corresponds to library rpart. The tuning parameter is the complexity 
# parameter (cp). The closer to 0, the more complex or deep the tree. The higher, the less complex the tree.
# In this particular example, even with a very complex and deep tree, the performance on test data still makes a few errors and hey! this is about cancer
# - CV

my_cart0 <- caret::train(my_formula, current_dataset_train, method = "rpart",model=TRUE,
                           trControl = trainControl(
                             method = "cv", number = 10, verboseIter = TRUE),
                           tuneGrid=expand.grid(cp=0.00005),
                         control = rpart.control(maxdepth = 20)
                         
)

# - Plot the tree
rpart.plot(my_cart0$finalModel)

# - CP table
my_cart0$finalModel$cptable

# - Prediction on test set
my_cart0_pred <-predict(my_cart0,current_dataset_test)

# - Performance on test set
confusionMatrix(my_cart0_pred,yvar_test, positive = "1")



# ------------------- BOOSTING ADABoost -----------------------------------

# --- Model 1: For full documentation see https://cran.r-project.org/web/packages/adabag/adabag.pdf
# For this boosting procedure, we tune between 2 and 10 trees to be fitted iteratively
# M1 is the specific type of ADABoost aglorithm to use. Currently only one available
# boos controls whether to use the full training sample in each tree or a boostrap samples
# coeflearn controls the type of weighting coefficients for the samples in each iteration
# maxdepth controls the maximum depth of each tree. See rpart.control for further documentation

my_cart1 <- caret::train(my_formula, current_dataset_train, method = "AdaBoost.M1",
                         trControl = trainControl(
                         method = "cv", number = 10, verboseIter = TRUE),
                         tuneGrid=expand.grid(mfinal = 2, coeflearn = "Breiman", maxdepth = 5),
                         boos = TRUE
                         
                         
)

# - Retrive actual final forest
my_cart1_model <- my_cart1$finalModel
# - Visualize each tree of the ensemble
rpart.plot(my_cart1_model$trees[[1]],roundint=FALSE)
rpart.plot(my_cart1_model$trees[[2]],roundint=FALSE)
# - Predict on test set
my_cart1_pred <-predict(my_cart1,current_dataset_test, type = "raw")
# - Performance on test set
confusionMatrix(my_cart1_pred,yvar_test, positive = "1")
# - Variable importance measures
plot(varImp(my_cart1))
# - CP table for each tree. Note that the complexity parameter behaviour for each tree in the
# forest is not the same
my_cart1_model$trees[[1]]$cptable
my_cart1_model$trees[[2]]$cptable
# -  Beta weight each tree by ADABoost M1
my_cart1_model$weights
