# -----------------------------------------------------------------------

# -------------       BAGGING - RANDOM FORESTS       --------------------

# -----------------------------------------------------------------------


rm(list = ls())

# --- Libraries ---

library(data.table)
library(caret)
library(ggplot2)
library(rpart.plot)
library(randomForest)
library(ipred)


# --- Read data sets ---

source("01_read_data.R")

# --- Select one current dataset for work

current_dataset_train <- credit_train
current_dataset_test <- credit_test
current_response_var <- "Sale_MF"

current_dataset_train[[current_response_var]] <- as.factor(current_dataset_train[[current_response_var]])
current_dataset_test[[current_response_var]] <- as.factor(current_dataset_test[[current_response_var]])

yvar <- current_dataset_train[[current_response_var]]
yvar_test <- current_dataset_test[[current_response_var]]


# --------------------------------- CART MODEL TRAINING --------------------------


my_formula <- as.formula(paste0(current_response_var, " ~ ."))


# --- Model 0: This model corresponds to library rpart. The tuning parameter is the complexity 
# parameter (cp). The closer to 0, the more complex or deep the tree. The higher, the less complex the tree.
# In this particular example, even with a cp of 5e-10 the three would not improve (see cp table)

# - CV

my_cart0 <- caret::train(my_formula, current_dataset_train, method = "rpart",model=TRUE,
                           trControl = trainControl(
                             method = "cv", number = 10, verboseIter = TRUE),
                         tuneGrid=expand.grid(cp=0.005)
)

# - Plot the tree
rpart.plot(my_cart0$finalModel)

# - CP table
my_cart0$finalModel$cptable

# - Prediction on test set
my_cart0_pred <-predict(my_cart0,current_dataset_test)

# - Performance on test set
confusionMatrix(my_cart0_pred,yvar_test, positive = "1")


# ------------------- BAGGED CART -----------------------------------

# --- Model 1: In this model we use procedure treebag. For more information on this procedure
# see https://cran.r-project.org/web/packages/ipred/ipred.pdf/ 
# The parameter nbagg controls the number of trees to be grown. The subsamples are by default with replacement
# and of the same size as the original training dataset. 

my_cart1 <- caret::train(my_formula, current_dataset_train, method = "treebag",
                         trControl = trainControl(
                         method = "cv", number = 10, verboseIter = TRUE),
                         nbagg = 4
)

# - Retrive actual final forest
my_cart1_model<- my_cart1$finalModel
# - Visualize one particular tree of the forest
rpart.plot(my_cart1_model$mtrees[[1]]$btree)
# - Predict on test set
my_cart1_pred <-predict(my_cart1,current_dataset_test, type = "raw")
# - Performance on test set
confusionMatrix(my_cart1_pred,yvar_test, positive = "1")
# - Variable importance measures
plot(varImp(my_cart1))
# - CP table for each tree. Note that the complexity parameter behaviour for each tree in the
# forest is not the same
my_cart1_model$mtrees[[1]]$btree$cptable
my_cart1_model$mtrees[[2]]$btree$cptable
my_cart1_model$mtrees[[3]]$btree$cptable
my_cart1_model$mtrees[[4]]$btree$cptable



# ----------------- RANDOM FOREST rf method -------------

# --- Model 2: This procedure is based on randomForest.
# For full information on the package and parameters of the training function see
# https://cran.r-project.org/web/packages/randomForest/randomForest.pdf
# This algorithm is more powerful for growing a more complex random forest. It
# has the capacity to resample the variables selected on each boostrap sample, which is
# controlled by the parameter mtry.

my_cart2 <- caret::train(my_formula, current_dataset_train, method = "rf",
                         trControl = trainControl(
                           method = "cv", number = 10, verboseIter = TRUE),
                         ntree = 4,
                         tuneGrid=expand.grid(mtry=5:28),
                         keep.forest = TRUE
)

# - Retrieve final model
my_cart2_model <- my_cart2$finalModel
# - Predict on test set
my_cart2_pred <-predict(my_cart2,current_dataset_test)
# - Performance on test set
confusionMatrix(my_cart2_pred,yvar_test, positive = "1")
# - Variable importance plot
plot(varImp(my_cart2))

# - Number of trees grown
my_cart2_model$ntree
# - Matrix depicting a selected tree of the forest
getTree(my_cart2_model, k=1, labelVar = T)

  