# --------------------------------------------------------------------------------------------------------------------------------------

# -------------       BOOSTING - DIGIT RECOGNITION (differentiate number "1" and "7" in pictures)      ----------------------------

# --- Algorithm found in:  N. Abdul Rahim et al.  /  Procedia Engineering   53  ( 2013 )  411 – 419 
# ---      Adaptive Boosting with SVM Classifier for Moving Vehicle Classification
# ----------------------------------------------------------------------------------------------------------------------------------------


rm(list = ls())

# --- Libraries ---

library(data.table)
library(caret)
library(ggplot2)
library(jpeg)


# --- Read data sets ---

source("01_read_data.R")

# --- Select one current dataset for work

current_dataset_train <- digits_71_train
current_dataset_test <- digits_71_test
current_response_var <- "one1_seven0"

current_dataset_train[[current_response_var]] <- as.factor(current_dataset_train[[current_response_var]])
current_dataset_test[[current_response_var]] <- as.factor(current_dataset_test[[current_response_var]])

yvar <- current_dataset_train[[current_response_var]]
yvar_test <- current_dataset_test[[current_response_var]]

# --- Model formula

my_formula <- as.formula(paste0(current_response_var, " ~ ."))

# -------------------------- BOOSTING LOOP ADABoost M1 -------------------------

nIters <- 5
my_reg_models <- list()
my_reg_models_beta <- numeric(nIters)



# --- Initial sample weights

Ntrain <- nrow(current_dataset_train)
all_samples <- seq(1, Ntrain)
all_samples_prob <- rep(1/Ntrain,Ntrain)


itt <- 1

# --- Loop

while(itt <= nIters){
  
  cat("----------------------- SVM Model ", itt, "--------------------------")
  
  selected_samples <- sample(x = all_samples, size = floor(Ntrain*0.8), prob = all_samples_prob,replace = TRUE)
  
  
  # --- SVM model 
  
  my_reg0 <- caret::train(my_formula, current_dataset_train[selected_samples], method = "svmRadialCost",model=TRUE,
                             trControl = trainControl(
                               method = "cv", number = 2, verboseIter = FALSE),
                             tuneGrid=expand.grid(C = 10)
                           
  )
  
  # - Final model 
  
  my_reg_models[[itt]] <- my_reg0
  my_reg0_fitted <-predict(my_reg0,current_dataset_train)
  my_reg0_errors <- (my_reg0_fitted != yvar)
  confusion_m <- confusionMatrix(my_reg0_fitted, yvar, positive = "1")
  print(confusion_m$byClass[1:2])
  
  # - Errors of the model 
  
  my_reg0_total_error <- sum(my_reg0_errors*all_samples_prob)
  my_reg_models_beta[itt] <- my_reg0_total_error/(1-my_reg0_total_error)
  
  # - Update distribution of weights of samples in train set
  
  new_samples_dist <- ifelse(my_reg0_errors==1, 1,my_reg_models_beta[itt])
  all_samples_prob <- new_samples_dist/sum(new_samples_dist)
  
  itt <- itt + 1

}

# --- Final Ensemble fitness on train set


nBoost <- 5
my_reg_ensemble_train <- matrix(NA, nrow = Ntrain, ncol = nBoost)

for(itt in 1:nBoost){
  my_reg_ensemble_train[,itt] <- (as.numeric(predict(my_reg_models[[itt]], current_dataset_train))-1)
}

my_reg_models_beta_matrix <- matrix(log(1/my_reg_models_beta), nrow = Ntrain, ncol = nBoost, byrow = T)

betas_fitted_1 <- my_reg_ensemble_train*my_reg_models_beta_matrix
betas_fitted_0 <- (1-my_reg_ensemble_train)*my_reg_models_beta_matrix

diff_fitted <- apply(betas_fitted_1,1,sum)-apply(betas_fitted_0,1,sum)
my_reg_ensemble_final_fitted <- as.factor(ifelse(diff_fitted>0,1,0))

confusion_ensemble_fitted <- confusionMatrix(my_reg_ensemble_final_fitted, yvar, positive = "1")
print(confusion_ensemble_fitted)



# --------------------------- CHECK ENSEMBLE IN TEST SET ---------------


# --- Final Ensemble of Boost iterations


iim <- 2 # Check model by model
cat("----------------------- SVM predictions model ", iim, "--------------------------")
my_reg_pred <- predict(my_reg_models[[iim]], current_dataset_test)
confusion_m_pred <- confusionMatrix(my_reg_pred, yvar_test, positive = "1")
print(confusion_m_pred$byClass[1:2])


# --- Performance on test set Ensemble of Boost. Best result with 2 out of 5 models

Ntest <- nrow(current_dataset_test)
nBoost <- 2
my_reg_ensemble_pred <- matrix(NA, nrow = Ntest, ncol = nBoost)

for(itt in 1:nBoost){
  my_reg_ensemble_pred[,itt] <- (as.numeric(predict(my_reg_models[[itt]], current_dataset_test))-1)
}

my_reg_models_beta_matrix <- matrix(log(1/my_reg_models_beta[1:nBoost]), nrow = Ntest, ncol = nBoost, byrow = T)

betas_predicted_1 <- my_reg_ensemble_pred*my_reg_models_beta_matrix
betas_predicted_0 <- (1-my_reg_ensemble_pred)*my_reg_models_beta_matrix

diff_pred <- apply(betas_predicted_1,1,sum)-apply(betas_predicted_0,1,sum)
my_reg_ensemble_final_pred <- as.factor(ifelse(diff_pred>0,1,0))

confusion_ensemble_pred <- confusionMatrix(my_reg_ensemble_final_pred, yvar_test, positive = "1")
print(confusion_ensemble_pred)


# --- Pictures of some examples in test set

bitmap_test <- copy(current_dataset_test)
bitmap_test[[current_response_var]] <- NULL

image_id <- 2053

image_matrix <- matrix(as.matrix(bitmap_test[image_id]), nrow=32, ncol=32, byrow = TRUE)
yvar_test[image_id]
writeJPEG(image_matrix, paste0("figures/04_image_",as.character(image_id),".jpeg"), quality = 1)

