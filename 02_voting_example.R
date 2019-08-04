# -----------------------------------------------------------------------

# -------------       VOTING  (PLS MODELS)      ----------------------

# -----------------------------------------------------------------------


rm(list = ls())

# --- Libraries ---

library(data.table)
library(caret)
library(ggplot2)



# --- Read data sets ---

source("01_read_data.R")

# --- Select one current dataset for work

current_dataset_train <- spectra_train
current_dataset_test <- spectra_test
current_response_var <- "Y"
yvar <- current_dataset_train[[current_response_var]]
yvar_test <- current_dataset_test[[current_response_var]]


# -------------------------- PLS MODEL TRAINING -----------------------------

# --- train Model by Tag in Caret


my_formula <- as.formula(paste0(current_response_var, " ~ ."))


# --- Model 0

# Fit a regular pls model. No preprocessing because this data is already centered and no scaling is needed. 
# In PLS models (partial least squares) the tuning parameter is the number of components for dimension reduction 'ncomp'
# - CV

my_pls0_cv <- caret::train(my_formula, current_dataset_train ,method = "simpls",
                       trControl = trainControl(
                         method = "cv", number = 10, verboseIter = TRUE),
                       tuneGrid=expand.grid(ncomp=1:20)
)

# CV tuning plots

my_pls0_cv_dt <- melt(my_pls0_cv$results, id.vars = "ncomp")

ggplot(my_pls0_cv_dt, aes(x = ncomp, y = value)) +
  geom_line(color = "white", size = 2)+
  facet_wrap(~variable, nrow = 3, ncol=2, scales = "free")+
  theme_dark()+
  ggtitle("PLS tuning number of components")

# - Train

my_pls0 <- caret::train(my_formula, current_dataset_train ,method = "simpls",
                       trControl = trainControl(
                         method = "cv", number = 10, verboseIter = FALSE),
                       tuneGrid=expand.grid(ncomp=10)
)

my_pls0

# - Predicted values on test

my_pls0_pred <-predict(my_pls0,current_dataset_test)
my_pls0_rmsep <- round(sqrt(mean((yvar_test - my_pls0_pred)^2)),6)

# Predicted values plot

my_pls0_pred_dt <- data.table(observed = yvar_test,
                           predicted = my_pls0_pred)


ggplot(my_pls0_pred_dt, aes(x = observed, y = predicted))+
  geom_point(colour = "red", size = 2)+
  geom_segment(x = min(my_pls0_pred_dt[["observed"]]),
               y = min(my_pls0_pred_dt[["predicted"]]),
               xend = max(my_pls0_pred_dt[["observed"]]),
               yend = max(my_pls0_pred_dt[["predicted"]]), colour = "blue")+
  ggtitle("Observed vs Predicted in test set")+
  geom_text(aes(x = quantile(yvar_test,0.3),y = quantile(my_pls0_pred,0.9),
                label = paste0("RMSEP: ", my_pls0_rmsep)))


# -------------------------- VOTING STRATEGY ---------------------

# --- train data subsets


id1 <- (rbinom(length(yvar),1,0.5) == 1)
id2 <- (yvar >= median(yvar))

current_dataset_train1 <- current_dataset_train[id1]
current_dataset_train2 <- current_dataset_train[id2]


# --- Model 1

# Fit a regular pls model 

my_pls1_cv <- caret::train(my_formula, current_dataset_train1 ,method = "simpls",
                       trControl = trainControl(
                         method = "cv", number = 10, verboseIter = TRUE),
                       tuneGrid=expand.grid(ncomp=1:20)
)

my_pls1_cv_dt <- melt(my_pls1_cv$results, id.vars = "ncomp")

ggplot(my_pls1_cv_dt, aes(x = ncomp, y = value)) +
  geom_line(color = "white", size = 2)+
  facet_wrap(~variable, nrow = 3, ncol=2, scales = "free")+
  theme_dark()+
  ggtitle("PLS tuning number of components train1")


# - Train

my_pls1 <- caret::train(my_formula, current_dataset_train1 ,method = "simpls",
                        trControl = trainControl(
                          method = "cv", number = 10, verboseIter = FALSE),
                        tuneGrid=expand.grid(ncomp=10)
)

my_pls1

# - Predicted values on test

my_pls1_pred <-predict(my_pls1,current_dataset_test)



# --- Model 2



# Fit a regular pls model 

my_pls2_cv <- caret::train(my_formula, current_dataset_train2 ,method = "simpls",
                           trControl = trainControl(
                             method = "cv", number = 10, verboseIter = TRUE),
                           tuneGrid=expand.grid(ncomp=1:20)
)

my_pls2_cv_dt <- melt(my_pls2_cv$results, id.vars = "ncomp")

ggplot(my_pls2_cv_dt, aes(x = ncomp, y = value)) +
  geom_line(color = "white", size = 2)+
  facet_wrap(~variable, nrow = 3, ncol=2, scales = "free")+
  theme_dark()+
  ggtitle("PLS tuning number of components train2")


# - Train

my_pls2 <- caret::train(my_formula, current_dataset_train2 ,method = "simpls",
                        trControl = trainControl(
                          method = "cv", number = 10, verboseIter = FALSE),
                        tuneGrid=expand.grid(ncomp=9)
)

my_pls2

# - Predicted values on test

my_pls2_pred <-predict(my_pls2,current_dataset_test)



# --- Voting 

my_pls12_pred <- 0.5*my_pls1_pred + 0.5*my_pls2_pred
my_pls12_rmsep <- round(sqrt(mean((yvar_test - my_pls12_pred)^2)),6)


# Predicted values plot

my_pls12_pred_dt <- data.table(observed = current_dataset_test[[current_response_var]],
                              predicted = my_pls12_pred)


ggplot(my_pls12_pred_dt, aes(x = observed, y = predicted))+
  geom_point(colour = "red", size = 2)+
  geom_segment(x = min(my_pls0_pred_dt[["observed"]]),
               y = min(my_pls0_pred_dt[["predicted"]]),
               xend = max(my_pls0_pred_dt[["observed"]]),
               yend = max(my_pls0_pred_dt[["predicted"]]), colour = "blue")+
  ggtitle("Observed vs Predicted in test set by voting")+
  geom_text(aes(x = quantile(yvar_test,0.3),y = quantile(my_pls12_pred,0.9),
                label = paste0("RMSEP: ", my_pls12_rmsep)))



# ------------ Optional

my_pls0_pred_dt[["model"]] <- "unique"
my_pls12_pred_dt[["model"]] <- "ensemble"


my_pls012_pred_dt <- rbind(my_pls0_pred_dt, my_pls12_pred_dt)


p <- ggplot(my_pls012_pred_dt, aes(x = observed, y = predicted))+
  geom_point(colour = "red", size = 2)+
  geom_segment(x = min(my_pls0_pred_dt[["observed"]]),
               y = min(my_pls0_pred_dt[["predicted"]]),
               xend = max(my_pls0_pred_dt[["observed"]]),
               yend = max(my_pls0_pred_dt[["predicted"]]), colour = "blue")+
  ggtitle("Observed vs Predicted in test set - animation- ")+
  facet_wrap(~model)

p


