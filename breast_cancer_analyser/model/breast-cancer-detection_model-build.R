# -----------------------------------------------------------------------

# -------------       BREAST CANCER MODEL FOR SHINY  --------------------

# -----------------------------------------------------------------------


rm(list = ls())

# --- Libraries ---

library(data.table)
library(caret)





# --- Read data sets ---

breast <- fread("model/breast-cancer-data/breast-cancer-data.csv", sep = ";") # Yvar: diagnosis (binary)
breast_trainIndex <- createDataPartition(breast[["diagnosis"]], p = .8, 
                                         list = FALSE, 
                                         times = 1)
current_dataset_train <- breast[breast_trainIndex[,1]]
current_dataset_test <- breast[-breast_trainIndex[,1]]
rm(breast, breast_trainIndex)

current_response_var <- "diagnosis"

current_dataset_train[[current_response_var]] <- as.factor(current_dataset_train[[current_response_var]])
current_dataset_test[[current_response_var]] <- as.factor(current_dataset_test[[current_response_var]])

yvar <- current_dataset_train[[current_response_var]]
yvar_test <- current_dataset_test[[current_response_var]]



# --------------------------------- CART MODEL TRAINING --------------------------

regressors <- c("concave_points_worst",
                "perimeter_worst"          ,
                "concave_points_mean"      ,
                "concavity_worst"          ,
                "texture_worst"            ,
                "smoothness_worst"          ,
                "area_se"                   ,
                "compactness_se"            ,
                "texture_se"                ,
                "texture_mean"              ,
                "smoothness_se"            
                )                

my_formula <- as.formula(paste0(current_response_var, " ~ ",paste(regressors, collapse = " + ")))




# ------------------- BOOSTING ADABoost -----------------------------------

# - TRAIN MODEL

my_cart1 <- caret::train(my_formula, current_dataset_train, method = "AdaBoost.M1",
                         trControl = trainControl(
                         method = "cv", number = 10, verboseIter = TRUE),
                         tuneGrid=expand.grid(mfinal = 3, coeflearn = "Breiman", maxdepth = 5),
                         boos = TRUE
                         
                         
)



# --- ELEMENTS OF THE MODEL FOR A USER FRIENDLY APP

my_cart1_model <- my_cart1$finalModel


# - Fitted and Predicted
my_cart1_fitted <-predict(my_cart1,current_dataset_train, type = "raw")
my_cart1_pred <-predict(my_cart1,current_dataset_test, type = "raw")

# - Confusion matrices
confusion_train <- confusionMatrix(my_cart1_fitted,yvar, positive = "1")
confusion_test <- confusionMatrix(my_cart1_pred,yvar_test, positive = "1")

my_cart1_summary_performance <- rbind(data.table(Type = "Training", 
                                                 N = nrow(current_dataset_train),
                                                 Accuracy = confusion_train$overall[1],
                                                 Sensitivity = confusion_train$byClass[1],
                                                 Specificity = confusion_train$byClass[2]),
                                      data.table(Type = "Test", 
                                                 N = nrow(current_dataset_test),
                                                 Accuracy = confusion_test$overall[1],
                                                 Sensitivity = confusion_test$byClass[1],
                                                 Specificity = confusion_test$byClass[2]))
           

# - Variable importance 


my_varimp_tree1 <- sort(varImp(my_cart1_model$trees[[1]])$Overall, decreasing = T)
my_cart1_varimp_tree1 <- data.table(Tree = "Tree 1", Variable = rownames(varImp(my_cart1_model$trees[[1]])), Importance = my_varimp_tree1/my_varimp_tree1[1])

my_varimp_tree2 <- sort(varImp(my_cart1_model$trees[[2]])$Overall, decreasing = T)
my_cart1_varimp_tree2 <- data.table(Tree = "Tree 2", Variable = rownames(varImp(my_cart1_model$trees[[2]])), Importance = my_varimp_tree2/my_varimp_tree2[1])

my_varimp_tree3 <- sort(varImp(my_cart1_model$trees[[3]])$Overall, decreasing = T)
my_cart1_varimp_tree3 <- data.table(Tree = "Tree 3", Variable = rownames(varImp(my_cart1_model$trees[[3]])), Importance = my_varimp_tree3/my_varimp_tree3[1])

varimp_summary <- rbind(my_cart1_varimp_tree1,my_cart1_varimp_tree2,my_cart1_varimp_tree3)


model_output <- list(trained_model = my_cart1,
                     summary_performance = my_cart1_summary_performance,
                     variable_importance = varimp_summary)

# --- SAVE MODEL ELEMENTS

save(model_output, file = "model/breast_cancer_model.RData")


# --- SAMPLE OF UNLABELED DATA FOR PREDICTION
sample_unlabeled_data <- data.table(id = as.character(1:10), model_output$trained_model$trainingData[1:20,-1])
fwrite(sample_unlabeled_data, file = "model/breast-cancer-data/newdata_unlabeled.csv", sep = ";")
