

rm(list = ls())

library(jpeg)
library(data.table)


train_labels <- fread("data/digits-data/raw/train_labels.csv", header=TRUE)
train_images <- fread("data/digits-data/raw/train_images.csv", header=TRUE)


test_labels <- fread("data/digits-data/raw/test_labels.csv", header=TRUE)
test_images <- fread("data/digits-data/raw/test_images.csv", header=TRUE)

xtrain_raw <- copy(train_images)
ytrain_raw <- copy(train_labels)

xtest_raw <- copy(test_images)
ytest_raw <- copy(test_labels)


xtrain_raw[["filename"]] <- NULL
ytrain_raw[["filename"]] <- NULL

xtest_raw[["filename"]] <- NULL
ytest_raw[["filename"]] <- NULL

colnames_x_old <- colnames(xtrain_raw)

iim <- 30

im1 <- matrix(as.matrix(xtrain_raw[iim]), nrow=32, ncol=32, byrow = TRUE)
ytrain_raw[iim]
writeJPEG(im1, "test.jpeg", quality = 1)


presence_71 <- t(apply(ytrain_raw,1,function(x) x==1)) + t(apply(ytrain_raw,1,function(x) x==8))
images71_selected_ids <- which(apply(presence_71,1,sum)==1)

xtrain <- xtrain_raw[images71_selected_ids]
ytrain_raw2 <- ytrain_raw[images71_selected_ids]
ytrain_vector <- apply(ytrain_raw2,1,function(x) ifelse(1%in%x,1,0))


colnames(xtrain) <- paste0("P_",colnames_x_old)
xtrain[["one1_seven0"]] <- ytrain_vector



presence_71_test <- t(apply(ytest_raw,1,function(x) x==1)) + t(apply(ytest_raw,1,function(x) x==7))
images71_selected_ids_test <- which(apply(presence_71_test,1,sum)==1)

xtest <- xtest_raw[images71_selected_ids_test]
ytest_raw2 <- ytest_raw[images71_selected_ids_test]
ytest_vector <- apply(ytest_raw2,1,function(x) ifelse(1%in%x,1,0))

colnames(xtest) <- paste0("P_",colnames_x_old)
xtest[["one1_seven0"]] <- ytest_vector


fwrite(xtrain,"data/digits-data/digits_train.csv", sep = ";")
fwrite(xtest,"data/digits-data/digits_test.csv", sep = ";")
