setwd("F:/Padayi/R code/AV-CAB-surge price--DT")
file <- read.csv("train_63qYitG.csv",na.strings = c("","NA"))#,stringsAsFactors = F)
file1 <- read.csv("test_XaoFywY.csv",na.strings = c("","NA"))
head(file)
dim(file)
str(file)
sapply(file,function(x) sum(is.na(x)))
summary(file)
unique(file$Surge_Pricing_Type)

#Checking missing values
table(is.na(file))
sapply(file, function(x) sum(is.na(x))/length(x))*100

#Replacing the NA values
names(sort(-table(file$Type_of_Cab))[1])
file$Type_of_Cab[which(is.na(file$Type_of_Cab))] <- "B"

summary(file$Customer_Since_Months)
file$Customer_Since_Months[which(is.na(file$Customer_Since_Months))] <- mean(file$Customer_Since_Months,na.rm=T)

summary(file$Life_Style_Index)
file$Life_Style_Index[which(is.na(file$Life_Style_Index))] <- mean(file$Life_Style_Index,na.rm=T)

summary(file$Confidence_Life_Style_Index)
names(sort(-table(file$Confidence_Life_Style_Index))[1])
file$Confidence_Life_Style_Index[which(is.na(file$Confidence_Life_Style_Index))] <- "B"

summary(file$Var1)
file$Var1[which(is.na(file$Var1))] <- mean(file$Var1,na.rm=T)

#Checking for NA again
anyNA(file)
sapply(file, function(x) sum(is.na(x))/length(x))*100

#Replacing the NA values as replaced in train dataset
summary(file1)
#Replacing the NA values
names(sort(-table(file1$Type_of_Cab))[1])
file1$Type_of_Cab[which(is.na(file1$Type_of_Cab))] <- "B"

summary(file1$Customer_Since_Months)
file1$Customer_Since_Months[which(is.na(file1$Customer_Since_Months))] <- mean(file1$Customer_Since_Months,na.rm=T)

summary(file1$Life_Style_Index)
file1$Life_Style_Index[which(is.na(file1$Life_Style_Index))] <- mean(file1$Life_Style_Index,na.rm=T)

summary(file1$Confidence_Life_Style_Index)
names(sort(-table(file1$Confidence_Life_Style_Index))[1])
file1$Confidence_Life_Style_Index[which(is.na(file1$Confidence_Life_Style_Index))] <- "B"

summary(file1$Var1)
file1$Var1[which(is.na(file1$Var1))] <- mean(file1$Var1,na.rm=T)

#Building the model
file <- file[,-1]
CART1 <- rpart::rpart(Surge_Pricing_Type ~.,data=file,method = "class")
rpart.plot::prp(CART1)


#Predicting the output
predictCART1 <- predict(CART1,newdata = file1,type = "class")

table(predictCART1)


#######################################################################
#Making new model by splitting train data into test and train
file
library(caTools)
aa <- sample.split(file$Surge_Pricing_Type, SplitRatio = 0.7)
train <- subset(file, aa==T)
test <-  subset(file, aa==F)

CART2 <- rpart::rpart(Surge_Pricing_Type ~., data=train, method="class")
rpart.plot::prp(CART2)

predictCART2 <- predict(CART2,newdata = test,type = "class")

table(test$Surge_Pricing_Type,predictCART2)

library(caret)
confusionMatrix(test$Surge_Pricing_Type,predictCART2)


#Making model by dropping the val1 column which has more than 50% of NA
file_d <- file
summary(file_d)
file_d <- file_d[,-9]
#######################################################################
#Making new model by splitting train data into test and train
file
library(caTools)
set.seed(123)
aa <- sample.split(file_d$Surge_Pricing_Type, SplitRatio = 0.7)
train <- subset(file_d, aa==T)
test <-  subset(file_d, aa==F)

CART2 <- rpart::rpart(Surge_Pricing_Type ~., data=train, method="class")
rpart.plot::prp(CART2)

predictCART2 <- predict(CART2,newdata = test,type = "class")

table(test$Surge_Pricing_Type,predictCART2)

library(caret)
confusionMatrix(test$Surge_Pricing_Type,predictCART2)


#Developing model using randomforest
library(randomForest)
fit <- randomForest(Surge_Pricing_Type ~.,data = train)