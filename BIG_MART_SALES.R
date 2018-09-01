#Set working directory
setwd("F:/Padayi/R code/AV-BIg mart sales--PCA")

#Load train and test dataset
train <- read.csv("Train_UWu5bXk.csv")
test <- read.csv("Test_u94Q5KV.csv")

#Checking the summary of dataset
summary(train)
summary(test)

#Adding a column in test dataset
test$Item_Outlet_Sales <- 1

#Combine the dataset
combi <- rbind(train,test)

#impute the missing values
summary(combi)

combi$Item_Weight[is.na(combi$Item_Weight)] <- median(combi$Item_Weight,na.rm=T)

#Impute 0 with median
combi$Item_Visibility <- ifelse(combi$Item_Visibility==0,median(combi$Item_Visibility),
                                combi$Item_Visibility)

#Find mode and impute
names(sort(-table(combi$Outlet_Size)))
table(combi$Outlet_Size,combi$Outlet_Type)
levels(combi$Outlet_Size)[1] <- "Other"

#Removing the dependent and identifier variables
my_data <- subset(combi,select = -c(Item_Outlet_Sales,Item_Identifier,Outlet_Identifier))

#Check available variables
colnames(my_data)

#Check variables class
str(my_data)

#Converting categorical variables into numeric variables
#load library
library(dummies)

#Create dummy dataframe
new_my_data <- dummy.data.frame(my_data,names = c("Item_Fat_Content","Item_Type",
                                                  "Item_Type","Outlet_Size",
                                                  "Outlet_Location_Type","Outlet_Type"))

#Check the dataset
str(new_my_data)

#Divide the new dataset into test and train
pca.train <- new_my_data[1:nrow(train),]
pca.test <- new_my_data[-(1:nrow(train)),]

#Principal component analysis
prin_comp <- prcomp(pca.train, scale. = T)
names(prin_comp)

prin_comp$center

prin_comp$scale

prin_comp$rotation

prin_comp$rotation[1:5,1:4]

dim(prin_comp$x)

#Plotting the resulting Principal Component
biplot(prin_comp, scale=0)

#Compute the SD of each principal component
std_dev <- prin_comp$sdev

#Compute variance
pr_var <- std_dev**2

#Check variance of first 10 component
pr_var[1:10]

#proportion of variance explained
prop_var <- pr_var/sum(pr_var)

prop_var[1:20]
#this shows the percentage of variance explained 1st,2nd.... Principal components

#A screeplot is used to access components or factors which explains
#more variablity in tht data
#it represents value in desecnding order

#scree plot
plot(prop_var, xlab = "Principal COmponent",
     ylab = "Proportion of variance explained",
     type="b")
#This shows us that 98.4% variance is explained by ~28 components


#Cumulative screeplot
plot(cumsum(prop_var), xlab = "Principal COmponent",
     ylab = "Proportion of variance explained",
     type="b")
#this shows that 98% variance is explained using 28 components
#therefore in this case we will select number of components as 28[PC1 to PC28]
#thus for modelling we will use this 28 components as predictors variables


#Add a training set with PRincipal Components
train.data <- data.frame(Item_Outlet_Sales=train$Item_Outlet_Sales,prin_comp$x)

#We are interested in first 28 PCAs
train.data <- train.data[,1:29]

#Run a decision tree
library(rpart)
rpart.model <- rpart(Item_Outlet_Sales ~ ., data=train.data, method= "anova")
rpart.model

#transform test into PCA
test.data <- predict(prin_comp,newdata = pca.test)
test.data <- as.data.frame(test.data)
  
#select the first 28 components
test.data <- test.data[,1:28]

#make prediction on test data
rpart.prediction <- predict(rpart.model,test.data)

#Exporting file for submission
write.csv(rpart.prediction,"PRediction.csv")
