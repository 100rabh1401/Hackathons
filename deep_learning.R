path = "F:\\Padayi\\R code\\Deep learning"
setwd(path)

#load libraries
library(data.table)
library(mlr)

#set variable names
setcol <- c("age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "target")

#load data
train <- read.table("adult.data.txt", header=F, sep=",", col.names=setcol,
                    na.strings = c(" ?"), stringsAsFactors = F)
test <- read.table("adult.test.txt", header=F, sep=",", col.names=setcol,
                   skip=1, na.strings= c(" ?"), stringsAsFactors = F)

#View(train)
setDT(train)
setDT(test)
\
#Data Sanity
dim(train)
dim(test)

str(train)
str(test)


#checking missing values
table(is.na(train))
sapply(train, function(x) sum(is.na(x))/length(x))*100
table(is.na(test))
sapply(test, function(x) sum(is.na(x))/length(x))*100
#this function is checking the percentage of NA's in particular column

#Check Target variable
#binary in nature  check if data is imbalanced
train[,.N/nrow(train),target]
test[,.N/nrow(test),target]
#this function checks the target column and define the output in binary terms and there
#percentage

#Remove extra characters
test[,target := substr(target,start=1,stop=nchar(target)-1)]


#remove leading whitespaces
library(stringr)
char_col <- colnames(train)[sapply(test,is.character)]

for(i in char_col)
  set(train,j=i, value = str_trim(train[[i]],side="left"))

#set all characters variables as factor
fact_col <- colnames(train)[sapply(train,is.character)]

for(i in fact_col)
  set(train,j=i, value = factor(train[[i]]))

for(i in fact_col)
  set(test,j=i, value = factor(test[[i]]))

#impute missing values

imp1 <- impute(train,target = "target",
               classes = list(integer = imputeMedian(), factor = imputeMode()))
imp2 <- impute(test,target = "target",
               classes = list(integer = imputeMedian(), factor = imputeMode()))

train <- setDT(imp1$data)
test <- setDT(imp2$data)

#load the package
require(h2o)

#start h2o
localH2o <- h2o.init(nthreads = -1, max_mem_size = "20G")

#load data on H2o
trainh2o <- as.h2o(train)
testh2o <- as.h2o(test)

#set variables
y <- "target"
x <- setdiff(colnames(trainh2o),y)

#train the model - without hidden layer
deepmodel <- h2o.deeplearning(x = x
                              ,y = y
                              ,training_frame = trainh2o
                              ,standardize = T
                              ,model_id = "deep_model"
                              ,activation = "Rectifier"
                              ,epochs = 100
                              ,seed = 1
                              ,nfolds = 5
                              ,variable_importances = T)

#compute variable importance and performance
h2o.varimp_plot(deepmodel,num_of_features = 20)
h2o.performance(deepmodel,xval = T) #84.5 % CV accuracy
