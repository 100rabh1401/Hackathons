setwd("F:\\Padayi\\R code\\Text Analytics using R(Spam data)")

#install all required packages
#install.packages(c("ggplot2","e1071","caret","quanteda","irlba","randomForest"))

#Load the data .csv into R studio
spam.raw <- read.csv("spam_text.csv",stringsAsFactors = F)
#View(spam.raw)

#Clean up the data frame
names(spam.raw) <- c("Label","Text")
#View(spam.raw)

#Check data to see if there is missing values
length(which(!complete.cases(spam.raw)))

#Convert our class label into as factor
spam.raw$Label <- as.factor(spam.raw$Label)

###First step tp explore the data
#lets take a look at the distribution of class labels
prop.table(table(spam.raw$Label))

#Next up, lets get a feel for the distribution of text lengths of the SMS
#message by adding a new feature for the length of each message
spam.raw$TextLength <- nchar(spam.raw$Text)
summary(spam.raw$TextLength)

#Visualize the distribution with ggplot2, adding segmentation for ham/spam
library(ggplot2)
ggplot(spam.raw,aes(x = TextLength, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Textcount", x = "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")

#Split data into train and test

#Our data has non trivial class imbalance, we'll use the mighty caret 
#package to create a random train/test split that ensures the correct ham/spam
#class label proportions (ie for stratified sampling)
library(caret)
#help(package="caret)

set.seed(12345)
indexes <- createDataPartition(spam.raw$Label,times = 1,
                               p = 0.7, list = FALSE)

train <- spam.raw[indexes,]
test <- spam.raw[-indexes,]

#Verify proportions
prop.table(table(train$Label))
prop.table(table(test$Label))

#Text analytics requires lots of data exploration, data preprocessing,
#data wrangling. 
#HTML escaped ampersand character
train$Text[23]

#Quanteda package has many useful aafunstions for quickly and easily working with
#text data
#install.packages("quanteda")
library(quanteda)

#tokenize SMS text messages
train.tokens <- tokens(train$Text, what="word",
                       remove_numbers = T , remove_punct = T,
                       remove_symbols = T , remove_hyphens = T)

#take a look at specific message and see how it performs
train.tokens[[123]]

#Lower case the tokens
train.tokens <- tokens_tolower(train.tokens)
train.tokens[[123]]

#remove the stopwords
train.tokens <- tokens_select(train.tokens, stopwords(),
                              selection = "remove")
train.tokens[[123]]

#perform stemming on the tokens
train.tokens <- tokens_wordstem(train.tokens, language = "english")
train.tokens[[123]]

#create first bag of words model
train.tokens.dfm <- dfm(train.tokens,tolower=F)

#transform to a matrix and inspect
train.tokens.matrix <- as.matrix(train.tokens.dfm)
dim(train.tokens.matrix)

#investigate the effect of stemming
colnames(train.tokens.matrix)[1:50]

#leveraging on cross validation

#setup the feature dataframe with labels
train.tokens.df <- cbind(Label = train$Label, as.data.frame(train.tokens.dfm))

#Often tokenizatin requires some additional pre processing
names(train.tokens.df)[c(146,148,235,238)]

#Cleanup column names
names(train.tokens.df) <- make.names(names(train.tokens.df))

#use caret to create stratified folds for 10 fold cross validation repeated
# 3 times(ie 30 random stratified samples)
set.seed(1624)
cv.folds <- createMultiFolds(train$Label , k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)

#for parallel processing
#install.packages("doSNOW")
library(doSNOW)

#time with code execution
start.time <- Sys.time()

#create a clsuter to work on 3 logical cores
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

#Algorithm

rpart.cv.1 <- train(Label ~ ., data = train.tokens.df , method = "rpart",
                    trControl = cv.cntrl , tuneLength = 7)

#processing is done stop cluster
stopCluster(cl)

#Total time of execution
total.time <- Sys.time() - start.time
total.time

#Check out our results
rpart.cv.1

#######################################################################


################################################################
#our function for calculating relative term frequency(TF)
term.frequency <- function(row){
  row/sum(row)
}

#our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col){
  corpus.size <- length(col)
  doc.count <- length(which(col>0))

  log10((corpus.size/doc.count))  
}

#our function for calculating tf-idf
tf.idf <- function(tf,idf) {
  tf * idf
}

#first step, normalize all documents via TF
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)

#second step, calculate the IDF vector that will use 
#both for training data and for test data
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)

#lastly,calculate TF-IDF for our training corpus
train.tokens.tfidf <- apply(train.tokens.df,2,tf.idf,idf=train.tokens.idf)
dim(train.tokens.tfidf)

#transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)

#check for incomplete cases
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]

#fix incomplete cases
train.tokens.tfidf[incomplete.cases,] <- rep(0.0,ncol(train.tokens.tfidf))
dim(train.tokens.idf)
sum(which(!complete.cases(train.tokens.tfidf)))

#make a clean data frame usinf the same process as before
train.tokens.tfidf.df <- cbind(Label=train$Label, as.data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

##Algorithm

rpart.cv.2 <- train(Label ~ ., data = train.tokens.tfidf.df , method = "rpart",
                    trControl = cv.cntrl , tuneLength = 7)
rpart.cv.2


########################################################################
#N grams alow us to augment our document term frequency matrices with
#word ordering. Often leads to increased performance

#Add bigrams to our feature matrix
train.tokens <- tokens_ngrams(train.tokens,n=1:2)
train.tokens[[456]]

#transform to dfm and then a matrix
train.tokens.dfm <- dfm(train.tokens, tolower = F)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
train.tokens.dfm

#normalize all documents via TF
#first step, normalize all documents via TF
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)

#second step, calculate the IDF vector that will use 
#both for training data and for test data
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)

#lastly,calculate TF-IDF for our training corpus
train.tokens.tfidf <- apply(train.tokens.df,2,tf.idf,idf=train.tokens.idf)
dim(train.tokens.tfidf)

#transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)

#check for incomplete cases
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]

#fix incomplete cases
train.tokens.tfidf[incomplete.cases,] <- rep(0.0,ncol(train.tokens.tfidf))
dim(train.tokens.idf)
sum(which(!complete.cases(train.tokens.tfidf)))

#make a clean data frame usinf the same process as before
train.tokens.tfidf.df <- cbind(Label=train$Label, as.data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

##Algorithm

rpart.cv.3 <- train(Label ~ ., data = train.tokens.tfidf.df , method = "rpart",
                    trControl = cv.cntrl , tuneLength = 7)
rpart.cv.3

#clean an unused objects in memory
gc()


###########################################################################
