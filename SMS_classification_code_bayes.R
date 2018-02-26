sms<-read.csv("sms2.csv",he=F,sep=";")
colnames(sms)<-c("type","text")
names(sms)
head(sms)

library(tm) #text mining package
library(e1071)
source("lift-roc-tab.R")
library(pROC)
library(wordcloud)

#corpus: vedi def di scarpa
#corpus 

corpus <- Corpus(VectorSource(sms$text))
inspect(corpus[1:3]) #tree first messages

#transform the corpus in lower case
clean_corpus <- tm_map(corpus, tolower)

#remove numbers
clean_corpus <- tm_map(clean_corpus, removeNumbers)

#remove punction
clean_corpus <- tm_map(clean_corpus, removePunctuation)

stopwords("en")[1:10]
clean_corpus <- tm_map(clean_corpus, removeWords,
                       stopwords("en"))

clean_corpus <- tm_map(clean_corpus, stripWhitespace)

#clean the corpus end



spam_indices <- which(sms$type == "spam")
spam_indices[1:3]
ham_indices <- which(sms$type == "ham")
ham_indices[1:3]



#only description
pal <- brewer.pal(6,"Dark2")
pal <- brewer.pal(9,"BuGn")
pal = brewer.pal(9,"Blues")
pal=rev(colorRampPalette(brewer.pal(9,"Blues"))(32)[seq(8,32,6)])
pal=colorRampPalette(brewer.pal(9,"Blues"))(32)[seq(8,32,6)]

wordcloud(clean_corpus, min.freq=40, scale=c(3,.5),colors=pal)

wordcloud(clean_corpus[ham_indices], min.freq=40, scale=c(3,.5),colors=pal)
?wordcloud

wordcloud(clean_corpus[spam_indices], min.freq=40, scale=c(4,.5),colors=pal)

#we want to build a spam filter! 

#Training and test set
#there is no temporal dipendence so i don't sample the rows

set.seed(1234)

random<- sample(1:nrow(sms),floor(0.75*nrow(sms)))
sms_train<-sms[random,]
sms_test <- sms[-random,]

table(sms_train$type)
table(sms_test$type)


corpus_train <- clean_corpus[random]
#corpus_test <- clean_corpus[setdiff(1:length(clean_corpus),random)]

corpus_test <-clean_corpus[-random]

#documentTermMatrix--> matrix where the rows are the messages and the
#columns are the words

sms_dtm <- DocumentTermMatrix(clean_corpus)
inspect(sms_dtm[1:4, 30:35])
dim(sms_dtm)
#training and test

# how many words appears 40 times or more
length(which(colSums(as.matrix(sms_dtm))>40))


#random<- sample(1:length(sms_dtm$dimnames$Docs),
#                floor(0.75*length(sms_dtm$dimnames$Docs)))


sms_dtm_train <- sms_dtm[random,]
sms_dtm_test <- sms_dtm[-random,]


#naive bayesian classifier
#cfr HASTIE

#if the number>0 then 1 
#otherwise 0
convert_count <- function(x){
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

  
five_times_words <- findFreqTerms(sms_dtm_train,words_ )
length(five_times_words)
five_times_words[1:5] #first 5


sms_dtm_train <- DocumentTermMatrix(corpus_train,
                                    control=list(dictionary = five_times_words))
sms_dtm_test <- DocumentTermMatrix(corpus_test,
                                   control=list(dictionary = five_times_words))



sms_dtm_train <- apply(sms_dtm_train, 2, convert_count)
sms_dtm_train[1:4, 30:35]

sms_dtm_test <- apply(sms_dtm_test, 2, convert_count)
sms_dtm_test[1:4, 30:35]


#the x and the response (spam ham ecc.)
classifier <- naiveBayes(sms_dtm_train, sms_train$type)
class(classifier)


predictions <- predict(classifier, newdata=sms_dtm_test)



table(predictions,sms_test$type) #use tabella sommario

tabella.sommario(predictions,sms_test$type)
a.naive<-lift.roc(predictions,as.numeric(sms_test$type)-1,type="crude")

roc_obj<-roc(as.numeric(sms_test$type)-1,as.numeric(predictions)-1) 
auc(roc_obj)

#NB laplace=1
laplace_ <- c(0.001,0.01,0.1,1)
graph_laplace <- rep(0,4)
i=1
for (l in laplace_ ){
  B.clas <- naiveBayes(sms_dtm_train, sms_train$type,laplace = l)
  class(B.clas)
  B.preds <- predict(B.clas, newdata=sms_dtm_test)
  s=table(B.preds, sms_test$type)
  graph_laplace[i] = (s[1,1]+s[2,2])/sum(s)
  i = 1+i
}

tab<-cbind(laplace_,graph_laplace)


plot(graph_laplace,type="l",ylab="accuracy",xlab = "Value of Beta Parmeters"
     ,main="Accuracy of new models")


#select only the words that appears at least 5 times
words_accur <- rep(0,7)
n_words = 1

for (words_ in c(5,7,10,12,15,17,20)){
  five_times_words <- findFreqTerms(sms_dtm_train,words_ )
  sms_dtm_train_1 <- DocumentTermMatrix(corpus_train,
                                        control=list(dictionary = five_times_words))
  sms_dtm_test_1 <- DocumentTermMatrix(corpus_test,
                                       control=list(dictionary = five_times_words))
  
  sms_dtm_train_1 <- apply(sms_dtm_train_1, 2, convert_count)
  
  sms_dtm_test_1 <- apply(sms_dtm_test_1, 2, convert_count)
  
  #the x and the response (spam ham ecc.)
  classifier <- naiveBayes(sms_dtm_train_1, sms_train$type,laplace = laplace_[which.max(graph_laplace)])
  class(classifier)
  
  
  predictions <- predict(classifier, newdata=sms_dtm_test_1)
  w = table(predictions,sms_test$type) #use tabella sommario
  words_accur[n_words] = (w[1,1]+w[2,2])/sum(w)
  
  n_words = n_words+1
}

tab_words<-cbind(c(5,7,10,12,15,17,20),words_accur)

plot(tab_words,type="l",ylab = "Accuracy",xlab = "N",main="Accuracy for different N")

B.clas <- naiveBayes(sms_dtm_train, sms_train$type,laplace = laplace_[which.max(graph_laplace)])
class(B.clas)
B.preds <- predict(B.clas, newdata=sms_dtm_test)
s=table(B.preds, sms_test$type)

tabella.sommario(B.preds,sms_test$type)
b<-lift.roc(B.preds,as.numeric(sms_test$type)-1,type="crude")

roc_obj<-roc(as.numeric(sms_test$type)-1,as.numeric(B.preds)-1) 
auc(roc_obj)

plot(a.naive[[3]], a.naive[[4]], type="l", 
     
     xlab="1- specificity", ylab="sensibility",col=2,main = "ROC Curve")
abline(0,1,lty=2)
lines(b[[3]], b[[4]], type="l", col=3)
legend("bottomright",c("Naive Classifier","With a priori distribution"),
       col=c(2:3),
       lty=1,cex=0.7)
