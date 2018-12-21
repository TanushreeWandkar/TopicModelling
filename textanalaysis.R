#### Accessing the URL and the data from webpage ####
library('rvest')

url <- 'https://www.ontario.ca/laws/statute/98e15'
urldata <- read_html(url)

################# Pick html data on specific tags only###################

datapara <- html_nodes(urldata,'.paragraph')
head(datapara)
datap <- html_nodes(urldata,'p')
head(datap)
datss <- html_nodes(urldata,'.subsection')
head(datss)
datsubpara <- html_nodes(urldata,'.subpara')
head(datsubpara)
datapnote <- html_nodes(urldata,'.Pnote')
head(datapnote)
dataysec <- html_nodes(urldata,'.Ysection')
head(dataysec)


###################Pick the data from these tags#########################

datapara_txt <- html_text(datapara)
head(datapara_txt)

datap_txt <- html_text(datap)
head(datap_txt)

datss_txt <- html_text(datss)
head(datss_txt)

datsubpara_txt <- html_text(datsubpara)
head(datsubpara_txt)

datapnote_txt <- html_text(datapnote)
head(datapnote_txt)

dataysec_txt <- html_text(dataysec)
head(dataysec_txt)

################ Combine all the vectors and convert as dataframe #########################

d1 <- c(datapara_txt,datap_txt,datss_txt,datsubpara_txt,datapnote_txt,dataysec_txt)
class(d1)

eleactdf <- as.data.frame(d1,col.names = 'DataText')
head(eleactdf)

#############Text cleaning and preprocessing using tm package ##############
#install.packages("tm")
library(NLP)
library(tm)
#install.packages("SnowballC")
library(SnowballC)
library(RColorBrewer)
#install.packages("topicmodels")
library(topicmodels)
#install.packages("slam")
library(slam)

### convert it into corpus ###
docs <- Corpus(VectorSource(d1))

#### convert text into lower case letters ###
d2 <- tm_map(docs,tolower)

#### remove all the english stopwords ####
d3 <- tm_map(d2,removeWords, stopwords("english"))

#### remove the numbers #####
d4 <- tm_map(d3,removeNumbers)

#### remove punctuations ####
d5 <- tm_map(d4,removePunctuation)

#### remove strip whitespace ####
d6 <- tm_map(d5,stripWhitespace)

class(d2)

### view the content of cleaned corpus ####
d6[[67]]$content

######################

dtm <- DocumentTermMatrix(d6)
library(wordcloud)
m <- as.matrix(dtm)
v <- sort(colSums(m),decreasing=TRUE)
head(v,14)
words <- names(v)
d <- data.frame(word=words, freq=v)
pal2 <- brewer.pal(8,"Dark2")
set.seed(1234)
wordcloud(d$word,d$freq,min.freq=150,colors = pal2)

dev.new(width = 550, height = 330, unit = "px")
wordcloud(d$word,d$freq,min.freq=100,colors = pal2)

write.csv(d,file='wordfreq.csv')

#########################################

barplot(d[1:25,]$freq, las = 2, names.arg = d[1:25,]$word,
        col ="lightblue", main ="Most frequent words",
        ylab = "Word frequencies")

############## TOPIC MODELLING ################

### Creating tf-idf matrix

term_tfidf <- tapply(dtm$v/row_sums(dtm)[dtm$i], dtm$j, mean) * log2(nDocs(dtm)/col_sums(dtm > 0))
summary(term_tfidf)

dtmNew <- dtm[,term_tfidf >= 0.1]
dtmNew1 <- dtmNew[row_sums(dtmNew) > 0,]

#Deciding best K value using Log-likelihood method

best.model <- lapply(seq(2, 15, by = 1), function(d){LDA(dtmNew1, d)})
best.model.logLik <- as.data.frame(as.matrix(lapply(best.model, logLik)))

final_best_model <- data.frame(topics=c(seq(2,15, by=1)), 
                               log_likelihood=as.numeric(as.matrix(best.model.logLik)))

head(final_best_model)
library(ggplot2)
with(final_best_model,qplot(topics,log_likelihood,color="red"))

#Based on the graph, we can choose the best model
k=final_best_model[which.max(final_best_model$log_likelihood),1]

cat("Best topic number is k=",k)

####### LDA #######

k = 13
SEED = 1234

rowTotals <- apply(dtm, 1, sum) #Find the sum of words in each Document
dtm.new   <- dtm[rowTotals> 0, ] 
inspect(dtm.new)


my_TM =
  list(Gibbs = LDA(dtm.new, k = k, method = "Gibbs",
                   control = list(seed = SEED, burnin = 1000,
                                  thin = 100, iter = 1000)))


topicgibbs = topics(my_TM[["Gibbs"]],1)
termsgibbs = terms(my_TM[["Gibbs"]],30)

write.csv(termsgibbs,'gibbsterms30.csv')
