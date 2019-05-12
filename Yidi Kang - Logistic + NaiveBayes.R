
####################################
# Viome Data Challenge - Yidi Kang #
####################################

require(tidyverse)
require(gridExtra) # operate data
require(ggplot2)   # visuallization
require(caret)     # split data
require(broom)     # logistic regression
require(pROC)      # roc curve
require(e1071)     # naiveBayes

# for plots
theme.info <- theme(plot.title = element_text(size=16, hjust=0.5),
                    axis.title = element_text(size=14),
                    axis.text = element_text(size=14))

# data loading
x <- read_delim("data.csv",delim=",")
y <- read_delim("labels.csv",delim=",")
uid <- read_delim("user_ids.csv",delim=",")

# Pre-process data
sumlist <- apply(x,1,sum)
data <- cbind(uid,x/sumlist)
# NA checking 
temp <- sapply(data,function(x) sum(is.na(x)))
any(temp !=0 )


# EDA
# mean and variance of features
mean <- apply(data[,-1], 2, mean)
var <- apply(data[,-1], 2, var)
cbind(mean,var)
# correlation matrix
corm <- cor(data[,-1])
corm
# Class(label) balance checking
p.1 <- y %>% ggplot(aes(x=label))+
  geom_histogram(bins=2)+
  ggtitle("check the class balancing")+
  theme.info
p.1

#######################
# Logistic regression #
#######################

# PCA: feature demenssion reducing
data.pr <- princomp(data[,-1],cor=TRUE)
plot(data.pr, type = "l")
## from the plot, we can see there is few improvement after Comp.5.
## so we choose to include Comp.1 to Comp.4 first.

pca.data <- predict(data.pr)
pca.data <- as.data.frame(pca.data[,c(1,2,3,4)])
pca.data$label <- y$label

# Split data into train and test
dpart <- createDataPartition(pca.data$label,p=0.1,list = F)
pca.test <- pca.data[dpart,]
pca.train <- pca.data[-dpart,]

# Logistic Regression model building
## set significant level: alpha = 0.1
glm.1 <- glm(label~.,data = pca.train,family = binomial(link='logit'))

summary(glm.1)
## For Null Deviance, the chi-square value is 67701; Residual Deviance is 67300.
## in summay(model), it's overall F-test for model, our H0 is all slope == 0; Ha is at least 1 slope != 0.
## All 4 p-values smaller than alpha, so we reject H0, all 4 Comp. are statistically significant.

# Anova : Analysis of Deviance Table
anova(glm.1,test = "Chisq")
## Comp.1+2+3 contribute most to reduce the resid.deviance, which inprove our model.


roc1.info <- roc(response=pca.train$label, 
                 predictor=glm.1$fitted.values, 
                 plot=TRUE, las=TRUE, 	
                 legacy.axes=TRUE, lwd=5,
                 main="ROC for Logistic Regression", cex.main=1.3, cex.axis=1.1, cex.lab=1.1)
## ROC cureve almost overlaps the 45 degree line, it's a really poor model.

# use test data to see the Accuracy
pt <- predict(glm.1,newdata=pca.test,type="response")
pt[pt>0.5]<-1
pt[pt<=0.5]<-0
print(paste("Accuracy of our logistica regression model on 4 primary components is",round(sum(pt==pca.test$label)/nrow(pca.test),2)))
## Obviously model has very low accuracy, so now we try another model.
## (I also tried to incraese the feature dimenssion in PCA, but didn't really helped.)

#########################
# NaiveBayes classifier #
#########################

# split data into train and test
data$label <- as.factor(y$label)
dpart <- createDataPartition(data$label,p=0.1,list = F)
test <- data[dpart,]
train <- data.frame(data[-dpart,])

# model building
nb.1 <- naiveBayes(label~.,data=train)

# plot ROC curve to see True Positive rate (y-axis) and False positive rate (x-axis) at different threshold value.
pred.nbtrain <- predict(nb.1,train,type="raw")
dp <- data.frame(pred.nbtrain)

roc2.info <- roc(response=train$label, 
                 predictor=dp$X1, 
                 plot=TRUE, las=TRUE, 	
                 legacy.axes=TRUE, lwd=5,
                 main="ROC for NaiveBayes", cex.main=1.3, cex.axis=1.1, cex.lab=1.1)
## ROC curve clearly betternthan Logistic regression.

# find the best threshold
best <- coords(roc2.info, x="best", input="threshold", ret=c("threshold", "sensitivity", "specificity"))
best

# find AUC of our model
area <- auc(response=train$label, predictor=dp$X1)
print(paste("AUC of ROC is",round(area,2)))
## if we are trying to predict sample has disease or not, we should consider the severity of Type 1 and Type 2 error.
## which means we should reduce False Negative error (Type 2): when the sample actually have disease but we predict negative.
## In our case, we just need to find the largest AUC, with regardless of the reducing Type 2 error problem.

# Using best threshold, Get accuracy on test data
pred.nb1 <- predict(nb.1,test,type="raw")
dp2 <- data.frame(pred.nb1)
predc2 <- rep(0,times=nrow(test))
predc2[which(dp2$X1>best[1])]<- 1
print( paste("when we use best threshold",best[1],", our model's accuracy on test data is",round(sum(predc2==test$label)/nrow(test),2)*100,"%"))

############################
# Thank you for your time :)
