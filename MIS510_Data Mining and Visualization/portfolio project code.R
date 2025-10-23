setwd("C:/Users/jdhum/OneDrive/Documents/Grad School/MIS510/sandbox")
germancredit.df<-read.csv("GermanCredit.csv")
library(gains)
library(MASS)
library(neuralnet)
library(caret)
library(pROC)

head(germancredit.df)
#create set of numeric variables 
germancredit.num<-germancredit.df[,c(3,11,14,23,27,29)]
#analysis of numeric variables
data.frame(mean=sapply(germancredit.num,mean,na.rm=TRUE),
           sd=sapply(germancredit.num,sd,na.rm=TRUE),
           min=sapply(germancredit.num,min,na.rm=TRUE),
           max=sapply(germancredit.num,max,na.rm=TRUE),
           median=sapply(germancredit.num,median,na.rm=TRUE),
           length=sapply(germancredit.num,length),
           miss.val=sapply(germancredit.num,
                           function(x)sum(length(which(is.na(x))))))
#produce histograms of numeric variables
for(i in c(3,11,14,23,27,29)){
  cname=colnames(germancredit.df)[i]
  hist(germancredit.df[,i],xlab=cname,main=c("histogram of ",cname))
}

#produce barplot of categorical variables
for(i in c(2,4,12,13,20,28)){
  cname=colnames(germancredit.df)[i]
  barplot(table(germancredit.df[,i]),xlab=cname,main=c("barplot of ",cname))
}

#barplot of purpose variable (dummy variables)
Purpose<-c(sum(germancredit.df$NEW_CAR),sum(germancredit.df$USED_CAR),
           sum(germancredit.df$FURNITURE),sum(germancredit.df$RADIO.TV),
           sum(germancredit.df$EDUCATION),sum(germancredit.df$RETRAINING))
Purpose.names<-c('New Car','Used Car', 'Furniture','Radio/TV','Education','Retraining')

barplot(Purpose,names.arg = Purpose.names, las=2)

#partition data set for logistic regression
set.seed(1)
credittrain.index<-sample(c(1:dim(germancredit.df)[1]),dim(germancredit.df)[1]*0.6)
credittrain.df<-germancredit.df[credittrain.index,]
creditvalid.df<-germancredit.df[-credittrain.index,]

#create logistic regression model
logit.reg<-glm(RESPONSE~., data=credittrain.df[,-1], family="binomial")
summary (logit.reg)
#validate model
logit.reg.pred<-predict(logit.reg, creditvalid.df[,-1], type="response")

#lift chart using model with all predictors
gain<-gains(creditvalid.df$RESPONSE, logit.reg.pred, groups=length(logit.reg.pred))
plot(c(0,gain$cume.pct.of.total*sum(creditvalid.df$RESPONSE))~c(0,gain$cume.obs),
     xlab="# of cases", ylab="cumulative", main="LR All predictors", type ="l")
lines(c(0,sum(creditvalid.df$RESPONSE))~c(0,dim(creditvalid.df)[1]), lty=2)

#confusion matrix for logistic regression model
logit.reg.pred.bin<-ifelse(logit.reg.pred>0.5,1,0)
confusionMatrix(factor(logit.reg.pred.bin),factor(creditvalid.df$RESPONSE))
#ROC curve for logistic regression model
r<-roc(creditvalid.df$RESPONSE,logit.reg.pred)
plot.roc(r)
auc(r)


#eliminate unneeded variables using Stepwise AIC model selection
step.model<-stepAIC(logit.reg, trace=FALSE)
coef(step.model)
#make prediction set with improved model
step.model.pred<-predict(step.model, creditvalid.df[,-1], type = "response")

#lift chart using stepwise selection method
gain2<-gains(creditvalid.df$RESPONSE, step.model.pred, groups=length(step.model.pred))
plot(c(0,gain2$cume.pct.of.total*sum(creditvalid.df$RESPONSE))~c(0,gain2$cume.obs),
     xlab="# of cases", ylab="cumulative", main="LR Stepwise Selection", type ="l")
lines(c(0,sum(creditvalid.df$RESPONSE))~c(0,dim(creditvalid.df)[1]), lty=2)

#confusion matrix for stepwise selection model
step.model.pred.bin<-ifelse(step.model.pred>0.5,1,0)
confusionMatrix(factor(step.model.pred.bin),factor(creditvalid.df$RESPONSE))
#ROC curve for stepwise selection model
r<-roc(creditvalid.df$RESPONSE,step.model.pred)
plot.roc(r)
auc(r)

#scaling data and removing obs variable
maxs <- apply(germancredit.df[,-1], 2, max) 
mins <- apply(germancredit.df[,-1], 2, min)
scaledcredit.df <- as.data.frame(scale(germancredit.df[,-1], center = mins, scale = maxs - mins))

#partition data set
set.seed(1)
scaledtrain.index<-sample(c(1:dim(scaledcredit.df)[1]),dim(scaledcredit.df)[1]*0.6)
scaledtrain.df<-scaledcredit.df[scaledtrain.index,]
scaledvalid.df<-scaledcredit.df[-scaledtrain.index,]



#neural net
neural.net<-neuralnet(RESPONSE~.,data = scaledtrain.df, hidden=25)
#analysis of neural net performance on training data
nn.train.pred=compute(neural.net, scaledtrain.df)
nn.train.pred.bin<-ifelse(nn.train.pred$net.result>0.5,1,0)
confusionMatrix(factor(nn.train.pred.bin),factor(credittrain.df$RESPONSE))

#validate neural net
nn.valid.pred=compute(neural.net, scaledvalid.df)
nn.valid.pred.bin<-ifelse(nn.valid.pred$net.result>0.5,1,0)
confusionMatrix(factor(nn.valid.pred.bin),factor(creditvalid.df$RESPONSE))


r<-roc(creditvalid.df$RESPONSE,nn.valid.pred$net.result)
plot.roc(r)
auc(r)

gain3<-gains(scaledvalid.df$RESPONSE, nn.valid.pred$net.result, groups=length(nn.valid.pred$net.result))
plot(c(0,gain3$cume.pct.of.total*sum(creditvalid.df$RESPONSE))~c(0,gain3$cume.obs),
     xlab="# of cases", ylab="cumulative", main="NN 2 hidden layer 20,13 nodes", type ="l")
lines(c(0,sum(creditvalid.df$RESPONSE))~c(0,dim(creditvalid.df)[1]), lty=2)
