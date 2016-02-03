#rpart is used to build the classification tree and fscaret is used for other data processing methods like splitting the training data set
library(rpart)
library("fscaret")
library(pROC)

#Setting the working directory
setwd("/home/karthik/Documents/CSE675/Ford/Data")

print("*****Starting variable preparation phase*****")
TrainSet <- read.csv("fordTrain.csv")
TrainSet <- data.frame(TrainSet)
TestSet <- read.csv("fordTest.csv")
TestSet <- data.frame(TestSet)
SolutionSet <- read.csv("Solution.csv")
SolutionSet <- data.frame(SolutionSet)

print("*****Starting Model Generation Phase*****")
fit <- rpart(IsAlert ~ V11 + E9, method = "class", data = TrainSet)
#V11 + E9 + E5

#Plotting the classification and pruned classification trees
plot(fit,uniform= TRUE,main= "Classification Tree for Ford Challenge")
text(fit,use.n=TRUE, all=TRUE, cex=.7)
post(fit, title = "Classification Tree for Ford Challenge")

pfit <- prune(fit,cp= fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
plot(pfit,uniform=TRUE,main = 'Pruned Classification Tree For Ford Challenge')
text(pfit,use.n=TRUE, all= TRUE, cex=.7)
post(pfit, title = "Pruned Classification Tree For Ford Challenge")

print("*****Predicting Values*****")
myPrediction <- predict(pfit, newdata = TestSet, type= 'class')
predictionMetric <- data.frame(myPrediction)

ResultTree <- data.frame(actual = SolutionSet$Prediction, calculated = as.numeric(as.character(predictionMetric$myPrediction)))

print("***Confusion Matrix is as Follows***")
ConfusionTableDT = table(ResultTree$actual, ResultTree$calculated)
ConfusionFrameDT = data.frame(ConfusionTableDT)
colnames(ConfusionFrameDT) <- c("Actual","Predicted","Count")
hitsDT = ConfusionFrameDT[1,3] + ConfusionFrameDT[4,3]
total = sum(ConfusionFrameDT[,3])
accuracyDT = hitsDT/total
print("ACCURACY OF DECISION TREE MODEL IS AS FOLLLOWS")
print(signif(accuracyDT,digits=2))

print("***Plotting ROC for the classification tree***")
rpartPlot <- roc(ResultTree$calculated, ResultTree$actual, ci=TRUE, of="thresholds", thresholds=0.9)

with(rpartPlot, plot(main='ROC Plot for Decision Tree Model',specificities, sensitivities, type='l', xlim=c(1,0), ylim=c(0,1),xaxs='i',yaxs='i'))