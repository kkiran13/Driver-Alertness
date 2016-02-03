#randomForest is used to build the ensemble forest and fscaret is used for other data processing methods like splitting the training data set
library("randomForest")
library("fscaret")
library(pROC)

#Setting the working directory
setwd("/home/karthik/Documents/CSE675/Ford/Data")

print("*****Starting variable preparation phase*****")
TrainSet <- read.csv("fordTrain.csv")
TestSet <- read.csv("fordTest.csv")
SolutionSet <- read.csv("Solution.csv")
partition <- createDataPartition(TrainSet$IsAlert, p=0.30, list=FALSE)
TrainSet <- data.frame(TrainSet[partition,])

print("*****Starting Model Generation Phase*****")
myRF <- randomForest(IsAlert ~ V11 + E9, data = TrainSet, do.trace = TRUE, importance=TRUE, method="class", ntree=50, forest=TRUE)
# P1 + P2 + P5 + E3 + E4 + E5 + E6 + E7 + E8 + E9 + E10 + V1 + V2 + V3 + V4 + V5 + V6 + V8 + V11
# myRF <- randomForest(IsAlert ~ ., data = TrainSet, do.trace = TRUE, importance=TRUE, method="class", ntree=100, forest=TRUE)
plot(myRF, uniform=TRUE, main = "MY RANDOM FOREST")

print("*****Predicting Values*****")
myPrediction <- predict(myRF, newdata = TestSet, type= 'class')
predictionMetric <- data.frame(myPrediction)

ResultRandomForest <- data.frame(actual=SolutionSet$Prediction, calculated=round(myPrediction))

print("***Confusion matrix is as follows***")
ConfusionTableRF = table(ResultRandomForest$actual, ResultRandomForest$calculated)
ConfusionFrameRF = data.frame(ConfusionTableRF)
colnames(ConfusionFrameRF) <- c("Actual","Predicted","Count")
hitsRF = ConfusionFrameRF[1,3] + ConfusionFrameRF[4,3]
total = sum(ConfusionFrameRF[,3])
accuracyRF = hitsRF/total
print("ACCURACY OF RANDOM FOREST MODEL IS AS FOLLLOWS")
print(signif(accuracyRF,digits=2))

print("***Plotting results***")
randomForestPlot <- roc(ResultRandomForest$calculated, ResultRandomForest$actual, ci=TRUE, of="thresholds", thresholds=0.9)
with(randomForestPlot, plot(main ='ROC Plot for Random Forest Model', specificities, sensitivities, type='l', xlim=c(1,0), ylim=c(0,1),xaxs='i',yaxs='i'))