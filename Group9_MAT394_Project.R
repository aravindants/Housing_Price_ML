library(dplyr)
library(ggplot2)
library(caTools)
library(corrgram)
library(randomForest)
library(relaimpo)
library(ggcorrplot)
library(e1071)

set.seed(100)

#loading data
data <- read.csv("C:\\Users\\HP\\Downloads\\data.csv")
head(data)

#checking for any missing values
any(is.na(data))
n_data <- filter(data, price != 0)
n_data <- filter(n_data, bathrooms != 0)

#removing some columns
df = subset(n_data, select = -c(date, street, city, country, statezip,yr_renovated, waterfront))
head(df)

#feature scaling
df$sqft_living <- scale(df$sqft_living)
df$sqft_lot <- scale(df$sqft_lot)
df$sqft_above <- scale(df$sqft_above)
df$floors <- scale(df$floors)
df$view <- scale(df$view)
df$bedrooms <- scale(df$bedrooms)
df$bathrooms <- scale(df$bathrooms)
df$condition <- scale(df$condition)
df$yr_built <- scale(df$yr_built)
df$sqft_basement <- scale(df$sqft_basement)

#forward and backward step to determine base features
base.mod <- lm(price ~ 1 , data=df)
all.mod <- lm(price ~ . , data= df)
stepMod <- step(base.mod, scope = list(lower = base.mod, upper = all.mod), direction = "both", trace = 0, steps = 1000)
shortlistedVars <- names(unlist(stepMod[[1]]))
shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"]
print(shortlistedVars)

#removing the rest of features
df = subset(df, select = -c(sqft_basement, sqft_lot))
head(df)

#correlation plot
corr <- round(cor(df), 1)
ggcorrplot(corr)

#splitting data into test and training
sampleSplit <- sample.split(Y=df$price, SplitRatio=0.80)
trainset <- subset(x=df, sampleSplit==TRUE)
testSet <- subset(x=df, sampleSplit==FALSE)

#training linear reg model
model <- lm(formula = price ~ ., data = trainset)
summary(model)

#prediction
prediction <- predict(model, testSet)

#making the actual vs predicted dataframe
modelEval <- cbind(testSet$price, prediction)
colnames(modelEval) <- c('Actual', 'Predicted')
modelEval <- as.data.frame(modelEval)

#plotting the model
plot(modelEval)

#getting rmse value
mse_reg <- mean((modelEval$Actual - modelEval$Predicted)^2)
rmse_reg <- sqrt(mse_reg)

#printing rmse
print(rmse_reg)

#training random forest
rf.forest <- randomForest(price ~ .,mtry = 1,data = trainset,importance=TRUE)

#ranking each feature
importance_rf <- importance(rf.forest)
varImportance <- data.frame(Variables = row.names(importance_rf),
                            Importance = round(importance_rf[ ,'%IncMSE'],2))
rankImportance <- varImportance %>% mutate(Rank = paste0('#',dense_rank(desc(Importance))))
print("Plot of variable importance")
print("Variable importance of initial model")
ggplot(rankImportance, aes(x = reorder(Variables, Importance),
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') +
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()

#prediciton and storing result in dataframe
result <- data.frame(testSet$price, predict(rf.forest,testSet,type = "response"))
colnames(result) <- c('Actual', 'Predicted')
result <- as.data.frame(result)

#plotting result
plot(result)

#getting rmse value
mse_rf <- mean((result$Actual - result$Predicted)^2)
rmse_rf <- sqrt(mse_rf)

#printing rmse value
print(rmse_rf)

#creating model using support vector machines
model_svm <- svm(price ~ ., trainset)

#creating predictions
prediction_svm <- predict(model_svm, testSet)

#creating dataframe with actual and predicted values
nmodelEval <- cbind(testSet$price, prediction_svm)
colnames(nmodelEval) <- c('Actual', 'Predicted')
nmodelEval <- as.data.frame(nmodelEval)

#getting rmse value
nmse_reg <- mean((nmodelEval$Actual - nmodelEval$Predicted)^2)
nrmse_reg <- sqrt(nmse_reg)

#plotting for svm
plot(nmodelEval)

#printing rmse
print(nrmse_reg)

#all rmse values
print(rmse_reg)
print(rmse_rf)
print(nrmse_reg)

