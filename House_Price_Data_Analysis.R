# Reading in the test and train data
test = read.csv("cwk_ames_test.csv", header=T, sep=",")
train = read.csv("cwk_ames_train.csv", header=T, sep=",")

# Adding log10 of SalePrice to Train set and calculating its mean
train$LogSalePrice <- log10(train$SalePrice)
mean_LogSale <- mean(train$LogSalePrice)
print(paste("The mean of the Log SalePrice is: ", mean_LogSale))

# Fitting a Multiple Linear Regression to LogSalePrice and showing summary
ml_model <- lm(LogSalePrice ~. -SalePrice, data=train)
print("Summary of fit:")
summary(ml_model)

# Generating all possible combination of regressors and naming them
poss_combs <- expand.grid(c(TRUE, FALSE), c(TRUE, FALSE), c(TRUE, FALSE), 
                           c(TRUE, FALSE), c(TRUE, FALSE), c(TRUE, FALSE), 
                           c(TRUE, FALSE), c(TRUE, FALSE), c(TRUE, FALSE))
names(poss_combs) <- colnames(train[1:9])

regressors <- colnames(train[1:9])

# Creating formula objects for response and all combinations of explanatory variables
poss_models <- apply(poss_combs, 1, function(x) as.formula(paste(c("LogSalePrice ~ 1", regressors[x]), collapse="+")))

# Fitting models for each possible combination
all_results <- lapply(poss_models, function(x) lm(x, data=train))

# Calculating AIC and BIC for each submodel
AIC_results <- sapply(X=all_results, FUN=AIC)
BIC_results <- sapply(X=all_results, FUN=BIC)

# Getting Minimum values of AIC and showing variables
AIC_min_index <- which.min(AIC_results)
AIC_min <- AIC_results[AIC_min_index]
AIC_min_regV <- regressors[as.logical(poss_combs[AIC_min_index,])]
print(paste("The min value of AIC obtained: ", AIC_min))
print("Regressors for AIC are: ")
print(AIC_min_regV)

# Getting Minimum values of BIC and showing variables
BIC_min_index <- which.min(BIC_results)
BIC_min <- BIC_results[BIC_min_index]
BIC_min_regV <- regressors[as.logical(poss_combs[BIC_min_index,])]
print(paste("The min value of BIC obtained: ", BIC_min))
print("Regressors for BIC are: ")
print(BIC_min_regV)

# List of number of regressors for each combination (same for AIC and BIC)
num_regressors <- apply(poss_combs, 1, function(x) sum(as.logical(x), na.rm = TRUE))

# Plotting AIC vs Num. of Regressors with  AIC_min value in Red and block filled
plot(num_regressors, AIC_results, col=ifelse(AIC_results==AIC_min, "red", "black"), 
     main="AIC vs Num. of Regressors", xlab="Num. of Regressors", ylab="AIC", 
     pch=ifelse(AIC_results==AIC_min, 19, 1), cex=ifelse(AIC_results==AIC_min, 1.33, 1))

# Plotting BIC vs Num. of Regressors with  AIC_min value in Red and block filled
plot(num_regressors, BIC_results, col=ifelse(BIC_results==BIC_min, "red", "black"), 
     main="BIC vs Num. of Regressors", xlab="Num. of Regressors", ylab="BIC", 
     pch=ifelse(BIC_results==BIC_min, 19, 1), cex=ifelse(BIC_results==BIC_min, 1.33, 1))

## Lasso Regression
library("glmnet")

# The lasso regression is trained with cross-validation
lasso_model <- cv.glmnet(as.matrix(train[,1:9]), train$LogSalePrice, alpha=1)

# The MSE is plotted against log Lambda
plot(lasso_model)

# Plotmo is used to to display coeff paths with labels for more detail
library("plotmo")
plot_glmnet(lasso_model$glmnet.fit, label=9)

# The minimum and 1 standard deviation lambda values are stored
lambda_min <- lasso_model$lambda.min
lambda_1se <- lasso_model$lambda.1se
print(paste("Lambda within 1 standard deviation is: ", lambda_1se))

# Coeffs for the 1se lambda are calculated
lasso_1se_fit <- glmnet(as.matrix(train[,1:9]), train$LogSalePrice, alpha=1)
lasso_1se_coef <- coef(lasso_1se_fit, s=lambda_1se)

# The salient regressors in the model are calculated
lasso_1se_regV <- colnames(train)[which(lasso_1se_coef[-1] != 0)]
print("Regressors for Lasso are: ")
print(lasso_1se_regV)

# Comparing the Regressors of each AIC, BIC and Lasso in that order
print("Regressors for the 3 models (AIC, BIC, Lasso) are: ")
print(AIC_min_regV)
print(BIC_min_regV)
print(lasso_1se_regV)

# Creating all possible combinations of best number of regressors from lasso
poss_regressors_n <- combn(colnames(train[1:9]), sum(lengths(lasso_1se_regV)))

# Fitting a multiple linear model to each of these combinations
poss_models_n <- apply(poss_regressors_n, 2, function(x) as.formula(paste(c("LogSalePrice ~ 1", x), collapse="+")))
all_results_n <- lapply(poss_models_n, function(x) lm(x, data=train))

# Calculating Min RSS and printing the corresponding regressors
RSS <- lapply(all_results_n, function(x) deviance(x))
min_RSS_index <- which.min(RSS)
RSS_min_regV <- poss_regressors_n[,min_RSS_index]

print(paste("Minimum RSS is for regressors: "))
print(RSS_min_regV)

# Predicting the response from the test data set for LM
RSS_min_lm = lm(as.formula(paste(c("LogSalePrice ~ 1", RSS_min_regV), collapse="+")), data=train)
pred_lm = predict(RSS_min_lm, test)

# Predicting the response from the test data set for Lasso M
pred_lasso = predict(lasso_model, newx = as.matrix(test), s = "lambda.1se")

# Plotting the predictions of the two models
print(paste("Mean LogSalePrice: ", mean_LogSale))
plot(pred_lm, pred_lasso, main="Predictions of the two models",
     xlab="Linear Model Prediction", ylab="Lasso Model Prediction", pch=3)
abline(h=mean_LogSale, col="red")
abline(0,1, col="red")
