# Databricks notebook source
# MAGIC %md
# MAGIC ## LASSO Regression model 
# MAGIC Lasso stands for Least Absolute Shrinkage and Selection Operator. It makes use of L1 regularization technique in the objective function. 
# MAGIC 
# MAGIC Lasso regression is a parsimonious model that performs L1 regularization. The L1 regularization adds a penalty equivalent to the absolute magnitude of regression coefficients and tries to minimize them
# MAGIC 
# MAGIC Lasso regression can perform in-built variable selection as well as parameter shrinkage. While using ridge regression one may end up getting all the variables but with Shrinked Paramaters.

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %sh 
# MAGIC install.packages("lme4")

# COMMAND ----------

library(lme4)
library(tidyverse)
library(pscl)
library(parameters)
library(gt)
library(gsubfn)
library(proto)
library(sqldf)
library(RSQLite)
library(glmnet)
library(usethis)
library(devtools)
library(visdat)
library(skimr)
library(caret)
library(DataExplorer)

# COMMAND ----------

# MAGIC %sh ls /dbfs/FileStore/tables/

# COMMAND ----------

data<-read.csv("/dbfs/FileStore/tables/all_vars_for_zeroinf_analysis.csv")

# COMMAND ----------

head(data, 10)

# COMMAND ----------

dim(data)

# COMMAND ----------

glimpse(data)

# COMMAND ----------

summary(data)

# COMMAND ----------

str(data)

# COMMAND ----------

colnames(data)

# COMMAND ----------

colnames(data) <- c( "county", "confirmed_cases" , "confirmed_deaths" ,
 "state"  ,"length_of_lockdown" ,"cases" ,"deaths","POP_ESTIMATE_2018" , "total_state_pop",	'Active_Physicians_per_100000_Population_2018_AAMC',	'Total_Active_Patient_Care_Physicians_per_100000_Population_2018_AAMC',	'Active_Primary_Care_Physicians_per_100000_Population_2018_AAMC',	'Active_Patient_Care_Primary_Care_Physicians_per_100000_Population_2018_AAMC',	'Active_General_Surgeons_per_100000_Population_2018_AAMC',	'Active_Patient_Care_General_Surgeons_per_100000_Population_2018_AAMC',	'Percentage_of_Active_Physicians_Who_Are_Female_2018_AAMC',	'Percentage_of_Active_Physicians_Who_Are_Intertiol_Medical_Graduates_IMGs-2018_AAMC',	'Percentage_of_Active_Physicians_Who_Are_Age_60_or_Older_2018_AAMC',	'MD_and_DO_Student_Enrollment_per_100000_Population_AY_2018_2019_AAMC',	'Student_Enrollment_at_Public_MD_and_DO_Schools_per_100000_Population_AY_2018_2019_AAMC',	'Percentage_Change_in_Student_Enrollment_at_MD_and_DO_Schools_2008_2018_AAMC',	'Percentage_of_MD_Students_Matriculating_In_State_AY_2018_2019_AAMC',	'Total_Residents_Fellows_in_ACGME_Programs_per_100000_Population_as_of_December_31_2018_AAMC',	'Total_Residents_Fellows_in_Primary_Care_ACGME_Programs_per_100000_Population_as_of_Dec_31_2018_AAMC',	'Percentage_of_Residents_in_ACGME_Programs_Who_Are_IMGs_as_of_December_31_2018_AAMC',	'Ratio_of_Residents_and_Fellows_GME_to_Medical_Students_UME-AY_2017_2018_AAMC',	'Percent_Change_in_Residents_and_Fellows_in_ACGME_Accredited_Programs_2008_2018_AAMC',	'Percentage_of_Physicians_Retained_in_State_from_Undergraduate_Medical_Education_UME-2018_AAMC',	'All_Specialties_AAMC',	'State_Local_Government_hospital_beds_per_1000_people_2019',	'Non_profit_hospital_beds_per_1000_people_2019',	'For_profit_hospital_beds_per_1000_people_2019',	'Total_hospital_beds_per_1000_people_2019',	'Total_nurse_practitioners_2019',	'Total_physician_assistants_2019',	'Total_Hospitals_2019',	'Total_Primary_Care_Physicians_2019',	'Surgery_specialists_2019',	'Emergency_Medicine_specialists_2019',	'Total_Specialist_Physicians_2019',	'ICU_Beds',	'pop_fraction',	'Length_of_Life_rank',	'Quality_of_Life_rank',	'Health_Behaviors_rank',	'Clinical_Care_rank',	'Social-Economic_Factors_rank',	'Physical_Environment_rank',	'Adult_smoking_percentage',	'Adult_obesity_percentage',	'Excessive_drinking_percentage',	'Population_per_sq_mile',	'House_per_sq_mile',	'Share_of_Tests_with_Positive_COVID_19_Results', 'Number_of_Tests_with_Results_per_1000_Population'
)

# COMMAND ----------

view(data)

# COMMAND ----------

select(data, county, confirmed_cases)

# COMMAND ----------

data %>%
    group_by(county) %>%
    summarise(count = n())

# COMMAND ----------

vis_miss(data)
vis_dat(data)

# COMMAND ----------

skim(data)

# COMMAND ----------

#DataExplorer::create_report(data)

# COMMAND ----------

#colnames(data)

# COMMAND ----------

data %>% 
select(county,length_of_lockdown, confirmed_cases,confirmed_deaths,POP_ESTIMATE_2018, POP_ESTIMATE_2018,ICU_Beds, Adult_obesity_percentage, Quality_of_Life_rank, Excessive_drinking_percentage, Population_per_sq_mile,
     Clinical_Care_rank, Adult_smoking_percentage,Total_Specialist_Physicians_2019,Physical_Environment_rank,  Number_of_Tests_with_Results_per_1000_Population) -> final_data

# COMMAND ----------

display(final_data)

# COMMAND ----------

#Creating dependent and independent variables. 
final_data %>% 
select(confirmed_deaths, ICU_Beds, Adult_obesity_percentage, Quality_of_Life_rank, Excessive_drinking_percentage, Population_per_sq_mile, Clinical_Care_rank, Adult_smoking_percentage, Total_Specialist_Physicians_2019, Physical_Environment_rank, Number_of_Tests_with_Results_per_1000_Population) -> X_variables


final_data %>% 
select(confirmed_deaths) -> y_target

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Linear Regression

# COMMAND ----------

set.seed(100) 

index = sample(1:nrow(final_data), 0.7*nrow(final_data)) 

train = final_data[index,] # Create the training data 
test = final_data[-index,] # Create the test data

dim(train)



# COMMAND ----------

train

# COMMAND ----------

dim(test)

# COMMAND ----------

lr = lm(confirmed_deaths ~  ICU_Beds + Adult_obesity_percentage + Quality_of_Life_rank + Excessive_drinking_percentage + Population_per_sq_mile + Clinical_Care_rank + Adult_smoking_percentage+ Total_Specialist_Physicians_2019 + Physical_Environment_rank + Number_of_Tests_with_Results_per_1000_Population, data = train)
summary(lr)

# COMMAND ----------

#Step 1 - create the evaluation metrics function

eval_metrics = function(model, df, predictions, target){
    resids = df[,target] - predictions
    resids2 = resids**2
    N = length(predictions)
    r2 = as.character(round(summary(model)$r.squared, 2))
    adj_r2 = as.character(round(summary(model)$adj.r.squared, 2))
    print(adj_r2) #Adjusted R-squared
    print(as.character(round(sqrt(sum(resids2)/N), 2))) #RMSE
}

# Step 2 - predicting and evaluating the model on train data
predictions = predict(lr, newdata = train)
eval_metrics(lr, train, predictions, target = 'confirmed_deaths')

# Step 3 - predicting and evaluating the model on test data
predictions = predict(lr, newdata = test)
eval_metrics(lr, test, predictions, target = 'confirmed_deaths')

# COMMAND ----------

# MAGIC %md 
# MAGIC The above output shows that RMSE, one of the two evaluation metrics, is 80.82 for train data and 84.69  for test data. On the other hand, R-squared value is around 25 percent for both train and test data, which indicates bad performance.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Regularization

# COMMAND ----------


cols_reg = c('county', 'confirmed_deaths', 'ICU_Beds', 'Adult_obesity_percentage', 'Quality_of_Life_rank', 'Excessive_drinking_percentage', 'Population_per_sq_mile', 'Clinical_Care_rank', 'Adult_smoking_percentage', 'Total_Specialist_Physicians_2019', 'Physical_Environment_rank', 'Number_of_Tests_with_Results_per_1000_Population')

dummies <- dummyVars(confirmed_deaths ~ ., data = final_data[,cols_reg])

train_dummies = predict(dummies, newdata = train[,cols_reg])

test_dummies = predict(dummies, newdata = test[,cols_reg])

print(dim(train_dummies)); print(dim(test_dummies))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Ridge Regression
# MAGIC 
# MAGIC Ridge regression is an extension of linear regression where the loss function is modified to minimize the complexity of the model. 
# MAGIC 
# MAGIC This modification is done by adding a penalty parameter that is equivalent to the square of the magnitude of the coefficients.

# COMMAND ----------

x = as.matrix(train_dummies)
y_train = train$confirmed_deaths

x_test = as.matrix(test_dummies)
y_test = test$confirmed_deaths

lambdas <- 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = lambdas)

summary(ridge_reg)

# COMMAND ----------



# COMMAND ----------

cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda

# COMMAND ----------

# MAGIC %md 
# MAGIC The optimal lambda value comes out to be 7.943282 and will be used to build the ridge regression model

# COMMAND ----------

# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))

  
  # Model performance metrics
data.frame(
  RMSE = RMSE,
  Rsquare = R_square
)
  
}

# Prediction and evaluation on train data
predictions_train <- predict(ridge_reg, s = optimal_lambda, newx = x)
eval_results(y_train, predictions_train, train)


# COMMAND ----------


# Prediction and evaluation on test data
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test, test)

# COMMAND ----------

# MAGIC %md 
# MAGIC 80.82 for train data and 84.69  for test data.
# MAGIC 
# MAGIC The above output shows that the RMSE and R-squared values for the ridge regression model on the training data are 80.89787 and 25 percent, respectively. For the test data, the results for these metrics are 84 and 7 percent, respectively. There is an improvement in the performance compared with linear regression model.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Lasso Regression model 
# MAGIC 
# MAGIC Lasso regression, or the Least Absolute Shrinkage and Selection Operator, is also a modification of linear regression. 
# MAGIC 
# MAGIC In lasso, the loss function is modified to minimize the complexity of the model by limiting the sum of the absolute values of the model coefficients (also called the l1-norm).
# MAGIC 
# MAGIC The loss function for lasso regression can be expressed as below:
# MAGIC 
# MAGIC Loss function = OLS + alpha * summation (absolute values of the magnitude of the coefficients)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC The first step to build a lasso model is to find the optimal lambda value using the code below.

# COMMAND ----------

lambdas <- 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements lasso regression
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

# Best 
lambda_best <- lasso_reg$lambda.min 
lambda_best

# COMMAND ----------

# MAGIC %md 
# MAGIC Once we have the optimal lambda value, we train the lasso model in the first line of code below.

# COMMAND ----------

lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)

predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
eval_results(y_train, predictions_train, train)



# COMMAND ----------

predictions_test <- predict(lasso_model, s = lambda_best, newx = x_test)
eval_results(y_test, predictions_test, test)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Elastic Net Regression model 
# MAGIC 
# MAGIC Elastic net regression combines the properties of ridge and lasso regression. 
# MAGIC 
# MAGIC It works by penalizing the model using both the 1l2-norm1 and the 1l1-norm1. 
# MAGIC 
# MAGIC The model can be easily built using the caret package, which automatically selects the optimal value of parameters alpha and lambda.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC - The first line of code creates the training control object train_cont which specifies how the repeated cross validation will take place. 
# MAGIC 
# MAGIC - The second line builds the elastic regression model in which a range of possible alpha and lambda values are tested and their optimum value is selected. The argument tuneLength specifies that 10 different combinations of values for alpha and lambda are to be tested.

# COMMAND ----------

# Set training control
train_cont <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 5,
                              search = "random",
                              verboseIter = TRUE)

# Train the model
elastic_reg <- train(confirmed_deaths ~ .,
                           data = train,
                           method = "glmnet",
                           preProcess = c("center", "scale"),
                           tuneLength = 10,
                           trControl = train_cont)


# Best tuning parameter
elastic_reg$bestTune

# COMMAND ----------

# MAGIC %md 
# MAGIC Once we have trained the model, we use it to generate the predictions and print the evaluation results for both the training and test datasets

# COMMAND ----------

# Make predictions on training set
predictions_train <- predict(elastic_reg, x)

eval_results(y_train, predictions_train, train) 



# COMMAND ----------



# COMMAND ----------

# Make predictions on test set
predictions_test <- predict(elastic_reg, x_test)
eval_results(y_test, predictions_test, test)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC The above output shows that the RMSE and R-squared values for the elastic net regression model on the training data are and  percent, respectively. 
# MAGIC 
# MAGIC The results for these metrics on the test data are  and 86 , respectively.
# MAGIC 
# MAGIC Conclusion
# MAGIC 
# MAGIC The performance of the models is summarized below:
# MAGIC 
# MAGIC - Linear Regression Model: Test set RMSE of  and R-square of  percent.
# MAGIC 
# MAGIC - Ridge Regression Model: Test set RMSE of and R-square of  percent.
# MAGIC 
# MAGIC - Lasso Regression Model: Test set RMSE of  and R-square of percent.
# MAGIC 
# MAGIC - ElasticNet Regression Model: Test set RMSE of  and R-square of  percent.

# COMMAND ----------

Y1 <- as.matrix(y_target)
is.matrix(Y1)
X1 <- as.matrix(X_variables)
is.matrix(X_variables)
CV = cv.glmnet(x=X1, y=Y1, family= "gaussian",standardize=FALSE, type.measure = "mae", alpha = 0)

# COMMAND ----------

plot(CV)

# COMMAND ----------

fit = (glmnet(x=X1, y=Y1, family= "gaussian", alpha=0,standardize=FALSE, lambda=CV$lambda.1se))


# COMMAND ----------

fit$beta[,1]
ridge <- as.matrix(fit$beta)

# COMMAND ----------


write.table(ridge,"/databricks/driver/ridge.txt", sep="\t")

# COMMAND ----------

plot(fit, label = TRUE)
plot(fit)
print(fit)

write.table

# COMMAND ----------


