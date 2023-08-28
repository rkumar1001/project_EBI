library(tidyverse)
library(scales)
library(lubridate)
library(plotly)
library(tidytext)
library(modelr)
library(caret)
library(ROSE)
library(glmnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library('ggcorrplot')
library(DataExplorer)


#load the csv file from local storage
df <- read.csv('C:\\Users\\rajat\\Desktop\\New folder (3)\\filtered.csv')

cat("Dataset has over 149 thousand rows. Dimension of the data frame : ", dim(df))
head(df, 4) #print first 4 rows 

# drop variables which will be of no use for our analysis
df <- df[,!names(df) %in% c("State","Pressure.in.","Amenity","Civil_Twilight","Nautical_Twilight","Astronomical_Twilight","County","Start_Time","End_Time","ID", "Start_Lat", "End_Lat","Start_Lng","End_Lng","Description","Number","Street","Side","Country","Timezone","Airport_Code","Weather_Timestamp","Wind_Direction","Zipcode")]
df %>% head(5)

df <- na.omit(df) #remove any row containing NA values

#check for any missing value
if (any(is.na(df))) {
  cat("There are missing values in the data.")
} else {
  cat("There are no missing values in the data.")
}


library(psych)

#convert Day to 1 otherwise 0
df[['Sunrise_Sunset']] <- ifelse(df[['Sunrise_Sunset']] == "Day", 1, 0)

# convert all the columns containg True to 1 and False to 0
df[['Traffic_Signal']] <- ifelse(df[['Traffic_Signal']] == "False", 0, 1)
df[['Junction']] <- ifelse(df[['Junction']] == "False", 0, 1)
df[['Bump']] <- ifelse(df[['Bump']] == "False", 0, 1)
df[['Crossing']] <- ifelse(df[['Crossing']] == "False", 0, 1)
df[['Give_Way']] <- ifelse(df[['Give_Way']] == "False", 0, 1)
df[['No_Exit']] <- ifelse(df[['No_Exit']] == "False", 0, 1)
df[['Railway']] <- ifelse(df[['Railway']] == "False", 0, 1)
df[['Roundabout']] <- ifelse(df[['Roundabout']] == "False", 0, 1)
df[['Station']] <- ifelse(df[['Station']] == "False", 0, 1)
df[['Stop']] <- ifelse(df[['Stop']] == "False", 0, 1)
df[['Traffic_Calming']] <- ifelse(df[['Traffic_Calming']] == "False", 0, 1)
df[['Turning_Loop']] <- ifelse(df[['Turning_Loop']] == "False", 0, 1)
df %>% head(5)


library(PerformanceAnalytics)
library(dplyr)

# select a random sample of 1000 rows
sampled_data <- df %>%
  sample_n(1000)

# select only the numerical variables
num_vars <- sampled_data %>%
  select_if(is.numeric)

corr.test(num_vars,use = "pairwise",method="spearman",adjust="none",alpha=.05)

dev.new()

# create correlation matrix plot for numerical variables
chart.Correlation(num_vars,method = "spearman",scatterplot = TRUE,pch = 16,
            col.regions = colorRampPalette(c("red", "white", "blue"))(100))


# Select only the numerical variables
num_vars <- df %>%
  select_if(is.numeric)

# Compute correlation matrix for numerical variables
corr_matrix <- cor(num_vars)

# Create a color palette for the correlation matrix plot
color_palette <- colorRampPalette(c("#6E5F00", "#E6B800", "#FFFFFF", "#4C4C4C", "#000000"))(100)

dev.new()

# Create correlation matrix plot
corrplot(corr_matrix,
         method = "number",
         type = "upper",
         col = color_palette,
         tl.col = "black",
         tl.srt = 45,
         tl.cex = 0.8)


# Load the required packages
library(ROCR)
library(pROC)

# Recode Severity variable as binary (1 for low severity and 2 for high severity)
df$Severity <- ifelse(df$Severity >= 1 & df$Severity <= 2, 1,
                      ifelse(df$Severity >= 3 & df$Severity <= 4, 2,
                             df$Severity))

# Multiple Logistics Regression

# Split the data into a training set and a testing set
set.seed(123) 
train_index <- sample(nrow(df), round(0.7 * nrow(df)), replace = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Perform multiple logarithmic regression on the training set
model <- lm(Severity ~ Visibility.mi. + Wind_Speed.mph. + 
               Distance.mi. + Temperature.F. + Wind_Chill.F. + Humidity... + 
               Precipitation.in. + Bump + Crossing + Give_Way + Junction + 
               No_Exit + Railway + Roundabout + Station + Stop + Traffic_Calming + 
               Traffic_Signal + Turning_Loop + Sunrise_Sunset, data = train_data)

# View the model summary
summary(model)

# Make predictions using the testing set
predictions <- predict(model, newdata = test_data, type = "response")

# Calculate mean absolute error
mae <- mean(abs(test_data$Severity - predictions))
print(paste0("MAE: ", mae))

# Calculate mean square error
mse <- mean((test_data$Severity - predictions)^2)
print(paste0("MSE: ", mse))

# Create a prediction object and calculate AUC
auc_obj <- roc(test_data$Severity, predictions)
auc_val <- auc(auc_obj)
# Print the AUC value
print(paste0("AUC: ", auc_val))

# decision tree
# Load the rpart package
library(rpart)
library(rpart.plot)

str(df)

# Split the data into a training set and a testing set
set.seed(123) # Set seed for reproducibility
train_index <- sample(nrow(df), round(0.7 * nrow(df)), replace = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Build the decision tree using the training set
model <- rpart(Severity ~ Visibility.mi. + Wind_Speed.mph. + 
                 Distance.mi. + Temperature.F. + Wind_Chill.F. + Humidity... + 
                 Precipitation.in. + Bump + Crossing + Give_Way + Junction + 
                 No_Exit + Railway + Roundabout + Station + Stop + Traffic_Calming + 
                 Traffic_Signal + Turning_Loop + Sunrise_Sunset, data = train_data)

# View the model summary
summary(model)

dev.new()

rpart.plot(model, main = "Decision Tree for Severity Prediction")

# Make predictions on the testing set
predicted_severity <- predict(model, newdata = test_data)

# Compare the predicted severity values to the actual severity values in the testing set
accuracy <- mean((test_data$Severity - predicted_severity )^2)
print(paste0("Mean Squared Error: ", accuracy))

mae <- mean(abs(test_data$Severity - predicted_severity))
print(paste0("MAE: ", mae))

# Create a prediction object and calculate AUC
auc_obj <- roc(test_data$Severity, predicted_severity)
print(predicted_severity)
auc_val <- auc(auc_obj)

# Print the AUC value
print(paste0("AUC: ", auc_val))




#Random forest
# Load the necessary packages
library(randomForest)
library(caret)

# Train the random forest model
set.seed(123)
elapsed_time <- system.time({

rf_model <- randomForest(Severity ~ Visibility.mi. + Wind_Speed.mph. + 
                           Distance.mi. + Temperature.F. + Wind_Chill.F. + Humidity... + 
                           Precipitation.in. + Bump + Crossing + Give_Way + Junction + 
                           No_Exit + Railway + Roundabout + Station + Stop + Traffic_Calming + 
                           Traffic_Signal + Turning_Loop + Sunrise_Sunset, data = train_data, 
                         importance = TRUE, ntree = 100, method = "rf")
})[3]

print(elapsed_time)

# Make predictions on the testing data
predictions <- predict(rf_model, newdata = test_data)

# Evaluate the performance of the model
rmse <- sqrt(mean((predictions - test_data$Severity)^2))
cat("Root mean squared error:", rmse, "\n")

mae <- mean(abs(predictions - test_data$Severity))
print(paste0("MAE: ", mae))

# Calculate ROC curve and AUC
roc_data <- roc(test_data$Severity, predictions)
auc_value <- auc(roc_data)
print(auc_value)

# Plot ROC curve
plot(roc_data, main = paste("ROC Curve (AUC = ", round(auc_value, 2), ")", sep=""))





