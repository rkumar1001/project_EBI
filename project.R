# Load required packages
.libPaths(c("~/R/library", .libPaths()))

library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
library(pROC)


#load the csv file from local storage (use relative path for portability)
df <- read.csv('updated_final.csv')

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

# Recode Severity variable as binary (0 for low severity, 1 for high severity)
df$Severity <- ifelse(df$Severity >= 1 & df$Severity <= 2, 0,
                      ifelse(df$Severity >= 3 & df$Severity <= 4, 1,
                             df$Severity))

# ============================================
# Class Distribution Analysis
# ============================================
cat("\n========================================\n")
cat("       CLASS DISTRIBUTION ANALYSIS      \n")
cat("========================================\n\n")

cat("Class Distribution:\n")
print(table(df$Severity))
cat("\nClass Proportions:\n")
print(prop.table(table(df$Severity)))

# Visualize class imbalance
dev.new()
ggplot(df, aes(x = factor(Severity, labels = c("Low Severity", "High Severity")))) +
  geom_bar(fill = c("steelblue", "coral")) +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  labs(title = "Class Distribution: Accident Severity",
       x = "Severity Level", y = "Count") +
  theme_minimal()

# ============================================
# Feature Engineering: Weather Risk Categories
# ============================================
cat("\n--- Creating Weather Risk Categories ---\n")

df$Weather_Risk <- case_when(
  df$Visibility.mi. < 2 ~ "Poor_Visibility",
  df$Precipitation.in. > 0.1 ~ "Precipitation",
  df$Wind_Speed.mph. > 20 ~ "High_Wind",
  TRUE ~ "Normal"
)

cat("Weather Risk Distribution:\n")
print(table(df$Weather_Risk))

# Visualize severity by weather risk
dev.new()
ggplot(df, aes(x = Weather_Risk, fill = as.factor(Severity))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("steelblue", "coral"), 
                    labels = c("Low Severity", "High Severity")) +
  labs(title = "Accident Severity by Weather Condition",
       x = "Weather Risk Category", y = "Proportion", fill = "Severity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ============================================
# Feature Engineering: Road Risk Score
# ============================================
cat("\n--- Creating Road Risk Score ---\n")

# Note: These features are already binary (0/1)
# We sum them to create a composite risk score
df$Road_Risk_Score <- df$Bump + df$Crossing + df$Junction + 
                       df$Railway + df$Roundabout + df$Stop + 
                       df$Traffic_Signal + df$Traffic_Calming

cat("Road Risk Score Distribution:\n")
print(table(df$Road_Risk_Score))

# ============================================
# Day vs Night Analysis
# ============================================
cat("\n--- Day vs Night Severity Analysis ---\n")

severity_by_time <- df %>%
  group_by(Sunrise_Sunset) %>%
  summarise(
    Total_Accidents = n(),
    High_Severity_Count = sum(Severity == 1),
    Severity_Rate = mean(Severity) * 100
  )

cat("Severity by Time of Day (0=Night, 1=Day):\n")
print(severity_by_time)

# Only create plot if we have both day and night data
if(length(unique(df$Sunrise_Sunset)) > 1) {
  dev.new()
  ggplot(df, aes(x = factor(Sunrise_Sunset, labels = c("Night", "Day")), 
                 fill = as.factor(Severity))) +
    geom_bar(position = "fill") +
    scale_fill_manual(values = c("steelblue", "coral"), 
                      labels = c("Low Severity", "High Severity")) +
    labs(title = "Accident Severity: Day vs Night",
         x = "Time of Day", y = "Proportion", fill = "Severity") +
    theme_minimal()
} else {
  cat("Note: All data is from a single time period (Day or Night only)\n")
}

# Split the data ONCE into training and testing sets (reused for all models)
set.seed(123) 
train_index <- sample(nrow(df), round(0.7 * nrow(df)), replace = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# ============================================
# Multiple Logistic Regression (using glm)
# ============================================

# Perform logistic regression on the training set
logistic_model <- glm(Severity ~ Visibility.mi. + Wind_Speed.mph. + 
               Distance.mi. + Temperature.F. + Wind_Chill.F. + Humidity... + 
               Precipitation.in. + Bump + Crossing + Give_Way + Junction + 
               No_Exit + Railway + Roundabout + Station + Stop + Traffic_Calming + 
               Traffic_Signal + Turning_Loop + Sunrise_Sunset, 
             data = train_data, family = binomial)

# View the model summary
summary(logistic_model)

# Make predictions using the testing set (get probabilities)
predictions_prob <- predict(logistic_model, newdata = test_data, type = "response")

# Convert probabilities to class predictions
predictions_class <- ifelse(predictions_prob > 0.5, 1, 0)

# Confusion Matrix for Logistic Regression
cat("\n--- Logistic Regression Results ---\n")
confusion_lr <- confusionMatrix(as.factor(predictions_class), as.factor(test_data$Severity))
print(confusion_lr)

# Calculate mean absolute error
mae <- mean(abs(test_data$Severity - predictions_prob))
print(paste0("MAE: ", mae))

# Calculate mean square error
mse <- mean((test_data$Severity - predictions_prob)^2)
print(paste0("MSE: ", mse))

# Create a prediction object and calculate AUC
auc_obj <- roc(test_data$Severity, predictions_prob)
auc_val <- auc(auc_obj)
# Print the AUC value
print(paste0("AUC: ", auc_val))

# ============================================
# Decision Tree
# ============================================

str(df)

# Build the decision tree using the training set (reusing the same train/test split)
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

# Convert to class predictions
predicted_class_dt <- ifelse(predicted_severity > 0.5, 1, 0)

# Confusion Matrix for Decision Tree
cat("\n--- Decision Tree Results ---\n")
confusion_dt <- confusionMatrix(as.factor(predicted_class_dt), as.factor(test_data$Severity))
print(confusion_dt)

# Compare the predicted severity values to the actual severity values in the testing set
mse_dt <- mean((test_data$Severity - predicted_severity)^2)
print(paste0("Mean Squared Error: ", mse_dt))

mae_dt <- mean(abs(test_data$Severity - predicted_severity))
print(paste0("MAE: ", mae_dt))

# Create a prediction object and calculate AUC
auc_obj_dt <- roc(test_data$Severity, predicted_severity)
auc_val_dt <- auc(auc_obj_dt)

# Print the AUC value
print(paste0("AUC: ", auc_val_dt))




# ============================================
# Random Forest
# ============================================

# Train the random forest model (reusing the same train/test split)
set.seed(123)
elapsed_time <- system.time({

# Using balanced sampling to handle imbalance
rf_model <- randomForest(as.factor(Severity) ~ Visibility.mi. + Wind_Speed.mph. + 
                           Distance.mi. + Temperature.F. + Wind_Chill.F. + Humidity... + 
                           Precipitation.in. + Bump + Crossing + Give_Way + Junction + 
                           No_Exit + Railway + Roundabout + Station + Stop + Traffic_Calming + 
                           Traffic_Signal + Turning_Loop + Sunrise_Sunset, data = train_data, 
                         importance = TRUE, ntree = 100, sampsize = c("0" = 5000, "1" = 5000))
})[3]

print(elapsed_time)

# Make predictions on the testing data
predictions_rf_class <- predict(rf_model, newdata = test_data, type = "class")
predictions_rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[, 2]

# Class predictions already obtained
predicted_class_rf <- as.numeric(as.character(predictions_rf_class))

# Confusion Matrix for Random Forest
cat("\n--- Random Forest Results ---\n")
confusion_rf <- confusionMatrix(as.factor(predicted_class_rf), as.factor(test_data$Severity))
print(confusion_rf)

# Evaluate the performance of the model
accuracy_rf <- sum(predicted_class_rf == test_data$Severity) / length(test_data$Severity)
cat("Accuracy:", round(accuracy_rf * 100, 2), "%\n")

mae_rf <- mean(abs(predicted_class_rf - test_data$Severity))
print(paste0("MAE: ", mae_rf))

# Calculate ROC curve and AUC using probabilities
roc_data_rf <- roc(test_data$Severity, predictions_rf_prob)
auc_value_rf <- auc(roc_data_rf)
print(paste0("AUC: ", auc_value_rf))

# Plot ROC curve
dev.new()
plot(roc_data_rf, main = paste("ROC Curve - Random Forest (AUC = ", round(auc_value_rf, 2), ")", sep=""))

# ============================================
# Feature Importance Visualization
# ============================================
importance_df <- data.frame(importance(rf_model))
importance_df$Feature <- rownames(importance_df)

# For classification, use MeanDecreaseGini
dev.new()
ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)", 
       x = "Feature", 
       y = "Mean Decrease in Gini") +
  theme_minimal()

# ============================================
# Cross-Validation for Random Forest
# ============================================
cat("\n--- 10-Fold Cross-Validation Results ---\n")
train_control <- trainControl(method = "cv", number = 10)
rf_cv <- train(as.factor(Severity) ~ Visibility.mi. + Wind_Speed.mph. + 
                 Distance.mi. + Temperature.F. + Wind_Chill.F. + Humidity... + 
                 Precipitation.in. + Bump + Crossing + Give_Way + Junction + 
                 No_Exit + Railway + Roundabout + Station + Stop + Traffic_Calming + 
                 Traffic_Signal + Turning_Loop + Sunrise_Sunset, 
               data = train_data, method = "rf", trControl = train_control, ntree = 50)
print(rf_cv)

# ============================================
# Model Comparison Dashboard
# ============================================
cat("\n========================================\n")
cat("       MODEL COMPARISON DASHBOARD       \n")
cat("========================================\n\n")

model_comparison <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(
    confusion_lr$overall["Accuracy"],
    confusion_dt$overall["Accuracy"],
    confusion_rf$overall["Accuracy"]
  ),
  AUC = c(auc_val, auc_val_dt, auc_value_rf),
  Sensitivity = c(
    confusion_lr$byClass["Sensitivity"],
    confusion_dt$byClass["Sensitivity"],
    confusion_rf$byClass["Sensitivity"]
  ),
  Specificity = c(
    confusion_lr$byClass["Specificity"],
    confusion_dt$byClass["Specificity"],
    confusion_rf$byClass["Specificity"]
  ),
  Kappa = c(
    confusion_lr$overall["Kappa"],
    confusion_dt$overall["Kappa"],
    confusion_rf$overall["Kappa"]
  )
)

cat("Model Performance Comparison:\n")
print(model_comparison)

# Visualize model comparison
model_comparison_long <- model_comparison %>%
  select(Model, Accuracy, AUC, Sensitivity, Specificity) %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

dev.new()
ggplot(model_comparison_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Model Performance Comparison",
       y = "Score", x = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylim(0, 1)

# ============================================
# Actionable Insights
# ============================================
cat("\n========================================\n")
cat("         ACTIONABLE INSIGHTS            \n")
cat("========================================\n\n")

# Top risk factors from Random Forest
importance_sorted <- importance_df[order(-importance_df$MeanDecreaseGini), ]
cat("Top 5 Risk Factors for Severe Accidents:\n")
print(head(importance_sorted[, c("Feature", "MeanDecreaseGini")], 5))

# Calculate risk thresholds based on severe accidents
cat("\n--- High-Risk Condition Thresholds ---\n")
cat(sprintf("Severe accidents avg visibility: %.2f miles (vs %.2f overall)\n", 
            mean(df$Visibility.mi.[df$Severity == 1]), 
            mean(df$Visibility.mi.)))
cat(sprintf("Severe accidents avg humidity: %.2f%% (vs %.2f%% overall)\n", 
            mean(df$Humidity...[df$Severity == 1]), 
            mean(df$Humidity...)))
cat(sprintf("Severe accidents avg distance: %.2f miles (vs %.2f overall)\n", 
            mean(df$Distance.mi.[df$Severity == 1]), 
            mean(df$Distance.mi.)))
cat(sprintf("Severe accidents avg temperature: %.2f°F (vs %.2f°F overall)\n", 
            mean(df$Temperature.F.[df$Severity == 1]), 
            mean(df$Temperature.F.)))

# Weather risk analysis
cat("\n--- Severity Rate by Weather Condition ---\n")
weather_severity <- df %>%
  group_by(Weather_Risk) %>%
  summarise(
    Total = n(),
    High_Severity = sum(Severity == 1),
    Severity_Rate = round(mean(Severity) * 100, 2)
  ) %>%
  arrange(desc(Severity_Rate))
print(weather_severity)

# Summary recommendations
cat("\n--- KEY RECOMMENDATIONS ---\n")
cat("1. Deploy additional resources during poor visibility conditions\n")
cat("2. Issue warnings when precipitation detected\n")
cat("3. Focus on intersections with multiple road features\n")
cat("4. Consider time-of-day patterns for patrol scheduling\n")

cat("\n========================================\n")
cat("       ANALYSIS COMPLETE                \n")
cat("========================================\n")





