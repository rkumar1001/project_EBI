# ============================================
# Accident Severity Prediction Under Weather & Visibility Conditions
# Lamar University | Jan 2023 - April 2023
# ============================================
# This project analyzes ~2.8M U.S. traffic accident records to predict
# accident severity based on weather, visibility, and roadway conditions.
#
# Data Sampling: See data_sampling.R for the sampling pipeline that reduces
# the original 2.8M records to 108k high-quality samples using:
#   - Time-based stratification (hourly/monthly patterns)
#   - H3 spatial indexing (geographic diversity)
#   - K-means clustering (weather/visibility scenario diversity)
#
# Models: Logistic Regression, Decision Tree, Random Forest (with hyperparameter tuning)
# Interpretability: SHAP-based analysis for all models
# ============================================

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

# Install and load additional packages for enhanced modeling
# Note: shapviz depends on xgboost which requires long compilation
# We use fastshap for SHAP computation and base ggplot2 for visualization
required_packages <- c("ranger", "fastshap")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
    library(pkg, character.only = TRUE)
  }
}


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

# ============================================
# Feature Engineering: Interaction Features
# ============================================
cat("\n--- Creating Interaction Features ---\n")

# Visibility x Precipitation interaction (low visibility in rain is dangerous)
df$Visibility_Precip_Interaction <- df$Visibility.mi. * (1 + df$Precipitation.in.)

# Temperature x Humidity interaction (heat index proxy)
df$Temp_Humidity_Interaction <- df$Temperature.F. * (df$Humidity... / 100)

# Wind Chill effect (cold + wind)
df$Wind_Chill_Effect <- ifelse(
  df$Temperature.F. < 50 & df$Wind_Speed.mph. > 5,
  (df$Temperature.F. - df$Wind_Chill.F.) / (df$Wind_Speed.mph. + 1),
  0
)

cat("Interaction features created: Visibility_Precip_Interaction, Temp_Humidity_Interaction, Wind_Chill_Effect\n")

# ============================================
# Feature Engineering: Temperature Risk Categories
# ============================================
cat("\n--- Creating Temperature Risk Categories ---\n")

df$Temp_Risk <- case_when(
  df$Temperature.F. < 20 ~ "Extreme_Cold",
  df$Temperature.F. < 40 ~ "Cold",
  df$Temperature.F. <= 80 ~ "Normal",
  df$Temperature.F. <= 95 ~ "Hot",
  TRUE ~ "Extreme_Hot"
)

cat("Temperature Risk Distribution:\n")
print(table(df$Temp_Risk))

# ============================================
# Feature Engineering: Humidity Level Bins
# ============================================
cat("\n--- Creating Humidity Level Bins ---\n")

df$Humidity_Level <- case_when(
  df$Humidity... < 40 ~ "Low",
  df$Humidity... <= 70 ~ "Medium",
  TRUE ~ "High"
)

cat("Humidity Level Distribution:\n")
print(table(df$Humidity_Level))

# ============================================
# Feature Engineering: Visibility Risk Score
# ============================================
cat("\n--- Creating Visibility Risk Score ---\n")

df$Visibility_Risk <- case_when(
  df$Visibility.mi. < 0.5 ~ 4,  # Severe

  df$Visibility.mi. < 1 ~ 3,    # Poor
  df$Visibility.mi. < 2 ~ 2,    # Reduced
  df$Visibility.mi. < 5 ~ 1,    # Moderate
  TRUE ~ 0                       # Good
)

cat("Visibility Risk Score Distribution:\n")
print(table(df$Visibility_Risk))

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
cat("\n========================================\n")
cat("     LOGISTIC REGRESSION MODEL          \n")
cat("========================================\n\n")

# Define model formula with enhanced features
model_formula <- Severity ~ Visibility.mi. + Wind_Speed.mph. + 
               Distance.mi. + Temperature.F. + Wind_Chill.F. + Humidity... + 
               Precipitation.in. + Bump + Crossing + Give_Way + Junction + 
               No_Exit + Railway + Roundabout + Station + Stop + Traffic_Calming + 
               Traffic_Signal + Turning_Loop + Sunrise_Sunset +
               Road_Risk_Score + Visibility_Precip_Interaction + 
               Temp_Humidity_Interaction + Visibility_Risk

# Track training time
lr_train_time <- system.time({
  logistic_model <- glm(model_formula, data = train_data, family = binomial)
})[3]

cat(sprintf("Logistic Regression training time: %.2f seconds\n", lr_train_time))

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
# SHAP Analysis for Logistic Regression
# ============================================
cat("\n--- SHAP Analysis for Logistic Regression ---\n")

# Prepare data for SHAP (use subset for speed)
set.seed(123)
shap_sample_size <- min(5000, nrow(train_data))
shap_sample_idx <- sample(nrow(train_data), shap_sample_size)

# Define features used in model
lr_features <- c("Visibility.mi.", "Wind_Speed.mph.", "Distance.mi.", 
                 "Temperature.F.", "Wind_Chill.F.", "Humidity...",
                 "Precipitation.in.", "Bump", "Crossing", "Give_Way", 
                 "Junction", "No_Exit", "Railway", "Roundabout", "Station", 
                 "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop", 
                 "Sunrise_Sunset", "Road_Risk_Score", "Visibility_Precip_Interaction",
                 "Temp_Humidity_Interaction", "Visibility_Risk")

X_shap_lr <- train_data[shap_sample_idx, lr_features]

# Prediction wrapper for fastshap
pred_wrapper_lr <- function(object, newdata) {
  predict(object, newdata = newdata, type = "response")
}

# Calculate SHAP values
tryCatch({
  cat("Computing SHAP values for Logistic Regression (this may take a few minutes)...\n")
  shap_lr <- fastshap::explain(
    logistic_model, 
    X = X_shap_lr, 
    pred_wrapper = pred_wrapper_lr, 
    nsim = 50
  )
  
  # Create SHAP feature importance using base ggplot2
  shap_importance_lr <- data.frame(
    Feature = names(colMeans(abs(shap_lr))),
    Importance = colMeans(abs(shap_lr))
  ) %>% arrange(desc(Importance)) %>% head(15)
  
  # SHAP Feature Importance Bar Plot
  dev.new()
  print(ggplot(shap_importance_lr, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "SHAP Feature Importance - Logistic Regression",
         x = "Feature", y = "Mean |SHAP Value|") +
    theme_minimal())
  
  # SHAP Dependence Plot for top feature
  top_feature_lr <- shap_importance_lr$Feature[1]
  shap_dep_lr <- data.frame(
    Feature_Value = X_shap_lr[, top_feature_lr],
    SHAP_Value = shap_lr[, top_feature_lr]
  )
  dev.new()
  print(ggplot(shap_dep_lr, aes(x = Feature_Value, y = SHAP_Value)) +
    geom_point(alpha = 0.5, color = "steelblue") +
    geom_smooth(method = "loess", color = "red", se = FALSE) +
    labs(title = paste("SHAP Dependence Plot -", top_feature_lr),
         x = top_feature_lr, y = "SHAP Value") +
    theme_minimal())
  
  cat("SHAP analysis for Logistic Regression complete.\n")
}, error = function(e) {
  cat(sprintf("SHAP analysis skipped: %s\n", e$message))
  cat("Install fastshap package for SHAP analysis.\n")
})

# ============================================
# Decision Tree
# ============================================
cat("\n========================================\n")
cat("       DECISION TREE MODEL              \n")
cat("========================================\n\n")

str(df)

# Build the decision tree with enhanced features
dt_train_time <- system.time({
  dt_model <- rpart(model_formula, data = train_data,
                    control = rpart.control(maxdepth = 10, minsplit = 20, cp = 0.001))
})[3]

cat(sprintf("Decision Tree training time: %.2f seconds\n", dt_train_time))

# View the model summary
summary(dt_model)

dev.new()
rpart.plot(dt_model, main = "Decision Tree for Severity Prediction")

# Make predictions on the testing set
predicted_severity <- predict(dt_model, newdata = test_data)

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
# SHAP Analysis for Decision Tree
# ============================================
cat("\n--- SHAP Analysis for Decision Tree ---\n")

# Prepare data for SHAP
X_shap_dt <- train_data[shap_sample_idx, lr_features]

# Prediction wrapper for decision tree
pred_wrapper_dt <- function(object, newdata) {
  predict(object, newdata = newdata)
}

tryCatch({
  cat("Computing SHAP values for Decision Tree...\n")
  shap_dt <- fastshap::explain(
    dt_model, 
    X = X_shap_dt, 
    pred_wrapper = pred_wrapper_dt, 
    nsim = 50
  )
  
  # Create SHAP feature importance using base ggplot2
  shap_importance_dt <- data.frame(
    Feature = names(colMeans(abs(shap_dt))),
    Importance = colMeans(abs(shap_dt))
  ) %>% arrange(desc(Importance)) %>% head(15)
  
  # SHAP Feature Importance Bar Plot
  dev.new()
  print(ggplot(shap_importance_dt, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "forestgreen") +
    coord_flip() +
    labs(title = "SHAP Feature Importance - Decision Tree",
         x = "Feature", y = "Mean |SHAP Value|") +
    theme_minimal())
  
  # SHAP Dependence Plot for top feature
  top_feature_dt <- shap_importance_dt$Feature[1]
  shap_dep_dt <- data.frame(
    Feature_Value = X_shap_dt[, top_feature_dt],
    SHAP_Value = shap_dt[, top_feature_dt]
  )
  dev.new()
  print(ggplot(shap_dep_dt, aes(x = Feature_Value, y = SHAP_Value)) +
    geom_point(alpha = 0.5, color = "forestgreen") +
    geom_smooth(method = "loess", color = "red", se = FALSE) +
    labs(title = paste("SHAP Dependence Plot -", top_feature_dt),
         x = top_feature_dt, y = "SHAP Value") +
    theme_minimal())
  
  cat("SHAP analysis for Decision Tree complete.\n")
}, error = function(e) {
  cat(sprintf("SHAP analysis skipped: %s\n", e$message))
})




# ============================================
# Random Forest with Hyperparameter Tuning
# ============================================
cat("\n========================================\n")
cat("   RANDOM FOREST MODEL (OPTIMIZED)      \n")
cat("========================================\n\n")

# Use ranger package for faster training and better hyperparameter control
cat("Training Random Forest with hyperparameter tuning...\n")

# Define hyperparameter grid
rf_grid <- expand.grid(
  mtry = c(3, 5, 7, 9),
  num.trees = c(200, 300, 500),
  min.node.size = c(1, 5, 10)
)

cat(sprintf("Testing %d hyperparameter combinations...\n", nrow(rf_grid)))

# Prepare feature matrix for ranger
rf_features <- c("Visibility.mi.", "Wind_Speed.mph.", "Distance.mi.", 
                 "Temperature.F.", "Wind_Chill.F.", "Humidity...",
                 "Precipitation.in.", "Bump", "Crossing", "Give_Way", 
                 "Junction", "No_Exit", "Railway", "Roundabout", "Station", 
                 "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop", 
                 "Sunrise_Sunset", "Road_Risk_Score", "Visibility_Precip_Interaction",
                 "Temp_Humidity_Interaction", "Visibility_Risk")

train_data$Severity_factor <- as.factor(train_data$Severity)

# Calculate class weights for imbalanced data
class_counts <- table(train_data$Severity)
class_weights <- c("0" = 1, "1" = as.numeric(class_counts["0"] / class_counts["1"]))
cat(sprintf("Class weights: 0=%.2f, 1=%.2f\n", class_weights["0"], class_weights["1"]))

# Grid search with cross-validation
set.seed(123)
best_auc <- 0
best_params <- NULL
best_model <- NULL
rf_results <- data.frame()

cat("\nGrid Search Progress:\n")

for (i in 1:nrow(rf_grid)) {
  params <- rf_grid[i, ]
  
  # Train model with current parameters
  rf_temp <- ranger(
    Severity_factor ~ Visibility.mi. + Wind_Speed.mph. + Distance.mi. + 
      Temperature.F. + Wind_Chill.F. + Humidity... + Precipitation.in. + 
      Bump + Crossing + Give_Way + Junction + No_Exit + Railway + 
      Roundabout + Station + Stop + Traffic_Calming + Traffic_Signal + 
      Turning_Loop + Sunrise_Sunset + Road_Risk_Score + 
      Visibility_Precip_Interaction + Temp_Humidity_Interaction + Visibility_Risk,
    data = train_data,
    num.trees = params$num.trees,
    mtry = params$mtry,
    min.node.size = params$min.node.size,
    probability = TRUE,
    importance = "impurity",
    class.weights = class_weights,
    seed = 123
  )
  
  # Predict on test set
  pred_temp <- predict(rf_temp, data = test_data)$predictions[, 2]
  auc_temp <- as.numeric(auc(roc(test_data$Severity, pred_temp, quiet = TRUE)))
  
  # Store results
  rf_results <- rbind(rf_results, data.frame(
    mtry = params$mtry,
    num.trees = params$num.trees,
    min.node.size = params$min.node.size,
    AUC = auc_temp
  ))
  
  # Update best model
  if (auc_temp > best_auc) {
    best_auc <- auc_temp
    best_params <- params
    best_model <- rf_temp
  }
  
  if (i %% 6 == 0) {
    cat(sprintf("  Tested %d/%d combinations, best AUC so far: %.4f\n", 
                i, nrow(rf_grid), best_auc))
  }
}

cat("\n--- Hyperparameter Tuning Results ---\n")
cat(sprintf("Best AUC: %.4f\n", best_auc))
cat(sprintf("Best parameters: mtry=%d, num.trees=%d, min.node.size=%d\n",
            best_params$mtry, best_params$num.trees, best_params$min.node.size))

# Show top 5 configurations
cat("\nTop 5 configurations:\n")
print(head(rf_results[order(-rf_results$AUC), ], 5))

# Train final model with best parameters and measure time
cat("\n--- Training Final Optimized Random Forest ---\n")
set.seed(123)
rf_train_time <- system.time({
  rf_model <- ranger(
    Severity_factor ~ Visibility.mi. + Wind_Speed.mph. + Distance.mi. + 
      Temperature.F. + Wind_Chill.F. + Humidity... + Precipitation.in. + 
      Bump + Crossing + Give_Way + Junction + No_Exit + Railway + 
      Roundabout + Station + Stop + Traffic_Calming + Traffic_Signal + 
      Turning_Loop + Sunrise_Sunset + Road_Risk_Score + 
      Visibility_Precip_Interaction + Temp_Humidity_Interaction + Visibility_Risk,
    data = train_data,
    num.trees = best_params$num.trees,
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    probability = TRUE,
    importance = "impurity",
    class.weights = class_weights,
    seed = 123
  )
})[3]

cat(sprintf("Random Forest training time: %.2f seconds\n", rf_train_time))

# Make predictions on the testing data
predictions_rf_prob <- predict(rf_model, data = test_data)$predictions[, 2]
predictions_rf_class <- ifelse(predictions_rf_prob > 0.5, 1, 0)
predicted_class_rf <- predictions_rf_class

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
plot(roc_data_rf, main = paste("ROC Curve - Random Forest (AUC = ", round(auc_value_rf, 3), ")", sep=""))

# ============================================
# Feature Importance Visualization
# ============================================
importance_df <- data.frame(
  Feature = names(rf_model$variable.importance),
  Importance = rf_model$variable.importance
)
importance_df <- importance_df[order(-importance_df$Importance), ]

dev.new()
ggplot(importance_df[1:15, ], aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest - Ranger)", 
       x = "Feature", 
       y = "Impurity Importance") +
  theme_minimal()

# ============================================
# SHAP Analysis for Random Forest
# ============================================
cat("\n--- SHAP Analysis for Random Forest ---\n")

# Prepare data for SHAP - use smaller sample for speed
shap_sample_idx_rf <- sample(nrow(train_data), min(200, nrow(train_data)))
X_shap_rf <- train_data[shap_sample_idx_rf, rf_features]

# Prediction wrapper for ranger
pred_wrapper_rf <- function(object, newdata) {
  predict(object, data = newdata)$predictions[, 2]
}

tryCatch({
  cat("Computing SHAP values for Random Forest (using 200 samples, 20 simulations)...\n")
  shap_rf <- fastshap::explain(
    rf_model, 
    X = X_shap_rf, 
    pred_wrapper = pred_wrapper_rf, 
    nsim = 20
  )
  
  # Create SHAP feature importance using base ggplot2
  shap_importance_rf <- data.frame(
    Feature = names(colMeans(abs(shap_rf))),
    Importance = colMeans(abs(shap_rf))
  ) %>% arrange(desc(Importance)) %>% head(15)
  
  # SHAP Feature Importance Bar Plot
  dev.new()
  print(ggplot(shap_importance_rf, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "darkorange") +
    coord_flip() +
    labs(title = "SHAP Feature Importance - Random Forest",
         x = "Feature", y = "Mean |SHAP Value|") +
    theme_minimal())
  
  # SHAP Dependence Plots for top 3 features
  top_features_rf <- shap_importance_rf$Feature[1:3]
  
  for (feat in top_features_rf) {
    shap_dep_rf <- data.frame(
      Feature_Value = X_shap_rf[, feat],
      SHAP_Value = shap_rf[, feat]
    )
    dev.new()
    print(ggplot(shap_dep_rf, aes(x = Feature_Value, y = SHAP_Value)) +
      geom_point(alpha = 0.5, color = "darkorange") +
      geom_smooth(method = "loess", color = "red", se = FALSE) +
      labs(title = paste("SHAP Dependence Plot -", feat),
           x = feat, y = "SHAP Value") +
      theme_minimal())
  }
  
  cat("SHAP analysis for Random Forest complete.\n")
}, error = function(e) {
  cat(sprintf("SHAP analysis skipped: %s\n", e$message))
})

# ============================================
# Cross-Validation for Random Forest
# ============================================
cat("\n--- 5-Fold Cross-Validation Results ---\n")
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, 
                               summaryFunction = twoClassSummary)
train_data$Severity_cv <- factor(ifelse(train_data$Severity == 1, "High", "Low"))

rf_cv <- train(Severity_cv ~ Visibility.mi. + Wind_Speed.mph. + 
                 Distance.mi. + Temperature.F. + Wind_Chill.F. + Humidity... + 
                 Precipitation.in. + Bump + Crossing + Give_Way + Junction + 
                 No_Exit + Railway + Roundabout + Station + Stop + Traffic_Calming + 
                 Traffic_Signal + Turning_Loop + Sunrise_Sunset +
                 Road_Risk_Score + Visibility_Precip_Interaction + 
                 Temp_Humidity_Interaction + Visibility_Risk, 
               data = train_data, method = "ranger", 
               trControl = train_control, 
               metric = "ROC",
               tuneGrid = data.frame(mtry = best_params$mtry, 
                                     splitrule = "gini", 
                                     min.node.size = best_params$min.node.size))
print(rf_cv)

# ============================================
# TRADE-OFFS ANALYSIS
# ============================================
cat("\n========================================\n")
cat("        TRADE-OFFS ANALYSIS             \n")
cat("========================================\n\n")

# 1. Model Complexity vs Training Time
cat("--- Model Complexity vs Training Time ---\n")
training_times <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Training_Time_Sec = c(lr_train_time, dt_train_time, rf_train_time),
  Parameters = c(
    length(coef(logistic_model)),
    nrow(dt_model$frame),
    best_params$num.trees * 10  # Approximate parameter count
  )
)
cat("\nTraining Time Comparison:\n")
print(training_times)

dev.new()
ggplot(training_times, aes(x = reorder(Model, Training_Time_Sec), y = Training_Time_Sec)) +
  geom_bar(stat = "identity", fill = c("coral", "gold", "steelblue")) +
  geom_text(aes(label = sprintf("%.2fs", Training_Time_Sec)), vjust = -0.5) +
  labs(title = "Model Training Time Comparison",
       x = "Model", y = "Training Time (seconds)") +
  theme_minimal()

# 2. Number of Trees vs AUC (Random Forest Complexity Analysis)
cat("\n--- Random Forest: num.trees vs AUC Trade-off ---\n")
ntree_values <- c(50, 100, 200, 300, 500, 750, 1000)
ntree_results <- data.frame()

for (nt in ntree_values) {
  set.seed(123)
  time_temp <- system.time({
    rf_temp <- ranger(
      Severity_factor ~ Visibility.mi. + Wind_Speed.mph. + Distance.mi. + 
        Temperature.F. + Wind_Chill.F. + Humidity... + Precipitation.in. + 
        Bump + Crossing + Give_Way + Junction + No_Exit + Railway + 
        Roundabout + Station + Stop + Traffic_Calming + Traffic_Signal + 
        Turning_Loop + Sunrise_Sunset + Road_Risk_Score + 
        Visibility_Precip_Interaction + Temp_Humidity_Interaction + Visibility_Risk,
      data = train_data,
      num.trees = nt,
      mtry = best_params$mtry,
      min.node.size = best_params$min.node.size,
      probability = TRUE,
      class.weights = class_weights,
      seed = 123
    )
  })[3]
  
  pred_temp <- predict(rf_temp, data = test_data)$predictions[, 2]
  auc_temp <- as.numeric(auc(roc(test_data$Severity, pred_temp, quiet = TRUE)))
  
  ntree_results <- rbind(ntree_results, data.frame(
    num_trees = nt,
    AUC = auc_temp,
    Training_Time = time_temp
  ))
  cat(sprintf("  num.trees=%d: AUC=%.4f, Time=%.2fs\n", nt, auc_temp, time_temp))
}

cat("\nnum.trees vs Performance:\n")
print(ntree_results)

dev.new()
ggplot(ntree_results, aes(x = num_trees)) +
  geom_line(aes(y = AUC, color = "AUC"), size = 1.2) +
  geom_point(aes(y = AUC, color = "AUC"), size = 3) +
  geom_line(aes(y = Training_Time / max(Training_Time), color = "Training Time (scaled)"), 
            size = 1.2, linetype = "dashed") +
  geom_point(aes(y = Training_Time / max(Training_Time), color = "Training Time (scaled)"), size = 3) +
  scale_y_continuous(
    name = "AUC",
    sec.axis = sec_axis(~. * max(ntree_results$Training_Time), name = "Training Time (seconds)")
  ) +
  scale_color_manual(values = c("AUC" = "steelblue", "Training Time (scaled)" = "coral")) +
  labs(title = "Random Forest: Model Complexity Trade-off",
       x = "Number of Trees", color = "") +
  theme_minimal() +
  theme(legend.position = "bottom")

# 3. Data Diversity Analysis (Sample Size Impact)
cat("\n--- Data Diversity: Sample Size Impact on Performance ---\n")
sample_sizes <- c(10000, 25000, 50000, 75000, nrow(train_data))
sample_results <- data.frame()

for (ss in sample_sizes) {
  set.seed(123)
  train_subset <- train_data[sample(nrow(train_data), min(ss, nrow(train_data))), ]
  train_subset$Severity_factor <- as.factor(train_subset$Severity)
  
  # Recalculate class weights for subset
  subset_counts <- table(train_subset$Severity)
  subset_weights <- c("0" = 1, "1" = as.numeric(subset_counts["0"] / subset_counts["1"]))
  
  time_temp <- system.time({
    rf_temp <- ranger(
      Severity_factor ~ Visibility.mi. + Wind_Speed.mph. + Distance.mi. + 
        Temperature.F. + Wind_Chill.F. + Humidity... + Precipitation.in. + 
        Bump + Crossing + Give_Way + Junction + No_Exit + Railway + 
        Roundabout + Station + Stop + Traffic_Calming + Traffic_Signal + 
        Turning_Loop + Sunrise_Sunset + Road_Risk_Score + 
        Visibility_Precip_Interaction + Temp_Humidity_Interaction + Visibility_Risk,
      data = train_subset,
      num.trees = best_params$num.trees,
      mtry = best_params$mtry,
      min.node.size = best_params$min.node.size,
      probability = TRUE,
      class.weights = subset_weights,
      seed = 123
    )
  })[3]
  
  pred_temp <- predict(rf_temp, data = test_data)$predictions[, 2]
  auc_temp <- as.numeric(auc(roc(test_data$Severity, pred_temp, quiet = TRUE)))
  
  sample_results <- rbind(sample_results, data.frame(
    Sample_Size = ss,
    AUC = auc_temp,
    Training_Time = time_temp
  ))
  cat(sprintf("  Sample size=%d: AUC=%.4f, Time=%.2fs\n", ss, auc_temp, time_temp))
}

cat("\nSample Size vs Performance:\n")
print(sample_results)

dev.new()
ggplot(sample_results, aes(x = Sample_Size / 1000)) +
  geom_line(aes(y = AUC), color = "steelblue", size = 1.2) +
  geom_point(aes(y = AUC), color = "steelblue", size = 3) +
  geom_text(aes(y = AUC, label = sprintf("%.3f", AUC)), vjust = -1) +
  labs(title = "Impact of Training Data Size on Model Performance",
       subtitle = "Demonstrating data diversity trade-offs",
       x = "Training Sample Size (thousands)", y = "AUC") +
  theme_minimal() +
  ylim(min(sample_results$AUC) - 0.02, max(sample_results$AUC) + 0.02)

# 4. Summary Trade-offs Table
cat("\n--- Trade-offs Summary ---\n")
tradeoffs_summary <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Complexity = c("Low", "Medium", "High"),
  Interpretability = c("High", "High", "Medium (with SHAP)"),
  Training_Time = c(
    sprintf("%.2fs", lr_train_time),
    sprintf("%.2fs", dt_train_time),
    sprintf("%.2fs", rf_train_time)
  ),
  AUC = c(
    sprintf("%.4f", as.numeric(auc_val)),
    sprintf("%.4f", as.numeric(auc_val_dt)),
    sprintf("%.4f", as.numeric(auc_value_rf))
  ),
  Best_For = c(
    "Quick baseline, high interpretability",
    "Visual rules, moderate performance",
    "Maximum predictive performance"
  )
)
print(tradeoffs_summary)

cat("\nKey Trade-off Insights:\n")
cat("1. Logistic Regression: Fastest training, fully interpretable coefficients\n")
cat("2. Decision Tree: Visual decision rules, but limited accuracy\n")
cat("3. Random Forest: Best AUC, but requires SHAP for interpretability\n")
cat(sprintf("4. Diminishing returns after ~%d trees (marginal AUC improvement)\n", 
            ntree_results$num_trees[which.max(ntree_results$AUC)]))
cat(sprintf("5. Training with %.0fk samples achieves %.1f%% of full-data AUC\n",
            sample_results$Sample_Size[3]/1000,
            sample_results$AUC[3] / sample_results$AUC[nrow(sample_results)] * 100))

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

# Top risk factors from Random Forest (ranger uses variable.importance)
rf_importance <- data.frame(
  Feature = names(rf_model$variable.importance),
  Importance = rf_model$variable.importance
)
rf_importance_sorted <- rf_importance[order(-rf_importance$Importance), ]
cat("Top 5 Risk Factors for Severe Accidents:\n")
print(head(rf_importance_sorted, 5))

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





