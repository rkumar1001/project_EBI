# 🚗 Traffic Accident Severity Prediction

A comprehensive machine learning project to predict traffic accident severity using environmental, road infrastructure, and weather conditions data. Built with R for statistical analysis and predictive modeling.

## 📋 Project Overview

This project analyzes traffic accident data to predict whether an accident will be **low severity** (severity levels 1-2: minor injuries/property damage) or **high severity** (severity levels 3-4: serious injuries/fatalities). The binary classification approach enables targeted emergency response and resource allocation.

### Key Applications
- **Emergency Response**: Prioritize resource deployment based on predicted severity
- **Urban Planning**: Identify dangerous road conditions and infrastructure features
- **Insurance**: Risk assessment and premium calculation based on conditions
- **Traffic Management**: Issue weather-based warnings during high-risk conditions
- **Policy Making**: Data-driven decisions for road safety improvements

## 📊 Dataset

- **Source**: US Traffic Accident Records
- **Size**: ~108,928 observations after preprocessing
- **File**: `updated_final.csv`
- **Target Variable**: `Severity` (binary: 0 = Low, 1 = High)
- **Class Distribution**: 92% Low Severity / 8% High Severity (imbalanced)

### Features Used (20 Predictors)

| Category | Features |
|----------|----------|
| **Environmental** | Temperature (°F), Wind Chill (°F), Humidity (%), Visibility (mi), Wind Speed (mph), Precipitation (in) |
| **Road Infrastructure** | Bump, Crossing, Junction, Railway, Roundabout, Stop, Traffic Signal, Traffic Calming, Give Way, No Exit, Station, Turning Loop |
| **Temporal** | Sunrise/Sunset (Day=1/Night=0 indicator) |
| **Impact** | Distance (mi) - extent of accident impact |

### Engineered Features
| Feature | Description |
|---------|-------------|
| **Weather_Risk** | Categorical: Poor_Visibility, Precipitation, High_Wind, Normal |
| **Road_Risk_Score** | Composite score (0-8) summing road infrastructure features |

## 🔧 Requirements

### System Requirements
- R version 4.0+ (tested on R 4.3.3)
- Linux/macOS/Windows

### R Packages
```r
# Core Data Manipulation
install.packages(c("ggplot2", "dplyr", "tidyr", "stringr"))

# Machine Learning
install.packages(c("caret", "rpart", "rpart.plot", "randomForest"))

# Statistical Analysis
install.packages(c("corrplot", "psych", "PerformanceAnalytics"))

# Model Evaluation
install.packages(c("pROC", "ROCR"))
```

## 🚀 Usage

```bash
# Navigate to project directory
cd /path/to/project_EBI

# Run the complete analysis
Rscript project.R

# Or run interactively in RStudio
# Open project.R and source the file
```

## 🔬 Methodology

### Data Preprocessing
1. **Column Selection**: Dropped irrelevant columns (ID, coordinates, timestamps, descriptions)
2. **Missing Values**: Removed rows with NA values using `na.omit()`
3. **Binary Encoding**: Converted boolean strings ("True"/"False") to 1/0
4. **Target Encoding**: Recoded 4-level severity to binary (1-2 → 0, 3-4 → 1)

### Train-Test Split
- **Ratio**: 70% training / 30% testing
- **Seed**: 123 for reproducibility
- **Single Split**: Same split used across all models for fair comparison

## 📈 Models Implemented

### 1. Logistic Regression
- **Function**: `glm()` with `family = binomial`
- **Purpose**: Baseline binary classification model
- **Output**: Probability scores converted to class predictions (threshold = 0.5)
- **Metrics**: Confusion matrix, MAE, MSE, AUC

### 2. Decision Tree (CART)
- **Package**: `rpart`
- **Purpose**: Interpretable rule-based classification
- **Visualization**: Tree plot using `rpart.plot`
- **Advantage**: Easy to understand decision rules

### 3. Random Forest (Primary Model)
- **Package**: `randomForest`
- **Trees**: 100 decision trees
- **Class Balancing**: Uses `sampsize = c("0" = 5000, "1" = 5000)` for balanced sampling
- **Feature Importance**: MeanDecreaseGini metric
- **Validation**: 10-fold cross-validation using `caret`

## 📉 Results

### Model Performance Comparison

| Model | Accuracy | AUC | Sensitivity | Specificity | Kappa |
|-------|----------|-----|-------------|-------------|-------|
| Logistic Regression | 92.3% | 0.622 | 99.9% | 0.04% | 0.001 |
| Decision Tree | 94.0% | 0.824 | 97.8% | 43.3% | 0.454 |
| **Random Forest** | **91.8%** | **0.841** | 93.4% | **52.8%** | **0.420** |

> **Note**: Random Forest achieves the highest AUC (0.841) and best minority class detection (52.8% specificity) despite slightly lower accuracy.

### Top 5 Risk Factors (Feature Importance)
| Rank | Feature | MeanDecreaseGini |
|------|---------|------------------|
| 1 | Distance (mi) | 988.0 |
| 2 | Humidity (%) | 80.2 |
| 3 | Temperature (°F) | 70.4 |
| 4 | Wind Chill (°F) | 64.2 |
| 5 | Wind Speed (mph) | 60.4 |

### Severity Rate by Weather Condition
| Weather Condition | Total Accidents | High Severity | Severity Rate |
|-------------------|-----------------|---------------|---------------|
| Precipitation | - | - | **16.7%** |
| High Wind (>20mph) | - | - | 9.1% |
| Poor Visibility (<2mi) | - | - | 9.0% |
| Normal | - | - | 7.6% |

### High-Risk Condition Thresholds
- **Visibility**: Severe accidents avg 9.64 mi (vs 9.69 mi overall)
- **Humidity**: Severe accidents avg 67.1% (vs 65.8% overall)
- **Temperature**: Severe accidents avg 60.3°F (vs 61.1°F overall)

## 📁 Project Structure

```
project_EBI/
├── project.R           # Main analysis script (474 lines)
│   ├── Data Loading & Preprocessing (Lines 1-80)
│   ├── Correlation Analysis (Lines 81-100)
│   ├── Class Distribution Analysis (Lines 104-130)
│   ├── Feature Engineering (Lines 131-175)
│   ├── Logistic Regression (Lines 200-245)
│   ├── Decision Tree (Lines 250-290)
│   ├── Random Forest (Lines 295-350)
│   ├── Cross-Validation (Lines 365-380)
│   ├── Model Comparison Dashboard (Lines 385-425)
│   └── Actionable Insights (Lines 430-474)
├── updated_final.csv   # Dataset (~108,928 rows)
├── README.md           # Project documentation
└── Rplots*.pdf         # Generated visualizations (auto-created)
```

## 📊 Output Visualizations

The script generates the following visualizations (saved as PDF):

| Plot | Description |
|------|-------------|
| **Class Distribution** | Bar chart showing severity class imbalance (92% vs 8%) |
| **Correlation Matrix** | Spearman correlation heatmap for numeric variables |
| **Weather Risk Analysis** | Stacked bar chart of severity by weather conditions |
| **Day vs Night Analysis** | Severity patterns by time of day (conditional) |
| **Decision Tree Plot** | Visual representation of CART decision rules |
| **ROC Curve** | Random Forest model discrimination (AUC visualization) |
| **Feature Importance** | Horizontal bar chart of MeanDecreaseGini scores |
| **Model Comparison** | Grouped bar chart comparing all model metrics |

## 🔍 Feature Engineering

### Weather Risk Categories
Created using `case_when()` with priority order:
```r
Poor_Visibility:  Visibility < 2 miles
Precipitation:    Precipitation > 0.1 inches
High_Wind:        Wind Speed > 20 mph
Normal:           All other conditions
```

### Road Risk Score
Composite score (0-8) calculated as sum of binary road features:
```r
Road_Risk_Score = Bump + Crossing + Junction + Railway + 
                  Roundabout + Stop + Traffic_Signal + Traffic_Calming
```

## ⚠️ Known Limitations

1. **Class Imbalance**: ~92% low severity vs ~8% high severity
   - **Mitigation**: Balanced sampling in Random Forest (`sampsize`)
   - Impact: Logistic regression struggles with minority class (0.04% specificity)

2. **Feature Variance**: Some binary road features have limited variance
   - Many observations have 0 for most road infrastructure features

3. **Temporal Granularity**: Only Day/Night indicator available
   - Hour-level or day-of-week patterns cannot be analyzed

4. **Geographic Context**: Location data (lat/lng, state, county) removed during preprocessing
   - Regional patterns not captured in current analysis

## 💡 Key Recommendations (from Analysis)

1. **Deploy additional resources during poor visibility conditions** - 2.2x higher severity rate
2. **Issue warnings when precipitation detected** - 16.7% severity rate (highest)
3. **Focus on intersections with multiple road features** - Road risk score correlates with severity
4. **Consider time-of-day patterns for patrol scheduling** - Day/night severity differences observed


## 👤 Author

**Rajat**

