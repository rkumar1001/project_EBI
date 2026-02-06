# 🚗 Accident Severity Prediction Under Weather & Visibility Conditions

A comprehensive machine learning project analyzing ~7.5M U.S. traffic accident records to predict accident severity using weather, visibility, and road infrastructure features. Built with R featuring advanced stratified sampling, hyperparameter-tuned models, and SHAP-based interpretability.


## 📋 Project Overview

This project implements a complete machine learning pipeline for predicting traffic accident severity (Low vs High) using environmental and road conditions. The binary classification enables targeted emergency response and resource allocation.

### Key Innovations
- **Advanced Stratified Sampling**: Combined time-based stratification, H3 spatial indexing, and K-means clustering
- **Hyperparameter Tuning**: Grid search optimization for Random Forest (36 configurations tested)
- **SHAP Interpretability**: Model-agnostic explanations for all three classifiers
- **Trade-offs Analysis**: Comprehensive comparison of model complexity, training time, and performance

### Applications
- **Emergency Response**: Prioritize resource deployment based on predicted severity
- **Urban Planning**: Identify dangerous road conditions and infrastructure features
- **Traffic Management**: Issue weather-based warnings during high-risk conditions
- **Policy Making**: Data-driven decisions for road safety improvements

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | US Accidents (March 2023 Release) |
| **Original Size** | ~7.5 million records |
| **Sampled Size** | 81,122 records (stratified sample) |
| **File** | `updated_final.csv` |
| **Target Variable** | `Severity` (binary: 0 = Low, 1 = High) |
| **Class Distribution** | 83.6% Low / 16.4% High Severity |

### Stratified Sampling Methodology
The sampling pipeline (`data_sampling.R`) implements a multi-dimensional stratification approach:

1. **Time-Based Stratification**: Year-Month + Hour bins (Morning/Afternoon/Evening/Night)
2. **Spatial Stratification**: H3 hexagonal grid at resolution 6 (~36 km² cells)
3. **Feature-Based Clustering**: K-means (k=10) on weather/visibility features
4. **Combined Strata**: Proportional sampling from each unique combination

### Features (20 Base + 6 Engineered)

| Category | Features |
|----------|----------|
| **Environmental** | Temperature (°F), Wind Chill (°F), Humidity (%), Visibility (mi), Wind Speed (mph), Precipitation (in) |
| **Road Infrastructure** | Bump, Crossing, Junction, Railway, Roundabout, Stop, Traffic Signal, Traffic Calming, Give Way, No Exit, Station, Turning Loop |
| **Temporal** | Sunrise/Sunset (Day=1/Night=0) |
| **Impact** | Distance (mi) - extent of accident impact |

### Engineered Features
| Feature | Description |
|---------|-------------|
| **Weather_Risk** | Categorical: Poor_Visibility, Precipitation, High_Wind, Normal |
| **Visibility_Precip_Interaction** | Visibility × (1 - Precipitation/max) |
| **Temp_Humidity_Interaction** | Temperature × √(Humidity/100) |
| **Wind_Chill_Effect** | Normalized temperature drop due to wind |
| **Temp_Risk** | Categorical: Extreme_Cold, Cold, Normal, Hot, Extreme_Hot |
| **Visibility_Risk** | Score 0-4 based on visibility thresholds |

## 🔧 Requirements

### System Requirements
- R version 4.0+ (tested on R 4.3.3)
- Linux/macOS/Windows
- ~4GB RAM for full dataset processing

### System Dependencies (Linux)
```bash
# Required for spatial packages (h3jsr, sf)
sudo apt-get install -y libudunits2-dev libgdal-dev libproj-dev libgeos-dev
```

### R Packages
```r
# Data Sampling Pipeline
install.packages(c("data.table", "h3jsr", "sf", "dplyr"))

# Core Analysis
install.packages(c("ggplot2", "tidyr", "stringr", "caret"))

# Machine Learning
install.packages(c("rpart", "rpart.plot", "ranger", "fastshap"))

# Statistical Analysis
install.packages(c("corrplot", "psych", "PerformanceAnalytics", "pROC"))
```

## 🚀 Usage

```bash
# Navigate to project directory
cd /path/to/project_EBI

# Step 1: Run data sampling (if using original 7.5M dataset)
Rscript data_sampling.R

# Step 2: Run complete analysis
Rscript project.R
```

## 🔬 Methodology

### Data Preprocessing
1. **Memory-Efficient Loading**: Using `data.table::fread()` for 7.5M records
2. **Missing Value Handling**: NA removal with quality verification
3. **Binary Encoding**: Boolean to 1/0 conversion
4. **Target Encoding**: 4-level severity → binary (1-2 → 0, 3-4 → 1)
5. **Feature Engineering**: 6 derived features for enhanced predictive power

### Train-Test Split
- **Ratio**: 70% training / 30% testing
- **Seed**: 123 for reproducibility

## 📈 Models Implemented

### 1. Logistic Regression
- **Function**: `glm()` with `family = binomial`
- **Purpose**: Baseline binary classification
- **SHAP Analysis**: ✅ Feature importance visualization

### 2. Decision Tree (CART)
- **Package**: `rpart`
- **Configuration**: maxdepth=10, minsplit=20, cp=0.001
- **Visualization**: Tree plot with `rpart.plot`
- **SHAP Analysis**: ✅ Dependence plots for top features

### 3. Random Forest (Optimized)
- **Package**: `ranger` (faster than randomForest)
- **Hyperparameter Tuning**: Grid search over 36 configurations
  - `mtry`: 2, 3, 4, 5, 7, 10
  - `num.trees`: 200, 300, 500
  - `min.node.size`: 5, 10
- **Class Balancing**: Case weights (1:5.14 ratio)
- **Best Parameters**: mtry=5, num.trees=500, min.node.size=10
- **SHAP Analysis**: ✅ Full interpretability suite
- **Cross-Validation**: 5-fold CV with ROC optimization

## 📉 Results

### Model Performance Comparison

| Model | Accuracy | AUC | Sensitivity | Specificity | Training Time |
|-------|----------|-----|-------------|-------------|---------------|
| Logistic Regression | 83.3% | 0.565 | 99.96% | 0.12% | 0.59s |
| Decision Tree | 83.3% | 0.704 | 99.80% | 1.38% | 2.40s |
| **Random Forest** | **83.3%** | **0.894** | 99.91% | 0.81% | **17.01s** |

> **Best Model**: Random Forest with AUC = 0.718 after hyperparameter optimization

### Cross-Validation Results
- **5-Fold CV ROC**: 0.716
- **Consistent performance** across folds

### Top 5 Risk Factors (Feature Importance)
| Rank | Feature | Importance Score |
|------|---------|------------------|
| 1 | Distance (mi) | 3512 |
| 2 | Temp-Humidity Interaction | 1589 |
| 3 | Wind Chill (°F) | 1217 |
| 4 | Humidity (%) | 1166 |
| 5 | Wind Speed (mph) | 1161 |

### Severity Rate by Weather Condition
| Weather Condition | Total Accidents | Severity Rate |
|-------------------|-----------------|---------------|
| **Precipitation** | 965 | **25.2%** |
| High Wind (>20mph) | 1,594 | 21.6% |
| Poor Visibility (<2mi) | 2,870 | 18.2% |
| Normal | 75,693 | 16.1% |

### Trade-offs Analysis Insights
| Insight | Finding |
|---------|---------|
| Trees vs AUC | Diminishing returns after ~500 trees |
| Sample Size | 50k samples achieve 99.9% of full-data AUC |
| Training Time | RF takes 28x longer than LR for 0.15 AUC gain |
| Interpretability | LR fully interpretable; RF requires SHAP |

## 📁 Project Structure

```
project_EBI/
├── data_sampling.R       # Stratified sampling pipeline
│   ├── H3 Spatial Indexing (resolution 6)
│   ├── K-means Clustering (k=10)
│   ├── Time-based Stratification
│   └── Quality Verification
├── project.R             # Main analysis script (~1060 lines)
│   ├── Data Loading & Preprocessing
│   ├── Exploratory Data Analysis
│   ├── Feature Engineering (6 new features)
│   ├── Logistic Regression + SHAP
│   ├── Decision Tree + SHAP
│   ├── Random Forest (Hyperparameter Tuned) + SHAP
│   ├── Cross-Validation
│   ├── Trade-offs Analysis
│   ├── Model Comparison Dashboard
│   └── Actionable Insights
├── US_Accidents_March23.csv  # Original dataset (~7.5M records)
├── updated_final.csv         # Stratified sample (81,122 records)
├── sampling_metadata.rds     # Sampling process metadata
├── README.md                 # Project documentation
└── Rplots*.pdf               # Generated visualizations
```

## 📊 Output Visualizations

| Plot | Description |
|------|-------------|
| **Class Distribution** | Bar chart showing severity distribution |
| **Correlation Matrix** | Spearman correlation heatmap |
| **Weather Risk Analysis** | Severity by weather conditions |
| **Feature Importance** | SHAP-based importance for each model |
| **SHAP Dependence Plots** | Feature effect on predictions |
| **Decision Tree Plot** | Visual CART decision rules |
| **ROC Curves** | Model discrimination comparison |
| **Trade-offs Charts** | Trees vs AUC, Sample size vs Performance |
| **Model Comparison** | Dashboard comparing all metrics |

## 🔍 SHAP Analysis

SHAP (SHapley Additive exPlanations) provides model-agnostic interpretability:

- **Feature Importance Bar Charts**: Mean |SHAP value| per feature
- **Dependence Plots**: How feature values affect predictions
- **Applied to**: All three models (LR, DT, RF)

Example insights from SHAP:
- Higher distance values consistently increase severity predictions
- Temperature-humidity interaction shows non-linear effects
- Wind chill has stronger impact at extreme values

## ⚠️ Limitations & Future Work

### Current Limitations
1. **Class Imbalance**: Addressed with case weights but specificity remains low
2. **Temporal Features**: Only Day/Night; finer granularity could improve predictions
3. **Geographic Patterns**: H3 used for sampling only, not as model feature

### Future Improvements
- [ ] XGBoost/LightGBM implementation
- [ ] SMOTE for synthetic minority oversampling
- [ ] Time-series features (hour, day-of-week, season)
- [ ] Geographic features as model inputs
- [ ] Deep learning approaches (neural networks)

## 💡 Key Recommendations

Based on the analysis findings:

1. **Deploy additional resources during precipitation** - 25.2% severity rate (highest)
2. **Issue warnings for high wind conditions** - 21.6% severity rate
3. **Monitor poor visibility situations** - 18.2% severity rate
4. **Focus on accidents with higher distance impact** - Top risk factor
5. **Consider temperature-humidity combinations** - 2nd most important feature

## 👤 Author

**Rajat**  
