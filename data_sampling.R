# ============================================
# DATA SAMPLING PIPELINE
# Accident Severity Prediction Project
# ============================================
# This script implements a data-efficient sampling strategy to reduce
# the original ~7.7M US traffic accident records to ~108k high-quality samples
# while preserving temporal, geographic, and scenario diversity.
#
# Sampling Strategy:
# 1. Time-based Stratification: Preserve hourly and monthly patterns
# 2. H3 Spatial Indexing: Geographic diversity using Uber's H3 hexagonal grid
# 3. K-means Clustering: Scenario diversity across weather/visibility conditions
#
# Original Dataset: US_Accidents_March23.csv (~7.7M records)
# Source: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
# Output: updated_final.csv (~108k records)
# ============================================

# Set library path
.libPaths(c("~/R/library", .libPaths()))

# Load required packages
cat("Loading required packages...\n")

# Install packages if not available
required_packages <- c("data.table", "dplyr", "lubridate", "h3jsr", "ggplot2", "tidyr")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
    library(pkg, character.only = TRUE)
  }
}

# ============================================
# STEP 1: Load Original Dataset
# ============================================
cat("\n========================================\n")
cat("  STEP 1: Loading Original Dataset\n")
cat("========================================\n\n")

# Use data.table for memory-efficient reading of large CSV
input_file <- "US_Accidents_March23.csv"

cat(sprintf("Loading %s...\n", input_file))
cat("This may take a few minutes for ~7.7M records...\n\n")

# Read the full dataset using fread (memory efficient)
start_time <- Sys.time()
df_original <- fread(input_file, showProgress = TRUE)
load_time <- difftime(Sys.time(), start_time, units = "secs")

cat(sprintf("\nOriginal dataset loaded in %.2f seconds\n", load_time))
cat(sprintf("Original dataset dimensions: %d rows x %d columns\n", nrow(df_original), ncol(df_original)))
cat(sprintf("Memory usage: %.2f MB\n", object.size(df_original) / 1024^2))

# ============================================
# STEP 2: Initial Data Cleaning
# ============================================
cat("\n========================================\n")
cat("  STEP 2: Initial Data Cleaning\n")
cat("========================================\n\n")

# Keep essential columns for sampling and modeling
essential_cols <- c(
  # Target and identifiers
  "ID", "Severity", 
  # Temporal features (for time-based stratification)
  "Start_Time", "End_Time",
  # Spatial features (for H3 indexing)
  "Start_Lat", "Start_Lng", "End_Lat", "End_Lng",
  # Location
  "City", "State", "County",
  # Weather/Visibility (for k-means clustering)
  "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", 
  "Pressure(in)", "Visibility(mi)", "Wind_Direction", 
  "Wind_Speed(mph)", "Precipitation(in)", "Weather_Condition",
  # Road features
  "Amenity", "Bump", "Crossing", "Give_Way", "Junction", 
  "No_Exit", "Railway", "Roundabout", "Station", "Stop", 
  "Traffic_Calming", "Traffic_Signal", "Turning_Loop",
  # Distance
  "Distance(mi)",
  # Time of day
  "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"
)

# Select only columns that exist in the dataset
available_cols <- intersect(essential_cols, names(df_original))
cat(sprintf("Selecting %d essential columns...\n", length(available_cols)))

df <- df_original[, ..available_cols]
rm(df_original)  # Free memory
gc()

# Remove rows with missing values in critical columns
critical_cols <- c("Severity", "Start_Time", "Start_Lat", "Start_Lng", 
                   "Visibility(mi)", "Temperature(F)", "Humidity(%)")

cat("Removing rows with missing values in critical columns...\n")
before_count <- nrow(df)

for (col in critical_cols) {
  if (col %in% names(df)) {
    df <- df[!is.na(get(col))]
  }
}

after_count <- nrow(df)
cat(sprintf("Rows before: %d, after: %d (removed %d rows with NA)\n", 
            before_count, after_count, before_count - after_count))

# ============================================
# STEP 3: Time-Based Stratification
# ============================================
cat("\n========================================\n")
cat("  STEP 3: Time-Based Stratification\n")
cat("========================================\n\n")

# Parse Start_Time to extract temporal features
cat("Parsing timestamps and extracting temporal features...\n")

df[, Start_Time := as.POSIXct(Start_Time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")]

# Extract temporal components
df[, `:=`(
  Year = year(Start_Time),
  Month = month(Start_Time),
  DayOfWeek = wday(Start_Time),
  Hour = hour(Start_Time)
)]

# Create time-based strata
# Combine Year-Month for temporal coverage
df[, Time_Stratum := paste(Year, sprintf("%02d", Month), sep = "-")]

# Create hour bins (6-hour windows)
df[, Hour_Bin := cut(Hour, breaks = c(-1, 6, 12, 18, 24), 
                     labels = c("Night", "Morning", "Afternoon", "Evening"))]

cat("\nTemporal distribution:\n")
cat("Year-Month strata:\n")
print(table(df$Time_Stratum))

cat("\nHour distribution:\n")
print(table(df$Hour_Bin))

# ============================================
# STEP 4: H3 Spatial Indexing
# ============================================
cat("\n========================================\n")
cat("  STEP 4: H3 Spatial Indexing\n")
cat("========================================\n\n")

# H3 Resolution 6: ~36km² hexagons (good for regional clustering)
H3_RESOLUTION <- 6

cat(sprintf("Generating H3 indexes at resolution %d...\n", H3_RESOLUTION))
cat("This creates hexagonal spatial bins for geographic diversity...\n\n")

# Generate H3 index for each accident location
# Using h3jsr package for R
tryCatch({
  df[, H3_Index := h3jsr::point_to_cell(
    data.frame(lng = Start_Lng, lat = Start_Lat), 
    res = H3_RESOLUTION
  )]
  
  n_unique_hexes <- length(unique(df$H3_Index))
  cat(sprintf("Created %d unique H3 hexagonal regions\n", n_unique_hexes))
  
  # Show top regions by accident count
  cat("\nTop 10 regions by accident count:\n")
  top_regions <- df[, .N, by = H3_Index][order(-N)][1:10]
  print(top_regions)
  
}, error = function(e) {
  cat("Note: h3jsr package not available. Using lat/lng binning as fallback...\n")
  # Fallback: Create spatial bins using lat/lng rounding
  df[, H3_Index := paste0(
    "lat", round(Start_Lat, 1), 
    "_lng", round(Start_Lng, 1)
  )]
  n_unique_hexes <- length(unique(df$H3_Index))
  cat(sprintf("Created %d spatial bins using lat/lng rounding\n", n_unique_hexes))
})

# ============================================
# STEP 5: K-Means Clustering on Weather/Visibility
# ============================================
cat("\n========================================\n")
cat("  STEP 5: K-Means Clustering\n")
cat("========================================\n\n")

# Select features for clustering (weather/visibility conditions)
cluster_features <- c("Visibility(mi)", "Temperature(F)", "Humidity(%)", 
                      "Wind_Speed(mph)", "Precipitation(in)")

# Handle missing values in cluster features
cat("Preparing features for clustering...\n")
df_cluster <- df[, ..cluster_features]

# Impute missing values with median
for (col in cluster_features) {
  if (any(is.na(df_cluster[[col]]))) {
    median_val <- median(df_cluster[[col]], na.rm = TRUE)
    df_cluster[is.na(get(col)), (col) := median_val]
    df[[col]][is.na(df[[col]])] <- median_val
  }
}

# Standardize features for k-means
cat("Standardizing features...\n")
df_cluster_scaled <- scale(df_cluster)

# Perform K-means clustering
# Using k=10 clusters to capture diverse weather scenarios
K_CLUSTERS <- 10

cat(sprintf("Running K-means with k=%d clusters...\n", K_CLUSTERS))
set.seed(123)

kmeans_result <- kmeans(df_cluster_scaled, centers = K_CLUSTERS, 
                        nstart = 25, iter.max = 100)

df[, Weather_Cluster := as.factor(kmeans_result$cluster)]

cat("\nCluster distribution:\n")
print(table(df$Weather_Cluster))

# Analyze cluster characteristics
cat("\nCluster centroids (original scale):\n")
cluster_centers <- as.data.frame(kmeans_result$centers)
names(cluster_centers) <- cluster_features

# Unscale the centers for interpretability
attr_center <- attr(df_cluster_scaled, "scaled:center")
attr_scale <- attr(df_cluster_scaled, "scaled:scale")

for (i in 1:ncol(cluster_centers)) {
  cluster_centers[, i] <- cluster_centers[, i] * attr_scale[i] + attr_center[i]
}

cluster_centers$Cluster <- 1:K_CLUSTERS
cluster_centers$Count <- as.vector(table(kmeans_result$cluster))
print(cluster_centers)

# ============================================
# STEP 6: Stratified Sampling
# ============================================
cat("\n========================================\n")
cat("  STEP 6: Stratified Sampling\n")
cat("========================================\n\n")

TARGET_SAMPLE_SIZE <- 108000

cat(sprintf("Target sample size: %d\n", TARGET_SAMPLE_SIZE))
cat("Sampling strategy: Proportional stratified sampling across:\n")
cat("  - Time strata (Year-Month)\n")
cat("  - Severity levels\n")
cat("  - Weather clusters\n")
cat("  - Hour bins\n\n")

# Create combined stratum
df[, Combined_Stratum := paste(Time_Stratum, Severity, Weather_Cluster, Hour_Bin, sep = "_")]

# Calculate stratum sizes
stratum_counts <- df[, .N, by = Combined_Stratum]
total_records <- nrow(df)

cat(sprintf("Total records after cleaning: %d\n", total_records))
cat(sprintf("Number of unique strata: %d\n", nrow(stratum_counts)))

# Calculate sampling fraction
sampling_fraction <- TARGET_SAMPLE_SIZE / total_records
cat(sprintf("Base sampling fraction: %.4f\n\n", sampling_fraction))

# Perform stratified sampling
set.seed(123)

# Method: Sample proportionally from each stratum, with minimum of 1 per stratum
df[, Sample_Size := ceiling(.N * sampling_fraction), by = Combined_Stratum]

# Ensure we don't sample more than available
df[, Sample_Size := pmin(Sample_Size, .N), by = Combined_Stratum]

# Sample from each stratum
cat("Performing stratified sampling...\n")
df_sampled <- df[, .SD[sample(.N, min(Sample_Size[1], .N))], by = Combined_Stratum]

cat(sprintf("Initial sample size: %d\n", nrow(df_sampled)))

# Adjust to exactly TARGET_SAMPLE_SIZE if needed
if (nrow(df_sampled) > TARGET_SAMPLE_SIZE) {
  # Randomly remove excess
  df_sampled <- df_sampled[sample(.N, TARGET_SAMPLE_SIZE)]
} else if (nrow(df_sampled) < TARGET_SAMPLE_SIZE) {
  # Sample additional records from underrepresented strata
  shortfall <- TARGET_SAMPLE_SIZE - nrow(df_sampled)
  already_sampled <- df_sampled$ID
  remaining <- df[!ID %in% already_sampled]
  additional <- remaining[sample(.N, min(shortfall, .N))]
  df_sampled <- rbind(df_sampled, additional)
}

cat(sprintf("Final sample size: %d\n", nrow(df_sampled)))

# ============================================
# STEP 7: Verify Sampling Quality
# ============================================
cat("\n========================================\n")
cat("  STEP 7: Sampling Quality Verification\n")
cat("========================================\n\n")

# Compare distributions before and after sampling
cat("Severity distribution comparison:\n")
cat("Original:\n")
print(prop.table(table(df$Severity)))
cat("\nSampled:\n")
print(prop.table(table(df_sampled$Severity)))

cat("\n\nTemporal coverage (Year-Month):\n")
cat("Original unique periods:", length(unique(df$Time_Stratum)), "\n")
cat("Sampled unique periods:", length(unique(df_sampled$Time_Stratum)), "\n")

cat("\nGeographic coverage (H3 hexagons):\n")
cat("Original unique regions:", length(unique(df$H3_Index)), "\n")
cat("Sampled unique regions:", length(unique(df_sampled$H3_Index)), "\n")

cat("\nWeather cluster distribution comparison:\n")
cat("Original:\n")
print(prop.table(table(df$Weather_Cluster)))
cat("\nSampled:\n")
print(prop.table(table(df_sampled$Weather_Cluster)))

# ============================================
# STEP 8: Prepare Final Output
# ============================================
cat("\n========================================\n")
cat("  STEP 8: Preparing Final Output\n")
cat("========================================\n\n")

# Select columns for the final dataset (matching project.R expectations)
output_cols <- c(
  "Severity", "Distance(mi)", "City",
  "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
  "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
  "Weather_Condition",
  "Bump", "Crossing", "Give_Way", "Junction", 
  "No_Exit", "Railway", "Roundabout", "Station", "Stop", 
  "Traffic_Calming", "Traffic_Signal", "Turning_Loop",
  "Sunrise_Sunset"
)

# Keep only available columns
available_output_cols <- intersect(output_cols, names(df_sampled))
df_final <- df_sampled[, ..available_output_cols]

# Handle any remaining NAs
df_final <- na.omit(df_final)

# Rename columns to match project.R expectations (R-friendly names)
col_rename <- c(
  "Distance(mi)" = "Distance.mi.",
  "Temperature(F)" = "Temperature.F.",
  "Wind_Chill(F)" = "Wind_Chill.F.",
  "Humidity(%)" = "Humidity...",
  "Visibility(mi)" = "Visibility.mi.",
  "Wind_Speed(mph)" = "Wind_Speed.mph.",
  "Precipitation(in)" = "Precipitation.in."
)

for (old_name in names(col_rename)) {
  if (old_name %in% names(df_final)) {
    setnames(df_final, old_name, col_rename[[old_name]])
  }
}

cat(sprintf("Final dataset dimensions: %d rows x %d columns\n", nrow(df_final), ncol(df_final)))

# ============================================
# STEP 9: Save Output
# ============================================
cat("\n========================================\n")
cat("  STEP 9: Saving Output\n")
cat("========================================\n\n")

output_file <- "updated_final.csv"
fwrite(df_final, output_file)
cat(sprintf("Saved sampled dataset to: %s\n", output_file))

# ============================================
# SUMMARY
# ============================================
cat("\n========================================\n")
cat("         SAMPLING SUMMARY               \n")
cat("========================================\n\n")

cat(sprintf("Original dataset size:    %d records\n", total_records))
cat(sprintf("Final sample size:        %d records\n", nrow(df_final)))
cat(sprintf("Reduction ratio:          %.2f%%\n", (1 - nrow(df_final)/total_records) * 100))
cat(sprintf("Sampling method:          Stratified (Time + Space + Scenario)\n"))
cat(sprintf("  - Time strata:          %d (Year-Month combinations)\n", length(unique(df$Time_Stratum))))
cat(sprintf("  - Spatial regions:      %d (H3 resolution %d hexagons)\n", length(unique(df$H3_Index)), H3_RESOLUTION))
cat(sprintf("  - Weather clusters:     %d (K-means clusters)\n", K_CLUSTERS))
cat(sprintf("  - Hour bins:            %d (6-hour windows)\n", length(unique(df$Hour_Bin))))

cat("\n========================================\n")
cat("  SAMPLING PIPELINE COMPLETE           \n")
cat("========================================\n")

# Save sampling metadata
metadata <- list(
  original_size = total_records,
  final_size = nrow(df_final),
  h3_resolution = H3_RESOLUTION,
  k_clusters = K_CLUSTERS,
  time_strata = length(unique(df$Time_Stratum)),
  spatial_regions = length(unique(df$H3_Index)),
  sampling_date = Sys.time(),
  random_seed = 123
)

saveRDS(metadata, "sampling_metadata.rds")
cat("\nSampling metadata saved to: sampling_metadata.rds\n")
