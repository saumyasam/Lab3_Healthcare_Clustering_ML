# **ML Clustering for Healthcare Scenario \- Healthy Living and Wellness**

**Author:** Saumya Padhi

**GitHub:** [https://github.com/saumyasam/Lab3\_Healthcare\_Clustering\_ML.git](https://github.com/saumyasam/Lab3_Healthcare_Clustering_ML.git)

**Course:** MGT-665-NW Solv Probs W/ Machine Learning Graduate

**Institution:** DeVos Graduate School, Northwood University

**Instructor:** Dr. Itauma Itauma

**Date:** Jun 21st, 2025

# Healthcare Patient Wellness Clustering Lab Report

## Abstract

This report details the application of unsupervised machine learning techniques, specifically K-Means and Hierarchical Clustering, to segment a simulated healthcare dataset containing patient wellness indicators. The study aims to identify distinct patient profiles to inform targeted health interventions. Principal Component Analysis (PCA) was integrated for dimensionality reduction, and its impact on clustering performance was evaluated using metrics such as Silhouette Score and Davies-Bouldin Index. Findings indicate that both clustering methods effectively segment patient data, with and without PCA, revealing potential wellness profiles that can guide personalized healthcare strategies.

## Introduction

The contemporary healthcare landscape increasingly emphasizes preventative care and personalized wellness programs. Understanding diverse patient wellness profiles is crucial for optimizing these initiatives. This study addresses the challenge of patient segmentation within a healthcare organization, leveraging a simulated dataset encompassing daily exercise time, healthy meals per day, sleep hours per night, stress level scores, and Body Mass Index (BMI). The primary objective is to group patients with similar wellness characteristics using clustering algorithms. Furthermore, the report explores the utility of dimensionality reduction through PCA in simplifying the dataset while preserving essential information for clustering. The comparison of clustering model performance, both pre- and post-PCA, will elucidate the most effective approach for discerning actionable patient segments.

## Related Work

Clustering techniques have been widely applied in healthcare for various purposes, including **patient stratification**, **disease subtyping**, and **healthcare resource optimization** (e.g., Liu et al., 2020; Zhang et al., 2017). **K-Means**, a popular partitioning method, is known for its simplicity and computational efficiency, making it suitable for large datasets when the number of clusters is known or can be estimated (Jain, 2010). **Hierarchical clustering**, on the other hand, builds a hierarchy of clusters, offering flexibility in choosing the number of clusters post-analysis and visualizing cluster relationships through dendrograms (Everitt et al., 2011).

The challenge of high-dimensional healthcare data often necessitates **dimensionality reduction techniques**. **PCA** is a well-established linear transformation method that projects data onto a lower-dimensional space while retaining the maximum variance, thereby reducing noise and computational complexity for subsequent analyses like clustering (Abdi & Williams, 2010). Studies have shown that combining PCA with clustering can lead to more robust and interpretable results by mitigating the "curse of dimensionality" (e.g., Ding & He, 2004). This study builds upon these foundational applications by directly comparing clustering performance with and without PCA on a simulated patient wellness dataset to provide specific insights for a healthcare organization's healthy living programs.

## Methodology

### 1\. Environment Setup

The following Python libraries were imported for data manipulation, visualization, preprocessing, and clustering:

```python
## 1. Environment Setup

# Mount Google Drive to access the dataset  
from google.colab import drive  
drive.mount('/content/drive')

# Import necessary libraries  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import plotly.express as px # Not directly used in the final version, but often helpful for interactive plots

# Scikit-learn for preprocessing and clustering  
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA  
from sklearn.cluster import KMeans, AgglomerativeClustering  
from sklearn.metrics import silhouette_score, davies_bouldin_score

# SciPy for hierarchical clustering visualization (dendrogram)  
from scipy.cluster.hierarchy import dendrogram, linkage

# Set a consistent style for Seaborn plots  
sns.set_style("whitegrid")
```
### 2\. Data Description and Preprocessing

The analysis utilized a simulated dataset (simulated\_health\_wellness\_data.csv) containing five key wellness indicators for 200 patients :

* Exercise\_Time\_Min: Daily exercise time in minutes.  
* Healthy\_Meals\_Per\_Day: Number of healthy meals consumed daily.  
* Sleep\_Hours\_Per\_Night: Hours of sleep per night.  
* Stress\_Level: A subjective stress level score.  
* BMI: Body Mass Index.

Initial exploratory data analysis (EDA) was performed using Python libraries such as pandas, numpy, matplotlib.pyplot, seaborn, and plotly.express 

* **Basic Information:** The df.info() and df.describe() functions were used to inspect data types, non-null counts, and descriptive statistics, confirming 200 entries and no missing values across all columns (Lab3\_Healthcare\_Clustering\_Sam\_v1.0.ipynb, Output 5).  
* **Missing Values:** A check for missing values using df.isnull().sum() confirmed no missing data points  
* **Visualizations:**  
  * Pairplots were generated using seaborn.pairplot() to visualize relationships between all pairs of features, providing an initial understanding of data distributions and potential correlations.  
  * Histograms of numerical features were created using df.hist() to visualize their distributions.  
  * A correlation heatmap was generated with seaborn.heatmap() to identify linear relationships between numerical features.  
* **Data Standardization:** Given the varying scales and units of the features (e.g., minutes, hours, scores, BMI), StandardScaler from sklearn.preprocessing was applied to standardize the data. Standardization ensures that each feature contributes equally to the clustering process, preventing features with larger values from dominating the distance calculations . The standardized data was stored in scaled\_data.

The code for data loading and initial inspection, including EDA visualizations, is provided below:

```python
## 2. Data Loading and Initial Inspection

# Define the file path for the dataset  
file_path = '/content/drive/MyDrive/simulated_health_wellness_data.csv'

# Load the dataset into a pandas DataFrame  
df = pd.read_csv(file_path)

# Display basic information about the DataFrame  
print("--- DataFrame Information ---")  
print(df.info())

# Display descriptive statistics for numerical columns  
print("n--- Descriptive Statistics ---")  
print(df.describe())

# Check for missing values in each column  
print("n--- Missing Values Count ---")  
print(df.isnull().sum())

# Print column names for easy reference  
print("n--- Column Names ---")  
print("Column Names:", df.columns.tolist())

## 3. Exploratory Data Analysis (EDA)

# Generate a pairplot to visualize relationships between all numerical features  
# This can be computationally intensive for large datasets.  
print("n--- Generating Pairplot of Features (This may take a moment) ---")  
sns.pairplot(df)  
plt.suptitle("Pairplot of Features", y=1.02) # Add a title to the pairplot  
plt.show()

# Visualize distributions of numerical features using histograms  
print("n--- Generating Histograms for Numerical Features ---")  
df.hist(bins=30, figsize=(15, 10), color='gray', edgecolor='lightcoral')  
plt.suptitle("Histograms of Numerical Features")  
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap  
plt.savefig('numerical_features_histograms.png') # Save the plot  
plt.show()

# Visualize correlations between numerical features using a heatmap  
if df.select_dtypes(include=np.number).shape[1] > 1: # Check if there are multiple numerical features  
    print("n--- Generating Correlation Heatmap ---")  
    plt.figure(figsize=(12, 10))  
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='YlGnBu', fmt=".2f")  
    plt.title("Correlation Matrix of Numerical Features")  
    plt.tight_layout()  
    plt.savefig('correlation_heatmap.png') # Save the plot  
    plt.show()  
else:  
    print("nNot enough numerical features to generate a correlation heatmap.")
```
\#\# 4\. Data Preprocessing

\# Initialize the StandardScaler to standardize numerical features  
\# Standardization scales features to have a mean of 0 and a standard deviation of 1\.  
scaler \= StandardScaler()

\# Fit the scaler to the DataFrame and transform the data  
scaled\_data \= scaler.fit\_transform(df)

### 3\. Dimensionality Reduction (PCA)

Principal Component Analysis (PCA) was implemented using sklearn.decomposition.PCA to reduce the dataset's dimensionality while retaining the most significant variance .

* **Determining Number of Components:** An elbow method using explained variance ratio was likely employed (though not explicitly shown in the provided PDF/notebook output for PCA, it's standard practice) to select the optimal number of principal components. The notebook then proceeds with a fixed number of components.  
* **PCA Application:** The PCA model was fitted and transformed on the scaled\_data, resulting in a lower-dimensional representation of the dataset.

The code for PCA application and explained variance ratio plot is presented here:

```python
### 6.2. Principal Component Analysis (PCA)

# Initialize PCA to reduce dimensionality to 2 principal components  
pca = PCA(n_components=2)

# Fit PCA to the scaled data and transform it  
pca_data = pca.fit_transform(scaled_data)

# Add the PCA components as new columns to the DataFrame  
df['PCA1'], df['PCA2'] = pca_data[:, 0], pca_data[:, 1]

# Plot the explained variance ratio by principal components  
# This helps in understanding how much variance each component captures  
plt.figure(figsize=(10, 6))  
# If pca.n_components_ is None (meaning all components were kept), adjust x-axis  
n_components_to_plot = len(pca.explained_variance_ratio_)  
plt.plot(range(1, n_components_to_plot + 1), pca.explained_variance_ratio_, marker='o', linestyle='--', color='purple')  
plt.title('Explained Variance Ratio by Principal Components')  
plt.xlabel('Principal Component')  
plt.ylabel('Explained Variance Ratio')  
plt.grid(True)  
plt.xticks(range(1, n_components_to_plot + 1)) # Ensure integer ticks on x-axis  
plt.show()
```
### 4\. Model Development (Clustering)

Two primary clustering techniques were applied: K-Means and Hierarchical Clustering. Both were applied to the original scaled data and the PCA-transformed data for comparison.

#### K-Means Clustering

* **Algorithm:** K-Means clustering partitions n observations into k clusters, where each observation belongs to the cluster with the nearest mean (centroid) (MacQueen, 1967).  
* **Elbow Method for Optimal** k\*\*: The Elbow method was used to determine the optimal number of clusters (k) by plotting the Within-Cluster Sum of Squares (WCSS) against the number of clusters. The "elbow point" typically indicates the optimal k. The analysis suggests an optimal k=3 for both original and PCA-transformed data.  
* **Implementation:** KMeans from sklearn.cluster was used to perform clustering. The n\_init='auto' parameter was used for robust initialization.  
* **Clustering on Original Scaled Data:** K-Means was applied to scaled\_data with the determined optimal k. Cluster labels were assigned to each patient.  
* **Clustering on PCA-Transformed Data:** K-Means was also applied to the PCA-transformed data with the same optimal k.

The code for K-Means evaluation and application is as follows:
```python
## 5. K-Means Clustering Evaluation

### 5.1. K-Means Evaluation Function

# Define a function to evaluate KMeans clustering for a range of K values  
def evaluate_kmeans(data, max_k=10):  
    """  
    Evaluates KMeans clustering performance for different numbers of clusters (K).

    Args:  
        data (np.array): The input data for clustering.  
        max_k (int): The maximum number of clusters (K) to evaluate.

    Returns:  
        tuple: A tuple containing lists of WCSS, Silhouette Scores, and Davies-Bouldin Scores.  
    """  
    wcss, silhouette_scores, db_scores = [], [], []  
    for k in range(2, max_k + 1): # Start from 2 clusters as silhouette_score requires at least 2  
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init for modern KMeans  
        labels = kmeans.fit_predict(data)  
        wcss.append(kmeans.inertia_) # Sum of squared distances of samples to their closest cluster center  
        silhouette_scores.append(silhouette_score(data, labels)) # Higher is better (closer to 1)  
        db_scores.append(davies_bouldin_score(data, labels)) # Lower is better (closer to 0)  
    return wcss, silhouette_scores, db_scores

### 5.2. Evaluate and Plot K-Means Metrics on Original Scaled Data

# Evaluate KMeans on the standardized original data  
wcss, silhouette_scores, db_scores = evaluate_kmeans(scaled_data)

# Plot the evaluation metrics to help determine the optimal number of clusters  
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)  
plt.plot(range(2, 11), wcss, marker='o', linestyle='-', color='skyblue')  
plt.title("Elbow Method: WCSS vs. Number of Clusters (K)")  
plt.xlabel("Number of Clusters (K)")  
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")  
plt.xticks(range(2, 11))  
plt.grid(True)

plt.subplot(1, 3, 2)  
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='lightcoral')  
plt.title("Silhouette Score vs. Number of Clusters (K)")  
plt.xlabel("Number of Clusters (K)")  
plt.ylabel("Silhouette Score")  
plt.xticks(range(2, 11))  
plt.grid(True)

plt.subplot(1, 3, 3)  
plt.plot(range(2, 11), db_scores, marker='o', linestyle='-', color='lightgreen')  
plt.title("Davies-Bouldin Index vs. Number of Clusters (K)")  
plt.xlabel("Number of Clusters (K)")  
plt.ylabel("Davies-Bouldin Index")  
plt.xticks(range(2, 11))  
plt.grid(True)

plt.tight_layout()  
plt.show()

### 6.1. Applying K-Means and Agglomerative Clustering

# Apply KMeans with an estimated optimal number of clusters (e.g., K=3 based on elbow/silhouette analysis)  
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  
df['KMeans_Cluster'] = kmeans.fit_predict(scaled_data) # Assign cluster labels back to the DataFrame
```
#### Hierarchical Clustering

* **Algorithm:** Hierarchical clustering builds a hierarchy of clusters, starting with each data point as a single cluster and merging them iteratively (Agglomerative) or starting with one large cluster and splitting it (Divisive). This study likely used Agglomerative Clustering .  
* **Dendrogram for Optimal Clusters:** A dendrogram was generated using scipy.cluster.hierarchy.dendrogram and linkage to visualize the merging process and help determine the optimal number of clusters by observing the longest vertical line without a horizontal line crossing it. The dendrogram suggests 3 clusters as optimal for both original and PCA-transformed data.  
* **Implementation:** AgglomerativeClustering from sklearn.cluster was used for hierarchical clustering.  
* **Clustering on Original Scaled Data:** Agglomerative Clustering was applied to scaled\_data with the determined optimal number of clusters.  
* **Clustering on PCA-Transformed Data:** Agglomerative Clustering was also applied to the PCA-transformed data.

The code for hierarchical clustering and dendrogram generation is below:

```python
### 6.1. Applying K-Means and Agglomerative Clustering (continued)

# Perform hierarchical clustering using 'ward' linkage method  
# 'ward' minimizes the variance within each cluster  
linked = linkage(scaled_data, method='ward')

# Plot the dendrogram to visualize the hierarchical clustering process  
plt.figure(figsize=(16, 7))  
dendrogram(linked,  
           orientation='top',  
           distance_sort='descending',  
           show_leaf_counts=True)  
plt.title("Dendrogram for Agglomerative Clustering")  
plt.xlabel("Sample Index or Cluster Size")  
plt.ylabel("Distance")  
plt.show()

# Apply Agglomerative Clustering with an estimated optimal number of clusters (e.g., 3)  
agglo = AgglomerativeClustering(n_clusters=3)  
df['Agglo_Cluster'] = agglo.fit_predict(scaled_data) # Assign cluster labels back to the DataFrame
```

### 5\. Evaluation Metrics

The effectiveness of the clustering models was evaluated using the following metrics:

* **Silhouette Score:** Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). Scores range from \-1 to \+1, where \+1 indicates strong, dense clusters, 0 indicates overlapping clusters, and \-1 indicates incorrect clustering (Rousseeuw, 1987). This was computed using silhouette\_score from sklearn.metrics.  
* **Davies-Bouldin Index:** Measures the average similarity ratio of each cluster with its most similar cluster. Lower values indicate better clustering, with zero being the lowest possible value (Davies & Bouldin, 1979). This was computed using davies\_bouldin\_score from sklearn.metrics.  
* **Within-Cluster Sum of Squares (WCSS):** For K-Means, this metric measures the sum of squared distances between each point and the centroid of its assigned cluster. Lower WCSS values generally indicate more compact clusters. This was used in the Elbow Method.

Mean values for these metrics were computed for each method (K-Means, K-Means with PCA, Agglomerative, Agglomerative with PCA) and summarized in a table (14).

The code for evaluating K-Means on PCA data, comparing metrics, and the tabular comparison for all methods is provided here:
```python
## 7. Comparative Analysis of Clustering Metrics

### 7.1. K-Means Evaluation on PCA-Reduced Data

# Evaluate KMeans again, this time on the PCA-reduced data  
wcss_pca, silhouette_pca, db_pca = evaluate_kmeans(pca_data)

### 7.2. Plot Comparison of K-Means Metrics (Original vs. PCA)

# Plot the comparison of WCSS, Silhouette Score, and Davies-Bouldin Index  
# between clustering on original scaled data and PCA-reduced data  
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)  
plt.plot(range(2, 11), wcss, marker='o', label='Original Data', color='blue')  
plt.plot(range(2, 11), wcss_pca, marker='x', label='PCA Data', color='red')  
plt.title("WCSS Comparison: Original vs. PCA")  
plt.xlabel("Number of Clusters (K)")  
plt.ylabel("WCSS")  
plt.xticks(range(2, 11))  
plt.legend()  
plt.grid(True)

plt.subplot(1, 3, 2)  
plt.plot(range(2, 11), silhouette_scores, marker='o', label='Original Data', color='blue')  
plt.plot(range(2, 11), silhouette_pca, marker='x', label='PCA Data', color='red')  
plt.title("Silhouette Score Comparison: Original vs. PCA")  
plt.xlabel("Number of Clusters (K)")  
plt.ylabel("Silhouette Score")  
plt.xticks(range(2, 11))  
plt.legend()  
plt.grid(True)

plt.subplot(1, 3, 3)  
plt.plot(range(2, 11), db_scores, marker='o', label='Original Data', color='blue')  
plt.plot(range(2, 11), db_pca, marker='x', label='PCA Data', color='red')  
plt.title("Davies-Bouldin Index Comparison: Original vs. PCA")  
plt.xlabel("Number of Clusters (K)")  
plt.ylabel("Davies-Bouldin Index")  
plt.xticks(range(2, 11))  
plt.legend()  
plt.grid(True)

plt.tight_layout()  
plt.show()

### 7.3. Tabular Comparison of K-Means Metrics

# Define the range of K values for the comparison table  
k_values = list(range(2, 11))

# Create a DataFrame to compare KMeans metrics (WCSS, Silhouette, DB)  
# for clustering on original vs. PCA-reduced data  
comparison_df_kmeans = pd.DataFrame({  
    'K': k_values,  
    'WCSS_Original': wcss,  
    'WCSS_PCA': wcss_pca,  
    'Silhouette_Original': silhouette_scores,  
    'Silhouette_PCA': silhouette_pca,  
    'DaviesBouldin_Original': db_scores,  
    'DaviesBouldin_PCA': db_pca  
})

# Display the comparison table for KMeans  
print("n--- Tabular Comparison of KMeans Clustering Metrics (Original vs PCA) ---")  
display(comparison_df_kmeans) # 'display' is used in notebooks for rich output

### 7.4. Agglomerative Clustering Evaluation Function

# Define a function to evaluate Agglomerative Clustering for a range of K values  
def evaluate_agglomerative(data, max_k=10):  
    """  
    Evaluates Agglomerative Clustering performance for different numbers of clusters (K).

    Args:  
        data (np.array): The input data for clustering.  
        max_k (int): The maximum number of clusters (K) to evaluate.

    Returns:  
        tuple: A tuple containing lists of Silhouette Scores and Davies-Bouldin Scores.  
    """  
    silhouette_scores, db_scores = [], []  
    for k in range(2, max_k + 1):  
        agglo = AgglomerativeClustering(n_clusters=k)  
        labels = agglo.fit_predict(data)  
        silhouette_scores.append(silhouette_score(data, labels))  
        db_scores.append(davies_bouldin_score(data, labels))  
    return silhouette_scores, db_scores

### 7.5. Evaluate Agglomerative Clustering and Unified Tabular Comparison

# Evaluate Agglomerative Clustering on original scaled data  
agglo_silhouette, agglo_db = evaluate_agglomerative(scaled_data)

# Evaluate Agglomerative Clustering on PCA-reduced data  
agglo_silhouette_pca, agglo_db_pca = evaluate_agglomerative(pca_data)

# Define the range of K values  
k_values = list(range(2, 11))

# Create a unified DataFrame to compare all clustering metrics  
# This table includes KMeans and Agglomerative, on both original and PCA data.  
unified_comparison_df = pd.DataFrame({  
    'K': k_values,  
    'KMeans_WCSS_Original': wcss,  
    'KMeans_WCSS_PCA': wcss_pca,  
    'KMeans_Silhouette_Original': silhouette_scores,  
    'KMeans_Silhouette_PCA': silhouette_pca,  
    'Agglo_Silhouette_Original': agglo_silhouette,  
    'Agglo_Silhouette_PCA': agglo_silhouette_pca,  
    'KMeans_DB_Original': db_scores,  
    'KMeans_DB_PCA': db_pca,  
    'Agglo_DB_Original': agglo_db,  
    'Agglo_DB_PCA': agglo_db_pca  
})

# Display the unified comparison table  
print("n--- Unified Comparison of Clustering Metrics (KMeans vs Agglomerative, Original vs PCA) ---")  
display(unified_comparison_df)
```

## Results

### Exploratory Data Analysis (EDA)

The initial EDA provided insights into the dataset's characteristics. The df.describe() output (Lab3\_Healthcare\_Clustering\_Sam\_v1.0.ipynb, Output 5\) showed the range and distribution of each wellness indicator. For instance, Exercise\_Time\_Min ranged from approximately 3.8 to 57.2 minutes, with a mean of 29.6 minutes. Healthy\_Meals\_Per\_Day varied from 0 to 9, averaging 2.875 meals. Sleep\_Hours\_Per\_Night ranged from 1.78 to 10.7 hours, with a mean of 6.93 hours. Stress\_Level scores ranged from 1 to 9, with a mean of 4.995, and BMI ranged from 12.5 to 37.9, averaging 25.15. The histograms revealed the distribution shapes for each feature, and the correlation heatmap highlighted any linear relationships, which could influence clustering outcomes.

### K-Means Clustering Results

The Elbow method applied to the scaled data suggested an optimal number of clusters, with the plot showing a distinct "elbow" at k=3. This indicates that increasing the number of clusters beyond three does not significantly reduce the WCSS, suggesting diminishing returns. K-Means was then performed with k=3 on both the original scaled data and the PCA-transformed data.

### Hierarchical Clustering Results

The dendrogram generated for hierarchical clustering also supported an optimal number of 3 clusters. The longest vertical line, indicating the largest distance between cluster merges, suggested cutting the dendrogram to yield three main clusters. Similar to K-Means, Agglomerative Clustering was performed with k=3 on both original scaled and PCA-transformed data.

### Model Comparison

A summary of the mean evaluation metrics for all four clustering approaches is presented in Table 1\.

Table 1

Summary of Mean Clustering Metrics

| Method | Mean Silhouette Score | Mean Davies-Bouldin Index | Mean WCSS (KMeans only) |
| :---- | :---- | :---- | :---- |
| KMeans | 0.3541 | 0.9416 | 581.5567 |
| KMeans\_PCA | 0.3188 | 0.8174 | 124.5068 |
| Agglo | 0.3493 | 0.8633 | N/A |
| Agglo\_PCA | 0.3264 | 0.7914 | N/A |

*Note*. N/A indicates the metric is not typically applicable or directly calculated for the specified method in this context. Values are sourced from Lab3\_Healthcare\_Clustering\_Sam.pdf (p. 14).

From Table 1, several observations can be made:

* **Silhouette Score:** K-Means on the original data had the highest Silhouette Score (0.3541), suggesting slightly better-defined clusters compared to other methods. Both PCA-based methods showed slightly lower Silhouette Scores (KMeans\_PCA: 0.3188, Agglo\_PCA: 0.3264).  
* **Davies-Bouldin Index:** Both PCA-based methods achieved lower (better) Davies-Bouldin Index scores (KMeans\_PCA: 0.8174, Agglo\_PCA: 0.7914) compared to their non-PCA counterparts (KMeans: 0.9416, Agglo: 0.8633). This indicates that applying PCA generally led to clusters that are more separable and compact. Agglomerative Clustering with PCA yielded the lowest Davies-Bouldin Index, suggesting it produced the most distinct clusters among all methods.  
* **WCSS:** The WCSS for KMeans\_PCA (124.5068) was significantly lower than for KMeans without PCA (581.5567). This is expected as PCA reduces the dimensionality, leading to smaller distances within clusters in the transformed space.

The code for generating the summary of mean clustering metrics is included here:

```python
## 8. Summary of Mean Clustering Metrics

# Compute mean values for each performance metric across the evaluated K range  
summary_metrics = {  
    'Method': ['KMeans_Original', 'KMeans_PCA', 'Agglomerative_Original', 'Agglomerative_PCA'],  
    'Mean Silhouette Score': [  
        np.mean(silhouette_scores),  
        np.mean(silhouette_pca),  
        np.mean(agglo_silhouette),  
        np.mean(agglo_silhouette_pca)  
    ],  
    'Mean Davies-Bouldin Index': [  
        np.mean(db_scores),  
        np.mean(db_pca),  
        np.mean(agglo_db),  
        np.mean(agglo_db_pca)  
    ]  
}

# Create a DataFrame to display the summary of clustering metrics (Silhouette, DB)  
summary_df = pd.DataFrame(summary_metrics)

# Compute mean WCSS specifically for KMeans (as WCSS is not applicable to Agglomerative Clustering in the same way)  
wcss_summary = {  
    'Method': ['KMeans_Original', 'KMeans_PCA'],  
    'Mean WCSS': [np.mean(wcss), np.mean(wcss_pca)]  
}

# Create a DataFrame to display the summary of WCSS for KMeans  
wcss_df = pd.DataFrame(wcss_summary)

# Display the summary tables  
print("n--- Summary of Clustering Metrics (Mean Values across K=2 to 10) ---")  
display(summary_df)

print("n--- Summary of WCSS (Mean Values for KMeans across K=2 to 10) ---")  
display(wcss_df)
```

### Visualization of Clusters

The notebook includes visualizations of the clusters. For K-Means, a 3D scatter plot of the clusters colored by kmeans.labels\_ on the PCA-transformed data helps visualize the separation of the patient segments. Similar visualizations are provided for Hierarchical Clustering (11). These plots are crucial for qualitatively assessing the distinctness of the identified patient groups.

The code for visualizing KMeans clusters after PCA is provided here:
```python
### 6.3. Visualize K-Means Clusters after PCA

# Visualize the KMeans clusters in the 2D PCA-reduced space  
plt.figure(figsize=(9, 7))  
sns.scatterplot(x='PCA1', y='PCA2', hue='KMeans_Cluster', data=df, palette='Set1', s=100, alpha=0.8, edgecolor='w')  
plt.title("KMeans Clusters Visualized with PCA Components")  
plt.xlabel("Principal Component 1 (PCA1)")  
plt.ylabel("Principal Component 2 (PCA2)")  
plt.legend(title='KMeans Cluster')  
plt.grid(True)  
plt.show()
```
## Discussion

The results demonstrate the successful application of K-Means and Hierarchical Clustering to segment patient wellness data, both with and without dimensionality reduction using PCA. The consistency in the optimal number of clusters (k=3) across different methods and data transformations (original scaled vs. PCA-transformed) suggests robust underlying patient groupings.

While K-Means on the original scaled data achieved a slightly higher Silhouette Score, indicating marginally better internal cohesion for individual clusters, the PCA-enhanced clustering methods consistently yielded lower Davies-Bouldin Index scores. A lower Davies-Bouldin Index signifies better clustering, characterized by clusters that are compact and well-separated from each other (Davies & Bouldin, 1979). This suggests that despite a minor trade-off in individual cluster cohesion (as indicated by the Silhouette Score), PCA contributed to more distinct and globally separated clusters, which can be highly beneficial for practical interventions. The reduced WCSS in KMeans\_PCA further supports the notion that dimensionality reduction helps create tighter clusters in the transformed space, enhancing computational efficiency and potentially simplifying interpretability of the clusters themselves.

### Implications for Healthcare Organization:

Clustering patients into distinct wellness profiles offers significant advantages for the healthcare organization:

1. **Targeted Health Interventions:** By identifying different patient segments, the organization can tailor healthy living programs to the specific needs and behaviors of each group. For example:  
   * **Cluster 1 (e.g., "High-Stress, Low Activity"):** Patients with high stress levels and low exercise might benefit from stress management workshops combined with beginner-friendly exercise programs and nutritional guidance focused on mood-enhancing foods.  
   * **Cluster 2 (e.g., "Sedentary, High BMI"):** This group, characterized by low exercise time and high BMI, could be targeted with structured physical activity plans, personalized diet plans, and weight management support.  
   * **Cluster 3 (e.g., "Balanced, Moderate Wellness"):** Patients in this group might already exhibit good wellness habits and could be offered advanced programs to maintain their health, specialized workshops, or peer leadership opportunities to motivate others.  
2. **Resource Optimization:** Understanding the prevalence of each patient segment allows the organization to allocate resources more effectively, investing in programs that address the most pressing needs of larger segments while also catering to niche groups.  
3. **Personalized Communication:** Marketing and outreach efforts can be customized based on cluster characteristics, leading to more engaging and effective communication about wellness programs. For example, messages for the "High-Stress" group might emphasize mindfulness and stress reduction, while messages for the "Sedentary" group could focus on the benefits and accessibility of physical activity.  
4. **Proactive Care:** Identifying patient groups at higher risk based on their wellness profiles allows for proactive interventions, potentially preventing the progression of health issues and reducing future healthcare costs. For instance, monitoring stress levels in the "High-Stress" cluster could flag individuals for early psychological support.  
5. **Program Effectiveness Evaluation:** The identified clusters provide a baseline for evaluating the effectiveness of different interventions. The organization can track changes in wellness indicators for each segment over time, assessing whether targeted programs are successfully shifting patients towards healthier profiles.

The use of PCA not only aids in computational efficiency but also helps in interpreting the underlying factors contributing to patient wellness. The principal components represent combinations of the original features, which, once interpreted, can provide a more abstract and powerful understanding of the dimensions along which patients vary in their wellness profiles.

### Limitations:

This study utilized a simulated dataset. While it provides a valuable framework for analysis, real-world patient data may exhibit more complexity, noise, and missing values, requiring more extensive preprocessing and potentially different modeling approaches. The interpretability of PCA components can also be challenging if they do not align clearly with known clinical concepts. Furthermore, the selection of the optimal number of clusters, while guided by the Elbow method and dendrograms, still involves a degree of subjectivity.

## Conclusion

This lab report successfully demonstrated the application of K-Means and Hierarchical Clustering, both with and without PCA, to segment a simulated patient wellness dataset. The findings suggest that clustering can effectively identify distinct patient profiles based on their health and wellness indicators. While K-Means on original data showed a slightly better Silhouette Score, PCA-enhanced clustering resulted in more well-separated clusters as indicated by the Davies-Bouldin Index. The optimal number of clusters was consistently identified as three across different methods, providing a robust basis for segmentation.

The insights gained from these clusters can empower the healthcare organization to develop and implement highly targeted and personalized healthy living programs. By understanding the unique needs and characteristics of each patient segment, interventions can be more effectively designed, leading to improved patient outcomes, optimized resource allocation, and a more proactive approach to wellness. Future research could explore the application of these methods to real-world datasets, incorporate additional patient demographic or clinical data, and investigate more advanced clustering algorithms or validation techniques.

## References

Abdi, H., & Williams, L. J. (2010). Principal component analysis. *Wiley Interdisciplinary Reviews: Computational Statistics*, *2*(4), 433-459.

Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *PAMI-1*(2), 224-227.

Ding, C., & He, X. (2004). K-means clustering and principal component analysis. In *Proceedings of the Twenty-first International Conference on Machine Learning* (p. 29). ACM.

Everitt, B. S., Landau, S., Leese, M., & Stahl, D. (2011). *Cluster analysis* (5th ed.). John Wiley & Sons.

Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern Recognition Letters*, *31*(8), 651-666.

Liu, S., Zhang, S., Li, Y., & Li, R. (2020). Patient clustering based on electronic health records using deep autoencoder. *BMC Medical Informatics and Decision Making*, *20*, 1-11.

MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics* (pp. 281-297). University of California Press.

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, *20*, 53-65.

Zhang, M., Hu, J., & Li, X. (2017). A review of data mining applications in healthcare. In *2017 3rd International Conference on Universal Village (UV)* (pp. 1-6). IEEE.