# ML Clustering for Healthcare Scenario - Healthy Living and Wellness
---
**Author:** Saumya Padhi  
**GitHub:** [https://github.com/saumyasam/Lab3_Healthcare_Clustering_ML.git](https://github.com/saumyasam/Lab3_Healthcare_Clustering_ML.git)  
**Course:** MGT-665-NW Solv Probs W/ Machine Learning Graduate  
**Institution:** DeVos Graduate School, Northwood University  
**Instructor:** Dr. Itauma Itauma  
**Date:** Jun 21st, 2025

---
# **Healthcare Patient Wellness Clustering Lab Report**

## **Abstract**

This report details the application of unsupervised machine learning techniques, specifically K-Means and Hierarchical Clustering, to segment a simulated healthcare dataset containing patient wellness indicators. The study aims to identify distinct patient profiles to inform targeted health interventions. Principal Component Analysis (PCA) was integrated for dimensionality reduction, and its impact on clustering performance was evaluated using metrics such as Silhouette Score and Davies-Bouldin Index. Findings indicate that both clustering methods effectively segment patient data, with and without PCA, revealing potential wellness profiles that can guide personalized healthcare strategies.

## **Introduction**

The contemporary healthcare landscape increasingly emphasizes preventative care and personalized wellness programs. Understanding diverse patient wellness profiles is crucial for optimizing these initiatives. This study addresses the challenge of patient segmentation within a healthcare organization, leveraging a simulated dataset encompassing daily exercise time, healthy meals per day, sleep hours per night, stress level scores, and Body Mass Index (BMI). The primary objective is to group patients with similar wellness characteristics using clustering algorithms. Furthermore, the report explores the utility of dimensionality reduction through PCA in simplifying the dataset while preserving essential information for clustering. The comparison of clustering model performance, both pre- and post-PCA, will elucidate the most effective approach for discerning actionable patient segments.

## **Related Work**

Clustering techniques have been widely applied in healthcare for various purposes, including **patient stratification**, **disease subtyping**, and **healthcare resource optimization** (e.g., Liu et al., 2020; Zhang et al., 2017). **K-Means**, a popular partitioning method, is known for its simplicity and computational efficiency, making it suitable for large datasets when the number of clusters is known or can be estimated (Jain, 2010). **Hierarchical clustering**, on the other hand, builds a hierarchy of clusters, offering flexibility in choosing the number of clusters post-analysis and visualizing cluster relationships through dendrograms (Everitt et al., 2011).

The challenge of high-dimensional healthcare data often necessitates **dimensionality reduction techniques**. **PCA** is a well-established linear transformation method that projects data onto a lower-dimensional space while retaining the maximum variance, thereby reducing noise and computational complexity for subsequent analyses like clustering (Abdi & Williams, 2010). Studies have shown that combining PCA with clustering can lead to more robust and interpretable results by mitigating the "curse of dimensionality" (e.g., Ding & He, 2004). This study builds upon these foundational applications by directly comparing clustering performance with and without PCA on a simulated patient wellness dataset to provide specific insights for a healthcare organization's healthy living programs.

## **Methodology**

### **Data Description and Preprocessing**

The analysis utilized a simulated dataset (simulated\_health\_wellness\_data.csv) containing five key wellness indicators for 200 patients (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 1):

* Exercise\_Time\_Min: Daily exercise time in minutes.  
* Healthy\_Meals\_Per\_Day: Number of healthy meals consumed daily.  
* Sleep\_Hours\_Per\_Night: Hours of sleep per night.  
* Stress\_Level: A subjective stress level score.  
* BMI: Body Mass Index.

Initial exploratory data analysis (EDA) was performed using Python libraries such as pandas, numpy, matplotlib.pyplot, seaborn, and plotly.express (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 1).

* **Basic Information:** The df.info() and df.describe() functions were used to inspect data types, non-null counts, and descriptive statistics, confirming 200 entries and no missing values across all columns (Lab3\_Healthcare\_Clustering\_Sam\_v1.0.ipynb, Output 5).  
* **Missing Values:** A check for missing values using df.isnull().sum() confirmed no missing data points (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 1).  
* **Visualizations:**  
  * Pairplots were generated using seaborn.pairplot() to visualize relationships between all pairs of features, providing an initial understanding of data distributions and potential correlations (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 1).  
  * Histograms of numerical features were created using df.hist() to visualize their distributions (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 2).  
  * A correlation heatmap was generated with seaborn.heatmap() to identify linear relationships between numerical features (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 2).  
* **Data Standardization:** Given the varying scales and units of the features (e.g., minutes, hours, scores, BMI), StandardScaler from sklearn.preprocessing was applied to standardize the data. Standardization ensures that each feature contributes equally to the clustering process, preventing features with larger values from dominating the distance calculations (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 1). The standardized data was stored in scaled\_data.

### **Dimensionality Reduction (PCA)**

Principal Component Analysis (PCA) was implemented using sklearn.decomposition.PCA to reduce the dataset's dimensionality while retaining the most significant variance (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 1).

* **Determining Number of Components:** An elbow method using explained variance ratio was likely employed (though not explicitly shown in the provided PDF/notebook output for PCA, it's standard practice) to select the optimal number of principal components. The notebook then proceeds with a fixed number of components.  
* **PCA Application:** The PCA model was fitted and transformed on the scaled\_data, resulting in a lower-dimensional representation of the dataset.

### **Model Development (Clustering)**

Two primary clustering techniques were applied: K-Means and Hierarchical Clustering. Both were applied to the original scaled data and the PCA-transformed data for comparison.

#### **K-Means Clustering**

* **Algorithm:** K-Means clustering partitions n observations into k clusters, where each observation belongs to the cluster with the nearest mean (centroid) (MacQueen, 1967).  
* **Elbow Method for Optimal** k**:** The Elbow method was used to determine the optimal number of clusters (k) by plotting the Within-Cluster Sum of Squares (WCSS) against the number of clusters. The "elbow point" typically indicates the optimal k (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 4). The analysis suggests an optimal k=3 for both original and PCA-transformed data.  
* **Implementation:** KMeans from sklearn.cluster was used to perform clustering. The n\_init='auto' parameter was used for robust initialization (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 4).  
* **Clustering on Original Scaled Data:** K-Means was applied to scaled\_data with the determined optimal k. Cluster labels were assigned to each patient.  
* **Clustering on PCA-Transformed Data:** K-Means was also applied to the PCA-transformed data with the same optimal k.

#### **Hierarchical Clustering**

* **Algorithm:** Hierarchical clustering builds a hierarchy of clusters, starting with each data point as a single cluster and merging them iteratively (Agglomerative) or starting with one large cluster and splitting it (Divisive). This study likely used Agglomerative Clustering (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 1).  
* **Dendrogram for Optimal Clusters:** A dendrogram was generated using scipy.cluster.hierarchy.dendrogram and linkage to visualize the merging process and help determine the optimal number of clusters by observing the longest vertical line without a horizontal line crossing it (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 9). The dendrogram suggests 3 clusters as optimal for both original and PCA-transformed data.  
* **Implementation:** AgglomerativeClustering from sklearn.cluster was used for hierarchical clustering.  
* **Clustering on Original Scaled Data:** Agglomerative Clustering was applied to scaled\_data with the determined optimal number of clusters.  
* **Clustering on PCA-Transformed Data:** Agglomerative Clustering was also applied to the PCA-transformed data.

### **Evaluation Metrics**

The effectiveness of the clustering models was evaluated using the following metrics:

* **Silhouette Score:** Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). Scores range from \-1 to \+1, where \+1 indicates strong, dense clusters, 0 indicates overlapping clusters, and \-1 indicates incorrect clustering (Rousseeuw, 1987). This was computed using silhouette\_score from sklearn.metrics.  
* **Davies-Bouldin Index:** Measures the average similarity ratio of each cluster with its most similar cluster. Lower values indicate better clustering, with zero being the lowest possible value (Davies & Bouldin, 1979). This was computed using davies\_bouldin\_score from sklearn.metrics.  
* **Within-Cluster Sum of Squares (WCSS):** For K-Means, this metric measures the sum of squared distances between each point and the centroid of its assigned cluster. Lower WCSS values generally indicate more compact clusters. This was used in the Elbow Method (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 4).

Mean values for these metrics were computed for each method (K-Means, K-Means with PCA, Agglomerative, Agglomerative with PCA) and summarized in a table (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 14).

## **Results**

### **Exploratory Data Analysis (EDA)**

The initial EDA provided insights into the dataset's characteristics. The df.describe() output (Lab3\_Healthcare\_Clustering\_Sam\_v1.0.ipynb, Output 5\) showed the range and distribution of each wellness indicator. For instance, Exercise\_Time\_Min ranged from approximately 3.8 to 57.2 minutes, with a mean of 29.6 minutes. Healthy\_Meals\_Per\_Day varied from 0 to 9, averaging 2.875 meals. Sleep\_Hours\_Per\_Night ranged from 1.78 to 10.7 hours, with a mean of 6.93 hours. Stress\_Level scores ranged from 1 to 9, with a mean of 4.995, and BMI ranged from 12.5 to 37.9, averaging 25.15. The histograms revealed the distribution shapes for each feature, and the correlation heatmap highlighted any linear relationships, which could influence clustering outcomes.

### **K-Means Clustering Results**

The Elbow method applied to the scaled data suggested an optimal number of clusters, with the plot showing a distinct "elbow" at k=3. This indicates that increasing the number of clusters beyond three does not significantly reduce the WCSS, suggesting diminishing returns (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 4). K-Means was then performed with k=3 on both the original scaled data and the PCA-transformed data.

### **Hierarchical Clustering Results**

The dendrogram generated for hierarchical clustering also supported an optimal number of 3 clusters. The longest vertical line, indicating the largest distance between cluster merges, suggested cutting the dendrogram to yield three main clusters (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 9). Similar to K-Means, Agglomerative Clustering was performed with k=3 on both original scaled and PCA-transformed data.

### **Model Comparison**

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
* **WCSS:** The WCSS for KMeans\_PCA (124.5068) was significantly lower than for KMeans without `PCA (581.5567)`. This is expected as PCA reduces the dimensionality, leading to smaller distances within clusters in the transformed space.

### **Visualization of Clusters**

The notebook includes visualizations of the clusters. For K-Means, a 3D scatter plot of the clusters colored by kmeans.labels\_ on the PCA-transformed data helps visualize the separation of the patient segments (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 6). Similar visualizations are provided for Hierarchical Clustering (Lab3\_Healthcare\_Clustering\_Sam.pdf, p. 11). These plots are crucial for qualitatively assessing the distinctness of the identified patient groups.

## **Discussion**

The results demonstrate the successful application of K-Means and Hierarchical Clustering to segment patient wellness data, both with and without dimensionality reduction using PCA. The consistency in the optimal number of clusters (k=3) across different methods and data transformations (original scaled vs. PCA-transformed) suggests robust underlying patient groupings.

While K-Means on the original scaled data achieved a slightly higher Silhouette Score, indicating marginally better internal cohesion for individual clusters, the PCA-enhanced clustering methods consistently yielded lower Davies-Bouldin Index scores. A lower Davies-Bouldin Index signifies better clustering, characterized by clusters that are compact and well-separated from each other (Davies & Bouldin, 1979). This suggests that despite a minor trade-off in individual cluster cohesion (as indicated by the Silhouette Score), PCA contributed to more distinct and globally separated clusters, which can be highly beneficial for practical interventions. The reduced WCSS in KMeans\_PCA further supports the notion that dimensionality reduction helps create tighter clusters in the transformed space, enhancing computational efficiency and potentially simplifying interpretability of the clusters themselves.

### **Implications for Healthcare Organization:**
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

### **Limitations:**  
This study utilized a simulated dataset. While it provides a valuable framework for analysis, real-world patient data may exhibit more complexity, noise, and missing values, requiring more extensive preprocessing and potentially different modeling approaches. The interpretability of PCA components can also be challenging if they do not align clearly with known clinical concepts. Furthermore, the selection of the optimal number of clusters, while guided by the Elbow method and dendrograms, still involves a degree of subjectivity.

## **Conclusion**

This lab report successfully demonstrated the application of K-Means and Hierarchical Clustering, both with and without PCA, to segment a simulated patient wellness dataset. The findings suggest that clustering can effectively identify distinct patient profiles based on their health and wellness indicators. While K-Means on original data showed a slightly better Silhouette Score, PCA-enhanced clustering resulted in more well-separated clusters as indicated by the Davies-Bouldin Index. The optimal number of clusters was consistently identified as three across different methods, providing a robust basis for segmentation.

The insights gained from these clusters can empower the healthcare organization to develop and implement highly targeted and personalized healthy living programs. By understanding the unique needs and characteristics of each patient segment, interventions can be more effectively designed, leading to improved patient outcomes, optimized resource allocation, and a more proactive approach to wellness. Future research could explore the application of these methods to real-world datasets, incorporate additional patient demographic or clinical data, and investigate more advanced clustering algorithms or validation techniques.

## **References**

Abdi, H., & Williams, L. J. (2010). Principal component analysis. *Wiley Interdisciplinary Reviews: Computational Statistics*, *2*(4), 433-459.

Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *PAMI-1*(2), 224-227.

Ding, C., & He, X. (2004). K-means clustering and principal component analysis. In *Proceedings of the Twenty-first International Conference on Machine Learning* (p. 29). ACM.

Everitt, B. S., Landau, S., Leese, M., & Stahl, D. (2011). *Cluster analysis* (5th ed.). John Wiley & Sons.

Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern Recognition Letters*, *31*(8), 651-666.

Liu, S., Zhang, S., Li, Y., & Li, R. (2020). Patient clustering based on electronic health records using deep autoencoder. *BMC Medical Informatics and Decision Making*, *20*(1), 1-11.

MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics* (pp. 281-297). University of California Press.

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, *20*, 53-65.

Zhang, M., Hu, J., & Li, X. (2017). A review of data mining applications in healthcare. In *2017 3rd International Conference on Universal Village (UV)* (pp. 1-6). IEEE.
