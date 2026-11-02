# Bubbles
This project is a tool that uses unsupervised machine learning to automatically cluster thousands of O*NET occupations by semantic similarity, then presents them as a zoomable, nested bubble interface so someone can start broad from a category like "I want to help people" and drill down.


For Project 1: Clustering text-based task statement (NLP approach)

Extract and embed: Vectorize the text data into dense, fixed-length vectors using a transformer model optimized for semantic text similarity
Dimension Reduction: High-dimensional embeddings can degrade distance metrics (the curse of dimensionality). Apply UMAP (Uniform Manifold Approximation and Projection) or PCA to reduce vectors to a manageable scale while preserving local and global structures.
Density-Based Clustering: Use HDBSCAN(research it) and K-Means and compare. 

For Project 2: Clustering by Numeric Attributes (Data Science Approach)

Pivot and Matrix Construction: Create a feature matrix where each row represents an O*NET-SOC code and each column represents a specific attribute score (e.g., Level of Active Listening, Importance of Mathematics)
Feature Standardization: Scale the numeric data using StandardScaler to handle any variance differences across distinct descriptor types
Partitioning Optimization: Since the numeric feature space is dense and continuous, K-Means++ or Agglomerative Hierarchical Clustering works remarkably well. Use the Silhouette Method or Elbow Method (Inertia) to automatically determine the mathematical optimal number of clusters (K)

O*NET data contains highly niche roles. Rely on density algorithms like HDBSCAN or set a strict silhouette distance threshold to capture unclassifiable job elements as independent vectors. Use TF-IDF on the text fields within each finalized cluster group to automatically pull out the top 5 distinguishing words. This provides an instant label or name for your newly generated job cluster.

