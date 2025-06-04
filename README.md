# Sentiment Analysis of Social Media Posts Using Structured Metadata

This project applies machine learning techniques to predict sentiment (Positive, Neutral, Negative) from structured metadata of social media posts collected from platforms like Twitter, Instagram, and LinkedIn. Instead of analyzing raw text, this approach relies on engagement metrics and contextual data, offering privacy-preserving sentiment insights.

## Dataset Overview

- Total Samples: 500 posts  
- Total Features: 17 structured attributes  
- Platforms: Twitter, Instagram, LinkedIn  
- Target Variable: Sentiment (Positive, Neutral, Negative), with an additional Sentiment Score  

Key Metadata Features:
- Engagement: Likes, Shares, Comments  
- Contextual: Platform, Post Type (text/image/video), Country  
- Temporal: Time of Day (morning, afternoon, evening, night)  
- User Info: User Type (Brand, Influencer)  

## Tech Stack and Libraries

- Languages: Python  
- Libraries:  
  - Data Handling: pandas, numpy  
  - Visualization: matplotlib, seaborn  
  - Preprocessing: sklearn.preprocessing (LabelEncoder, OneHotEncoder, StandardScaler)  
  - Modeling: sklearn.ensemble.RandomForestClassifier, sklearn.svm.SVC, sklearn.linear_model.LogisticRegression  
  - Evaluation: sklearn.metrics (confusion_matrix, classification_report, roc_auc_score)  
  - Unsupervised Learning: sklearn.cluster.KMeans, sklearn.decomposition.PCA  
  - Class Balancing: imblearn.over_sampling.SMOTE  
  - Hyperparameter Tuning: sklearn.model_selection.GridSearchCV  
  - Association Rule Mining: mlxtend.frequent_patterns.apriori, mlxtend.frequent_patterns.association_rules  

## Machine Learning Techniques Applied

Supervised Learning:
- Random Forest (best performance, 100% accuracy)
- Support Vector Machine (SVM)
- Logistic Regression

Unsupervised Learning:
- KMeans Clustering
  - Used to explore natural groupings of sentiment
  - Evaluated with Silhouette Score (0.15)

Association Rule Mining:
- Apriori Algorithm
  - Discovered co-occurrence patterns (e.g., {Brand} → {Link})

## Evaluation Metrics

- Accuracy: 100% for Random Forest and SVM  
- F1-Score: 1.0  
- ROC AUC: High across all sentiment classes  
- Silhouette Score: 0.15 (KMeans)  

## Visualizations

- Confusion Matrix (for RF and SVM)  
- Feature Importance (Random Forest)  
- ROC Curve (One-vs-Rest strategy)  
- PCA-based Cluster Visualization  
- Association Rules Table with Support, Confidence, and Lift  

## Project Workflow

1. Data cleaning and preprocessing  
2. Feature encoding and scaling  
3. Supervised model training and tuning  
4. Unsupervised clustering and rule mining  
5. Performance evaluation and visualization  

## Challenges

- No textual data: Limited nuance in sentiment detection  
- Class imbalance: Resolved using SMOTE  
- Weak clustering: Structured metadata lacks strong separability for unsupervised learning  

## Future Work

- Integrate textual content for NLP-based sentiment analysis (e.g., using TF-IDF, BERT)  
- Build hybrid models combining metadata and raw text  
- Explore deep learning (LSTM, transformers) for sequential context  
