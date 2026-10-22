# Implementation of Clustering Algorithms

## Problem Statement
Implement clustering algorithms on a customer dataset to identify meaningful groups based on annual income and spending score. Evaluate the clustering results using Silhouette Score and Davies-Bouldin Index.

## Algorithms Used
- K-Means Clustering
- Hierarchical Clustering

## Dataset
Mall Customers dataset containing customer annual income and spending score.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

## How to Run
1. Install Python.
2. Install the required libraries:
   `pip install -r requirements.txt`
3. Run the program:
   `python main.py`

## Evaluation Metrics
- **Silhouette Score:** Higher values generally indicate better-separated clusters.
- **Davies-Bouldin Index:** Lower values generally indicate better-defined clusters.

## Output
The program generates two cluster visualization graphs and a CSV file containing the evaluation metrics.

## Conclusion
K-Means and Hierarchical Clustering are used to group customers based on their annual income and spending score. Their clustering quality is compared using evaluation metrics and visualizations.