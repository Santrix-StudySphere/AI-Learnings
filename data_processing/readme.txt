#Code related to data pre processing in python will be written in this folder

##Importing Pandas liabrary:
 >pip install pandas

 ## Install sci-kit learn
 >pip install scikit-learn

 ## Data Normalization:
#🔹 Definition:
#Normalization is the process of scaling numerical data into a standard range (often between 0 and 1) so that all features contribute equally to a model.
#It’s used in machine learning to prevent features with large ranges (like salary in lakhs vs. age in years) from dominating over smaller-ranged features.
### Why Normalization is needed?

## Different Scales Issue:
# Example: Age (20–60) vs. Salary (20,000–2,00,000). Without normalization, models like KNN, Neural Networks, or Gradient Descent–based algorithms would give more importance to Salary since its values are larger.
## Improves Convergence:
# In optimization algorithms (like gradient descent), normalized data helps the algorithm converge faster.
## Improves Accuracy:
# Many models (e.g., distance-based algorithms like KNN, SVM, Clustering) rely on distance. Normalization ensures fair distance calculations.

Here are the common normalization methods we use:
1. Min-Max Normalization (Rescaling):
    Formula:
    X′=(​X−Xmin​​)/(Xmax​−Xmin)
# Scales values to a fixed range, usually [0,1].

Example: Age data (18–60) → scaled to [0–1].

2. Z-Score Normalization (Standardization):
    Formula:
    z = (x - u)/(sd)
