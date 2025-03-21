import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data/boston_housing_data.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())
print("\n")

# Show descriptive statistics
print("Descriptive Statistics:")
print(data.describe())
print("\n")

# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Distribution of the target variable
plt.figure(figsize=(8, 5))
sns.histplot(data['MEDV'], bins=30, kde=True)
plt.title('Distribution of Median Value of Homes (MEDV)')
plt.xlabel('Median Value (in $1000s)')
plt.ylabel('Frequency')
plt.show()

# Scatter plots to visualize relationships between 'RM' (average number of rooms) and 'MEDV'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RM', y='MEDV', data=data)
plt.title('Relationship between Average Number of Rooms and Median Value of Homes')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Median Value (in $1000s)')
plt.show()

# Boxplot for 'MEDV' based on 'CHAS' (Charles River dummy variable)
plt.figure(figsize=(8, 5))
sns.boxplot(x='CHAS', y='MEDV', data=data)
plt.title('Median Value of Homes Based on Proximity to Charles River')
plt.xlabel('Proximity to Charles River (1=Yes, 0=No)')
plt.ylabel('Median Value (in $1000s)')
plt.show()
