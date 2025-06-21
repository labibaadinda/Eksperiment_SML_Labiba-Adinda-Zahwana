import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load Data
df = pd.read_csv('data/sleep-health_life-style.csv')

# Step 2: Data Quality Check
print(df.describe())  # Descriptive statistics
print(df.dtypes)  # Data types of each column
print(df.isnull().sum())  # Checking for missing values
print(df.duplicated().sum())  # Checking for duplicates


# Step 3: Exploratory Data Analysis
# Example: Gender Analysis
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', palette='pastel', data=df)
plt.title('Distribution of Gender')
plt.show()

# Example: BMI Category Analysis
plt.figure(figsize=(8, 6))
sns.countplot(x='BMI Category', palette='pastel', data=df)
plt.title('Distribution of BMI Category')
plt.show()

# Step 4: NaN means have no disease
# Replace NaN values dengan 'Healthy' di kolom 'Sleep Disorder'
df['Sleep Disorder'] = df['Sleep Disorder'].replace(np.nan, 'Healthy')

# Display the updated DataFrame
df.head()

# Step 5: Split 'Blood Pressure' into 'Systolic BP' and 'Diastolic BP'
df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)

# Convert to numeric
df[['Systolic BP', 'Diastolic BP']] = df[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)

# Drop the original 'Blood Pressure' column
df = df.drop('Blood Pressure', axis=1)


# Step 6: Outlier Detection and Treatment (e.g., using IQR)

num_col = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
           'Heart Rate', 'Daily Steps', 'Systolic BP', 'Diastolic BP']

Q1 = df[num_col].quantile(0.25)  # First quartile (25th percentile)
Q3 = df[num_col].quantile(0.75)  # Third quartile (75th percentile)
IQR = Q3 - Q1  # Interquartile Range

# Remove rows where any value in the numerical columns is an outlier
df = df[~((df[num_col] < (Q1 - 1.5 * IQR)) | (df[num_col] > (Q3 + 1.5 * IQR))).any(axis=1)]


# Step 7: Handling Categorical Variables (e.g., label encoding or one-hot encoding)
# Example for Label Encoding (you can choose based on your model requirements)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['BMI Category'] = label_encoder.fit_transform(df['BMI Category'])
df['Sleep Disorder'] = label_encoder.fit_transform(df['Sleep Disorder'])
df.head()


# Step 8: Final Data Check
print(df.head())  # Display the first few rows of the processed data

# Step 10: Save Processed Data to a new file
df.to_csv('processed_data.csv', index=False)

# Optional: Visualizing correlation matrix for numerical data
corr_matrix = df.select_dtypes(include=[np.number]).corr().round(2)
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Optional: Visualizing distributions of numerical columns
plt.figure(figsize=(15, 18))
sns.histplot(df['Age'], kde=False, color='pink')
sns.histplot(df['Sleep Duration'], kde=False, color='pink')
sns.histplot(df['Heart Rate'], kde=False, color='pink')
plt.tight_layout()
plt.show()
