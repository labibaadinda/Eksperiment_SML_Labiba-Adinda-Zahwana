import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_sleep_data(df):
    """
    Melakukan preprocessing pada dataset sleep-health_life-style.
    Argumen:
        df (DataFrame): Dataset awal yang sudah dibaca
    Return:
        DataFrame: Dataset yang sudah diproses
    """

    # Step 1: Data Quality Check
    print("Descriptive Statistics:\n", df.describe())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicated Rows:\n", df.duplicated().sum())

    # Step 2: Handle missing values in 'Sleep Disorder'
    df['Sleep Disorder'] = df['Sleep Disorder'].replace(np.nan, 'Healthy')

    # Step 3: Split 'Blood Pressure' into 'Systolic BP' and 'Diastolic BP'
    df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)
    df[['Systolic BP', 'Diastolic BP']] = df[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)
    df = df.drop('Blood Pressure', axis=1)

    # Step 4: Outlier Removal (IQR Method)
    num_col = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
               'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic BP', 'Diastolic BP']

    Q1 = df[num_col].quantile(0.25)
    Q3 = df[num_col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[num_col] < (Q1 - 1.5 * IQR)) | (df[num_col] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Step 5: Encode categorical features
    le = LabelEncoder()
    for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
        df[col] = le.fit_transform(df[col])

    return df

def plot_eda(df):
    """
    Menampilkan visualisasi EDA dasar dari dataset.
    """
    # Gender Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Gender', palette='pastel', data=df)
    plt.title('Distribution of Gender')
    plt.show()

    # BMI Category Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='BMI Category', palette='pastel', data=df)
    plt.title('Distribution of BMI Category')
    plt.show()

    # Correlation Matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr().round(2)
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Distributions of Numerical Columns
    plt.figure(figsize=(15, 18))
    for i, col in enumerate(['Age', 'Sleep Duration', 'Heart Rate']):
        plt.subplot(3, 1, i + 1)
        sns.histplot(df[col], kde=False, color='pink')
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
