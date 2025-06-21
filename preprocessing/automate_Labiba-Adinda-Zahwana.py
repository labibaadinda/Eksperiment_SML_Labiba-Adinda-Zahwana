import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    # Step 1: Data Quality Check
    
    
    print("Descriptive Stats:\n", df.describe())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())

    # Step 2: Replace NaN in Sleep Disorder with 'Healthy'
    df['Sleep Disorder'] = df['Sleep Disorder'].replace(np.nan, 'Healthy')

    # Step 3: Split 'Blood Pressure' into 'Systolic BP' and 'Diastolic BP'
    df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)
    df[['Systolic BP', 'Diastolic BP']] = df[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)
    df.drop('Blood Pressure', axis=1, inplace=True)

    # Step 4: Remove Outliers (IQR Method)
    num_col = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
               'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic BP', 'Diastolic BP']
    
    Q1 = df[num_col].quantile(0.25)
    Q3 = df[num_col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[num_col] < (Q1 - 1.5 * IQR)) | (df[num_col] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Step 5: Label Encoding for Categoricals
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df


def visualize_distribution(df):
    # Gender Countplot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Gender', palette='pastel', data=df)
    plt.title('Distribution of Gender')
    plt.show()

    # BMI Category Countplot
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

    # Histograms for some numeric columns
    plt.figure(figsize=(15, 18))
    sns.histplot(df['Age'], kde=False, color='pink')
    sns.histplot(df['Sleep Duration'], kde=False, color='pink')
    sns.histplot(df['Heart Rate'], kde=False, color='pink')
    plt.tight_layout()
    plt.show()


# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    # Load raw data
    raw_df = pd.read_csv('data/sleep-health_life-style.csv')

    # Visualize raw (optional)
    visualize_distribution(raw_df)

    # Preprocess data
    processed_df = preprocess_data(raw_df)

    # Save cleaned data
    processed_df.to_csv('processed_data.csv', index=False)
    print("✅ Processed data saved to 'processed_data.csv'")
