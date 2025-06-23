import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


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

import os

if __name__ == "__main__":
    input_file = os.path.join("..", "sleep-health_life-style_raw.csv")  # file input di root project
    output_file = os.path.join(".", "sleep-health_life-style_preprocessing.csv")  # simpan di dalam folder 'preprocessing'

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    df = pd.read_csv(input_file)
    df = preprocess_sleep_data(df)
    df.to_csv(output_file, index=False)

    print("Preprocessing complete.")
