# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
df = pd.read_csv("data/train.csv")
print("Dataset Loaded Successfully")
print(df.head())

# Step 2: Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Step 3: Fill missing Age values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column (too many missing)
df.drop('Cabin', axis=1, inplace=True)

# Step 4: Basic statistics
print("\nStatistical Summary:\n", df.describe())

# Step 5: Categorical analysis
print("\nSurvival Counts:\n", df['Survived'].value_counts())
print("\nPclass Counts:\n", df['Pclass'].value_counts())
print("\nGender Counts:\n", df['Sex'].value_counts())

# Step 6: Visualizations

# 6a: Survival count plot
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.show()

# 6b: Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title("Survival by Gender")
plt.show()

# 6c: Survival by Pclass
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Survived', hue='Pclass')
plt.title("Survival by Passenger Class")
plt.show()

# 6d: Age distribution
plt.figure(figsize=(8,4))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# 6e: Correlation heatmap
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Columns)")
plt.show()
