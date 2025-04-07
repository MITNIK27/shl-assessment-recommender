import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the enriched data
df = pd.read_csv("data/enriched_assessments.csv")

# --- BASIC INFO ---
print("\n🔍 Basic Data Info:")
print(df.info())

# --- NULL CHECKS ---
print("\n❓ Null Values Summary:")
print(df.isnull().sum())

# --- UNIQUE VALUES IN KEY COLUMNS ---
key_columns = ['test_type', 'remote_testing_support', 'adaptive/irt_support']
for col in key_columns:
    print(f"\n📌 Unique values in '{col}':")
    print(df[col].value_counts(dropna=False))

# --- DESCRIPTION LENGTH CHECK ---
df['description_length'] = df['description'].astype(str).apply(len)
print("\n✏️ Description Length Stats:")
print(df['description_length'].describe())

# --- PLOT 1: Test Type Distribution ---
plt.figure(figsize=(10, 4))
sns.countplot(y='test_type', data=df, order=df['test_type'].value_counts().index)
plt.title("Test Type Distribution")
plt.xlabel("Count")
plt.ylabel("Test Type")
plt.tight_layout()
plt.show()

# --- PLOT 2: Remote Testing Support ---
plt.figure(figsize=(6, 3))
sns.countplot(x='remote_testing_support', data=df)
plt.title("Remote Testing Support Availability")
plt.xlabel("Remote Testing Support")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- PLOT 3: Adaptive/IRT Support ---
plt.figure(figsize=(6, 3))
sns.countplot(x='adaptive/irt_support', data=df)
plt.title("Adaptive/IRT Support Availability")
plt.xlabel("Adaptive/IRT Support")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- PLOT 4: Description Length Distribution ---
plt.figure(figsize=(8, 4))
sns.histplot(df['description_length'], bins=30, kde=True)
plt.title("Distribution of Description Lengths")
plt.xlabel("Description Length (chars)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
