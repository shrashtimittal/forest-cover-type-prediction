import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1️⃣ Load the dataset
df = pd.read_csv(r"data\train.csv")

# drop the ID column (not needed for modelling)
df = df.drop(columns=['Id'])

# identify column groups
numeric_cols = [
    'Elevation','Aspect','Slope',
    'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]
wilderness_cols = [c for c in df.columns if c.startswith('Wilderness')]
soil_cols = [c for c in df.columns if c.startswith('Soil')]
target_col = 'Cover_Type'

# 2️⃣ Target distribution
plt.figure(figsize=(7,4))
df[target_col].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.xlabel("Cover Type")
plt.ylabel("Count")
plt.title("Cover Type Distribution")
plt.tight_layout()
plt.show()

# 3️⃣ Numeric histograms
df[numeric_cols].hist(bins=30, figsize=(12,10))
plt.suptitle('Numeric Feature Distributions', y=0.92)
plt.tight_layout()
plt.show()

# 4️⃣ Correlation heatmap
corr = df[numeric_cols + [target_col]].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix (numeric + target)")
plt.tight_layout()
plt.show()

# 5️⃣ Wilderness & soil sparsity
print("\nWilderness column non-zero ratio:")
print((df[wilderness_cols].sum()/len(df)).sort_values())

print("\nSoil column non-zero ratio (showing 10 rarest):")
print((df[soil_cols].sum()/len(df)).sort_values().head(10))

# 6️⃣ Example derived feature
df['Distance_To_Hydrology'] = np.sqrt(
    df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2
)
print("\nDerived Distance_To_Hydrology summary:\n", df['Distance_To_Hydrology'].describe())
