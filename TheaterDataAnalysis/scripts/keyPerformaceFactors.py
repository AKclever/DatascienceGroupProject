import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# Step 1: Load the Data
# --------------------------
input_file = "/TheaterDataAnalysis/scripts/data/cleaned/cleaned_theater_data2.csv"  # Update path
df = pd.read_csv(input_file)

# Inspect the data
print("Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())

# --------------------------
# Step 2: Data Cleaning and Numeric Conversion
# --------------------------
# Columns like "Vaatajad, tuhat" might need to be converted from strings like "35,60" to floats.
# We'll write a small function to convert comma decimal to float.
def to_float(x):
    if pd.isna(x) or x == '..':
        return np.nan
    return float(str(x).replace(",", ".").replace(" ", ""))

numeric_cols = [
    "Lavastused",
    "..uuslavastused",
    "Etendused",
    "..külalisetendused",
    "Vaatajad, tuhat",
    "Teatriskäigud 1000 elaniku kohta",
    "Keskmine piletihind, eurot"
]

# Convert these columns to floats
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(to_float)

# Now we have numeric columns that we can use for correlation.

# --------------------------
# Step 3: Select Outcome and Predictors
# --------------------------
# Let's assume we want to understand which factors drive attendance ("Vaatajad, tuhat").
# Predictors might be the number of performances (Etendused), number of new performances (..uuslavastused),
# average ticket price (Keskmine piletihind, eurot), etc.

outcome_col = "Vaatajad, tuhat"
predictor_cols = [
    "Lavastused",
    "..uuslavastused",
    "Etendused",
    "..külalisetendused",
    "Teatriskäigud 1000 elaniku kohta",
    "Keskmine piletihind, eurot"
]

# Filter rows with non-missing data for the chosen columns
df_filtered = df.dropna(subset=[outcome_col] + predictor_cols)

print("\nFiltered Data Shape:", df_filtered.shape)

# --------------------------
# Step 4: Correlation Analysis
# --------------------------
corr_matrix = df_filtered[[outcome_col] + predictor_cols].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Attendance and Potential Drivers")
plt.tight_layout()
plt.show()

# --------------------------
# Step 5: Simple Regression Model (Optional)
# --------------------------
X = df_filtered[predictor_cols]
y = df_filtered[outcome_col]

# Some models benefit from scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)
print("\nLinear Regression Model Performance:")
print(f"R² (Train): {r2_train:.2f}")
print(f"R² (Test):  {r2_test:.2f}")

coef_df = pd.DataFrame({
    "Predictor": predictor_cols,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)

print("\nModel Coefficients:")
print(coef_df)

# --------------------------
# Step 6: Interpretation
# --------------------------
# The correlation matrix and model coefficients will help you see which variables have the strongest relationship
# with attendance. For example, if "Etendused" (number of performances) and "Keskmine piletihind, eurot" show strong
# correlation and large regression coefficients, they might be key drivers.

# Report findings:
# "The number of performances (Etendused) and the average ticket price (Keskmine piletihind) show a strong correlation
# with attendance, suggesting these are key performance drivers."


