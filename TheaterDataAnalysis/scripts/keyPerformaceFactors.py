import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ========== USER CONFIGURATION ========== #
input_file = "/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/scripts/data/cleaned/cleaned_theater_data3.xlsx"  # Replace with the actual path
output_dir = "/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/scripts/data/cleaned/"
os.makedirs(output_dir, exist_ok=True)

## Load the Excel file
df = pd.read_excel(input_file, dtype=str)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Identify numeric columns
numeric_columns = [
    "..investeeringud, tuhat eurot",
    "..tulud piletimüügist, tuhat eurot",
    "Arvestuslikud inimtööaastad",
    "Halduskulud, tuhat eurot",
    "Costs and investments (thousands of euros)",
    "Muud kulud, tuhat eurot",
    "Muud toetused, tuhat eurot",
    "Omatulud, tuhat eurot",
    "Tulud omavalitsuse eelarvest, tuhat eurot",
    "Tulud riigieelarvest, tuhat eurot",
    "Revenue (thousands of euros)",
    "Tööjõukulud, tuhat eurot"
]


# Function to clean numeric values
def clean_numeric(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val == "..":
        return np.nan
    val = val.replace(" ", "").replace(",", ".")
    try:
        return float(val)
    except ValueError:
        return np.nan


for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# Convert 'Aasta' to numeric if it exists
if 'Aasta' in df.columns:
    df['Aasta'] = df['Aasta'].apply(clean_numeric)

# Drop rows with no year or no revenue data
df = df.dropna(subset=['Aasta', 'Revenue (thousands of euros)'])

print("Data after cleaning:")
print(df.head())

# Compute correlation matrix for numeric columns
num_df = df.select_dtypes(include=[np.number])
corr_matrix = num_df.corr()

print("Correlation Matrix:")
print(corr_matrix)

# Save correlation matrix
corr_path = os.path.join(output_dir, "correlation_matrix.csv")
corr_matrix.to_csv(corr_path, sep=";")

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

# Regression: OLS model (Statsmodels)
X_cols = ["Tööjõukulud, tuhat eurot", "Omatulud, tuhat eurot"]
reg_df = df.dropna(subset=X_cols + ["Revenue (thousands of euros)"])

X = reg_df[X_cols]
Y = reg_df["Revenue (thousands of euros)"]

# Add a constant for OLS
X_ols = sm.add_constant(X)
model = sm.OLS(Y, X_ols).fit()
print("OLS Regression Results:")
print(model.summary())

# Save regression results
regression_summary_path = os.path.join(output_dir, "regression_summary.txt")
with open(regression_summary_path, "w") as f:
    f.write(model.summary().as_text())

# Check residuals of OLS model
residuals = model.resid
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax[0])
ax[0].set_title("Residuals Distribution")
sns.scatterplot(x=model.fittedvalues, y=residuals, ax=ax[1])
ax[1].axhline(0, color='red', linestyle='--')
ax[1].set_title("Residuals vs Fitted")
residual_plot_path = os.path.join(output_dir, "residual_plots.png")
plt.savefig(residual_plot_path)
plt.close()

# Simple Predictive Model with Scikit-Learn
# Using the same variables as OLS for a quick comparison
if len(reg_df) > 5:
    # Only proceed if we have enough data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Scikit-Learn Linear Regression:")
    print(f"R2 on test set: {r2:.3f}")
    print(f"RMSE on test set: {rmse:.3f}")

    # Save predictions and actuals
    preds_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    preds_path = os.path.join(output_dir, "predictions.csv")
    preds_df.to_csv(preds_path, sep=";", index=False)

    # Plot predicted vs actual
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Revenue")
    plt.ylabel("Predicted Revenue")
    plt.title("Actual vs Predicted Revenue")
    prediction_plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(prediction_plot_path)
    plt.close()
else:
    print("Not enough data points to perform train-test split for the predictive model.")

print("Analysis complete. Outputs saved in:", output_dir)
