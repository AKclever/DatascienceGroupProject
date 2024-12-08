import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set your file paths
input_excel_path = "/TheaterDataAnalysis/scripts/data/cleaned/cleaned_theater_data3.xlsx"

# Read the data from Excel
df = pd.read_excel(input_excel_path)

# The dataset has rows for each year and category, with some rows having empty year cells.
# Let's forward-fill the 'Aasta' column so that all rows have a year.
df['Aasta'] = df['Aasta'].ffill()

# Convert 'Aasta' to integer if it's numeric
df['Aasta'] = pd.to_numeric(df['Aasta'], errors='coerce', downcast='integer')

# We might also have categories repeated with empty lines. Forward-fill 'Kategooria' as well.
df['Kategooria'] = df['Kategooria'].ffill()

# Now let's clean numeric columns. They use a comma as decimal and space as thousands separator.
# Define a helper function to convert these strings to floats:
def convert_to_float(x):
    if pd.isna(x):
        return None
    if x == '..':
        return None
    # Remove spaces
    x = str(x).replace(" ", "")
    # Replace commas with dots
    x = x.replace(",", ".")
    try:
        return float(x)
    except:
        return None

# List the columns we want to analyze
numeric_columns = [
    "..tulud piletimüügist, tuhat eurot",
    "Tulud riigieelarvest, tuhat eurot",
    "Revenue (thousands of euros)"
]

# Convert these columns to float
for col in numeric_columns:
    df[col] = df[col].apply(convert_to_float)

# Filter to only include "Teatrid kokku" (Theaters total) if you want the aggregate trend
df_total = df[df['Kategooria'] == "Teatrid kokku"].copy()

# Group by year if needed. Each year should have one row for "Teatrid kokku" based on your structure.
# If there's only one row per year for "Teatrid kokku", grouping might not be necessary.
# But let's ensure by grouping and summing if there's any duplication.
df_total_yearly = df_total.groupby('Aasta', as_index=False).agg({
    "..tulud piletimüügist, tuhat eurot": "sum",
    "Tulud riigieelarvest, tuhat eurot": "sum",
    "Revenue (thousands of euros)": "sum"
})

# Check data
print("Data Summary:")
print(df_total_yearly.describe())

# Set a style
sns.set_style("whitegrid")

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Ticket revenue trend
sns.lineplot(data=df_total_yearly, x='Aasta', y='..tulud piletimüügist, tuhat eurot', ax=axes[0], marker='o')
axes[0].set_title("Annual Ticket Revenue Over Time (Teatrid kokku)")
axes[0].set_ylabel("Ticket Revenue (k€)")

# State subsidies trend
sns.lineplot(data=df_total_yearly, x='Aasta', y='Tulud riigieelarvest, tuhat eurot', ax=axes[1], marker='o')
axes[1].set_title("Annual State Subsidies Over Time (Teatrid kokku)")
axes[1].set_ylabel("State Subsidies (k€)")

# Total revenue trend
sns.lineplot(data=df_total_yearly, x='Aasta', y='Revenue (thousands of euros)', ax=axes[2], marker='o')
axes[2].set_title("Annual Total Revenue Over Time (Teatrid kokku)")
axes[2].set_ylabel("Total Revenue (k€)")
axes[2].set_xlabel("Year")

plt.tight_layout()
plt.show()

# Optionally, save figures
output_dir = "/path/to/output/figures"
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "theater_trends_over_time.png"), dpi=300)

# Additional analyses (optional):
df_total_yearly['TicketRevenue_pct_change'] = df_total_yearly['..tulud piletimüügist, tuhat eurot'].pct_change() * 100
df_total_yearly['StateSubs_pct_change'] = df_total_yearly['Tulud riigieelarvest, tuhat eurot'].pct_change() * 100
df_total_yearly['TotalRevenue_pct_change'] = df_total_yearly['Revenue (thousands of euros)'].pct_change() * 100

print("\nPercentage changes year-over-year:")
print(df_total_yearly[['Aasta', 'TicketRevenue_pct_change', 'StateSubs_pct_change', 'TotalRevenue_pct_change']])
