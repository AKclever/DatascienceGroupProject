import pandas as pd
import numpy as np
import os
import ast

# Input and output paths
raw_csv_path = "/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/scripts/data/raw/theater_data.csv"
output_dir = "/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/data/processed"
os.makedirs(output_dir, exist_ok=True)

output_csv_path = os.path.join(output_dir, 'cleaned_theater_data1.csv')
output_excel_path = os.path.join(output_dir, 'cleaned_theater_data1.xlsx')

# Step 1: Read the raw CSV file
df_raw = pd.read_csv(raw_csv_path)

# Step 2: Extract 'id', 'size', and 'value'
ids = ast.literal_eval(df_raw.loc[0, 'id'])
size = ast.literal_eval(df_raw.loc[0, 'size'])
values = ast.literal_eval(df_raw.loc[0, 'value'])

# Convert 'value' into an array and reshape
values_array = np.array(values, dtype=object).reshape(size)

# Helper function to get dimension labels
def get_dimension_labels(df, dimension_name, dimension_size):
    index_to_label = {}
    # We look for columns like dimension.<DimensionName>.category.index.X and dimension.<DimensionName>.category.label.X
    prefix_index = f"dimension.{dimension_name}.category.index."
    prefix_label = f"dimension.{dimension_name}.category.label."
    # Extract all indices for this dimension
    index_cols = [c for c in df.columns if c.startswith(prefix_index)]
    for col in index_cols:
        idx_str = col.split('.')[-1]  # the last part should be the numeric index
        index_val = int(df.loc[0, col])
        label_col = prefix_label + idx_str
        label_val = df.loc[0, label_col]
        index_to_label[index_val] = label_val
    # Sort by index to ensure correct order
    labels = [index_to_label[i] for i in range(dimension_size)]
    return labels

# Get labels for each dimension
year_labels = get_dimension_labels(df_raw, 'Aasta', size[0])
category_labels = get_dimension_labels(df_raw, 'Kategooria', size[1])
indicator_labels = get_dimension_labels(df_raw, 'Näitaja', size[2])

# Create MultiIndex and DataFrame
index = pd.MultiIndex.from_product([year_labels, category_labels, indicator_labels], names=ids)
data_flat = values_array.flatten()
df_final = pd.DataFrame(data=data_flat, index=index, columns=['value']).reset_index()

# Clean and convert 'value' column
df_final['value'] = df_final['value'].replace({None: np.nan, 'None': np.nan})
df_final['value'] = pd.to_numeric(df_final['value'], errors='coerce')

# Pivot the DataFrame
df_pivot = df_final.pivot_table(
    index=['Aasta', 'Kategooria'],
    columns='Näitaja',
    values='value',
    aggfunc='first'
).reset_index()

# Flatten columns if needed
df_pivot.columns.name = None

# Rename columns to desired English names
df_pivot.rename(columns={
    'Aasta': 'Year',
    'Kategooria': 'Category',
    'Teatrite arv': 'Teatrite arv',
    'Lavastused': 'Lavastused',
    '..uuslavastused': '..uuslavastused',
    'Etendused': 'Etendused',
    'Vaatajad, tuhat': 'Vaatajad, tuhat',
    'Teatriskäigud 1000 elaniku kohta': 'Teatriskäigud 1000 elaniku kohta',
    'Saalid': 'Saalid',
    'Istekohad': 'Istekohad'
}, inplace=True)

# Reorder columns as desired
desired_columns = [
    'Year', 'Category', 'Teatrite arv', 'Lavastused', '..uuslavastused',
     'Etendused', 'Vaatajad, tuhat',
    'Teatriskäigud 1000 elaniku kohta', 'Saalid', 'Istekohad'
]
df_pivot = df_pivot[desired_columns]

# Replace NaN with '..'
df_pivot.fillna('..', inplace=True)

# Format numbers
numeric_columns = [
    'Teatrite arv', 'Lavastused', '..uuslavastused',
     'Etendused', 'Vaatajad, tuhat',
    'Teatriskäigud 1000 elaniku kohta', 'Saalid', 'Istekohad'
]

def format_value(x):
    # If '..' then just return it
    if x == '..':
        return x
    # Try to convert to float
    try:
        val = float(str(x).replace(',', '.'))  # ensure conversion
        # Format with space as thousands separator
        formatted = f"{val:,.1f}".replace(",", " ")  # starts by replacing comma with space for thousands
        # Now we have something like "1 023.1", we need comma as decimal separator
        formatted = formatted.replace('.', ',')
        # If it was actually an integer (like number of theaters) we might want no decimals.
        # Let's check if the value is actually integer-like:
        if val.is_integer():
            # format as integer with space as thousands separator
            formatted_int = f"{int(val):,}".replace(",", " ")
            return formatted_int
        return formatted
    except:
        return x

for col in numeric_columns:
    df_pivot[col] = df_pivot[col].apply(format_value)

# Sort by year and category
df_pivot.sort_values(by=['Year', 'Category'], inplace=True, key=lambda x: x.where(x != '', np.nan))
df_pivot.reset_index(drop=True, inplace=True)

# Remove repeated year labels, leaving them blank for subsequent rows within the same year
df_pivot['Year'] = df_pivot['Year'].where(df_pivot['Year'].ne(df_pivot['Year'].shift()), '')

# Ensure the first occurrence of each year is visible
# Sort operation above should retain order, but if needed:
for year in df_pivot['Year'].unique():
    if year != '':
        indices = df_pivot.index[df_pivot['Year'] == year].tolist()
        if indices:
            # first occurrence should show the year, others blank
            for i, idx in enumerate(indices):
                if i == 0:
                    df_pivot.at[idx, 'Year'] = year
                else:
                    df_pivot.at[idx, 'Year'] = ''

# Print final result
print(df_pivot.to_string(index=False))

# Save to CSV with European formatting (semicolon as sep, comma as decimal)
# Using float_format='%.1f' doesn't apply since we pre-formatted as strings. We just save as is.
df_pivot.to_csv(
    output_csv_path,
    index=False,
    encoding='utf-8',
    sep=';'
)

# Save to Excel
df_pivot.to_excel(output_excel_path, index=False)
