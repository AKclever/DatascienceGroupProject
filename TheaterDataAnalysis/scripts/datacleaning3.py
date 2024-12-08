import pandas as pd
import numpy as np
import os
import ast

# Adjust these paths as needed
raw_csv_path = "/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/scripts/data/raw/theater_data3.csv"
output_dir = "/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/data/processed"
os.makedirs(output_dir, exist_ok=True)

output_csv_path = os.path.join(output_dir, 'cleaned_theater_data3.csv')
output_excel_path = os.path.join(output_dir, 'cleaned_theater_data3.xlsx')

df_raw = pd.read_csv(raw_csv_path)

# Extract 'id', 'size', and 'value'
ids = ast.literal_eval(df_raw.loc[0, 'id'])
size = ast.literal_eval(df_raw.loc[0, 'size'])
values = ast.literal_eval(df_raw.loc[0, 'value'])

values_array = np.array(values, dtype=object).reshape(size)

def get_dimension_labels(df, dimension_name, dimension_size):
    prefix_index = f"dimension.{dimension_name}.category.index."
    prefix_label = f"dimension.{dimension_name}.category.label."
    index_to_label = {}
    index_cols = [c for c in df.columns if c.startswith(prefix_index)]
    for col in index_cols:
        idx_str = col.split('.')[-1]
        index_val = int(df.loc[0, col])
        label_col = prefix_label + idx_str
        label_val = df.loc[0, label_col]
        index_to_label[index_val] = label_val

    sorted_keys = sorted(index_to_label.keys())
    if len(sorted_keys) != dimension_size:
        print(f"Warning: For dimension {dimension_name}, expected {dimension_size} categories, got {len(sorted_keys)}.")
    labels = [index_to_label[k] for k in sorted_keys]
    return labels

# According to the snippet, let's assume:
# ids = ['Kategooria', 'Näitaja', 'Aasta']
# size = [3, 13, 17] (for example)
category_labels = get_dimension_labels(df_raw, 'Kategooria', size[0])
indicator_labels = get_dimension_labels(df_raw, 'Näitaja', size[1])
year_labels = get_dimension_labels(df_raw, 'Aasta', size[2])

# Create MultiIndex
index = pd.MultiIndex.from_product([category_labels, indicator_labels, year_labels], names=ids)
data_flat = values_array.flatten()
df_final = pd.DataFrame(data=data_flat, index=index, columns=['value']).reset_index()

# Convert 'value' column to numeric where possible
df_final['value'] = df_final['value'].replace({None: np.nan, 'None': np.nan})
df_final['value'] = pd.to_numeric(df_final['value'], errors='coerce')

# Pivot the DataFrame so that 'Näitaja' forms columns and we have Year/Category in rows
# Adjust as needed: for example, if you want rows as Year and Category and columns as Näitaja:
df_pivot = df_final.pivot_table(
    index=['Aasta', 'Kategooria'],
    columns='Näitaja',
    values='value',
    aggfunc='first'
).reset_index()

df_pivot.columns.name = None

# If you have columns like 'Tulud, tuhat eurot' etc., rename them here.
# Make sure these original column names actually exist in 'df_pivot'.
# Below is just an example, adjust to your dataset's actual indicators:
rename_map = {
   'Tulud, tuhat eurot': 'Revenue (thousands of euros)',
   'Kulud ja investeeringud kokku, tuhat eurot': 'Costs and investments (thousands of euros)',
   # Add other renames as needed...
}
df_pivot.rename(columns=rename_map, inplace=True)

# If you have a desired column order, specify it here, ensuring it matches
# your dataset's actual columns:
# Example (adjust these as needed):
desired_columns = ['Aasta', 'Kategooria', 'Revenue (thousands of euros)', 'Costs and investments (thousands of euros)']
# Only reorder if these columns actually exist:
existing_columns = [c for c in desired_columns if c in df_pivot.columns]
if existing_columns:
    df_pivot = df_pivot[['Aasta', 'Kategooria'] + [c for c in df_pivot.columns if c not in ['Aasta','Kategooria']]]

# Replace NaN with '..'
df_pivot.fillna('..', inplace=True)

# If you want to format numeric columns similarly:
numeric_columns = [c for c in df_pivot.columns if c not in ['Aasta', 'Kategooria']]

def format_value(x):
    if x == '..':
        return x
    try:
        val = float(str(x).replace(',', '.'))
        formatted = f"{val:,.2f}".replace(",", " ")
        # Replace '.' with ',' for decimals
        formatted = formatted.replace('.', ',')
        # If it's essentially an integer, remove decimals
        if val.is_integer():
            formatted_int = f"{int(val):,}".replace(",", " ")
            return formatted_int
        return formatted
    except:
        return x

for col in numeric_columns:
    df_pivot[col] = df_pivot[col].apply(format_value)

# Sort by year and category
df_pivot.sort_values(by=['Aasta', 'Kategooria'], inplace=True, key=lambda x: x.where(x != '', np.nan))
df_pivot.reset_index(drop=True, inplace=True)

# Remove repeated year labels, leaving them blank for subsequent rows within the same year
df_pivot['Aasta'] = df_pivot['Aasta'].where(df_pivot['Aasta'].ne(df_pivot['Aasta'].shift()), '')

print(df_pivot.to_string(index=False))

# Save to CSV and Excel
df_pivot.to_csv(
    output_csv_path,
    index=False,
    encoding='utf-8',
    sep=';'
)

# Install openpyxl if not installed: pip install openpyxl
df_pivot.to_excel(output_excel_path, index=False)

