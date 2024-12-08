import pandas as pd
import numpy as np
import os
import ast

# Adjust these paths as needed
raw_csv_path = "/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/scripts/data/raw/theater_data2.csv"
output_dir = "/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/data/processed"
os.makedirs(output_dir, exist_ok=True)

output_csv_path = os.path.join(output_dir, 'cleaned_theater_data2.csv')
output_excel_path = os.path.join(output_dir, 'cleaned_theater_data2.xlsx')

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
        # This is the index value from the data, not necessarily starting at 0
        index_val = int(df.loc[0, col])
        label_col = prefix_label + idx_str
        label_val = df.loc[0, label_col]
        index_to_label[index_val] = label_val

    # Now just sort the keys:
    sorted_keys = sorted(index_to_label.keys())
    if len(sorted_keys) != dimension_size:
        print(f"Warning: For dimension {dimension_name}, expected {dimension_size} categories, got {len(sorted_keys)}.")

    labels = [index_to_label[k] for k in sorted_keys]
    return labels




# Extract labels for each dimension
# According to the given data, 'id' might look like: ['Aasta', 'Kategooria', 'Lavastuse liik / žanr', 'Näitaja']
year_labels = get_dimension_labels(df_raw, 'Aasta', size[0])
category_labels = get_dimension_labels(df_raw, 'Kategooria', size[1])
genre_labels = get_dimension_labels(df_raw, 'Lavastuse liik / žanr', size[2])
indicator_labels = get_dimension_labels(df_raw, 'Näitaja', size[3])

# Create MultiIndex
index = pd.MultiIndex.from_product([year_labels, category_labels, genre_labels, indicator_labels], names=ids)

data_flat = values_array.flatten()
df_final = pd.DataFrame(data=data_flat, index=index, columns=['value']).reset_index()

# Convert 'value' column to numeric where possible
df_final['value'] = df_final['value'].replace({None: np.nan, 'None': np.nan})
df_final['value'] = pd.to_numeric(df_final['value'], errors='coerce')

# Pivot the DataFrame so that Näitaja form columns
df_pivot = df_final.pivot_table(
    index=['Aasta', 'Kategooria', 'Lavastuse liik / žanr'],
    columns='Näitaja',
    values='value',
    aggfunc='first'
).reset_index()

df_pivot.columns.name = None

# Rename columns if needed to match desired naming
df_pivot.rename(columns={
    'Lavastused': 'Lavastused',
    '..uuslavastused': '..uuslavastused',
    'Etendused': 'Etendused',
    '..külalisetendused': '..külalisetendused',
    'Vaatajad, tuhat': 'Vaatajad, tuhat',
    'Teatriskäigud 1000 elaniku kohta': 'Teatriskäigud 1000 elaniku kohta',
    'Keskmine piletihind, eurot': 'Keskmine piletihind, eurot'
}, inplace=True)

# Ensure we have the correct order of columns
desired_columns = [
    'Aasta', 'Kategooria', 'Lavastuse liik / žanr',
    'Lavastused', '..uuslavastused', 'Etendused', '..külalisetendused',
    'Vaatajad, tuhat', 'Teatriskäigud 1000 elaniku kohta', 'Keskmine piletihind, eurot'
]
df_pivot = df_pivot[desired_columns]

# Replace NaN with '..'
df_pivot.fillna('..', inplace=True)

# Formatting numbers (similar to previous code)
numeric_columns = [
    'Lavastused', '..uuslavastused', 'Etendused', '..külalisetendused',
    'Vaatajad, tuhat', 'Teatriskäigud 1000 elaniku kohta', 'Keskmine piletihind, eurot'
]


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

# Sort and format the 'Aasta' column so that repeated years are blanked out
df_pivot.sort_values(by=['Aasta', 'Kategooria', 'Lavastuse liik / žanr'], inplace=True)
df_pivot.reset_index(drop=True, inplace=True)

df_pivot['Aasta'] = df_pivot['Aasta'].where(df_pivot['Aasta'].ne(df_pivot['Aasta'].shift()), '')

# Print final
print(df_pivot.to_string(index=False))

# Save to CSV and Excel
df_pivot.to_csv(
    output_csv_path,
    index=False,
    encoding='utf-8',
    sep=';'
)

# Ensure openpyxl is installed: pip install openpyxl
df_pivot.to_excel(output_excel_path, index=False)
