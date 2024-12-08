import pandas as pd
import numpy as np
import ast
import os

# Step 1: Read and parse the CSV file
df = pd.read_csv("/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/scripts/data/raw/theater_data4.csv")

# Extract 'id', 'size', and 'value'
ids = ast.literal_eval(df.loc[0, 'id'])
size = ast.literal_eval(df.loc[0, 'size'])
values = ast.literal_eval(df.loc[0, 'value'])

# Convert 'value' into an array and reshape
values_array = np.array(values, dtype=object)
values_array = values_array.reshape(size)

# Function to get labels for a dimension
def get_dimension_labels(df, dimension_name, size):
    index_to_label = {}
    for col in df.columns:
        if col.startswith(f'dimension.{dimension_name}.category.index.'):
            idx = col.split('.')[-1]
            index = int(df.loc[0, col])
            label_col = f'dimension.{dimension_name}.category.label.{idx}'
            label = df.loc[0, label_col]
            index_to_label[index] = label
    labels = [index_to_label[i] for i in range(size)]
    return labels

# Get labels for each dimension
year_labels = get_dimension_labels(df, 'Aasta', size[0])
sihtgrupp_labels = get_dimension_labels(df, 'Sihtgrupp', size[1])
naitaja_labels = get_dimension_labels(df, 'Näitaja', size[2])

# Create MultiIndex and DataFrame
index = pd.MultiIndex.from_product([year_labels, sihtgrupp_labels, naitaja_labels], names=ids)
data_flat = values_array.flatten()
df_final = pd.DataFrame(data=data_flat, index=index, columns=['value']).reset_index()

# Handle missing values
df_final['value'] = df_final['value'].replace({None: np.nan})
df_final['value'] = pd.to_numeric(df_final['value'], errors='coerce')

# Step 2: Pivot the DataFrame
df_pivot = df_final.pivot_table(
    index=['Aasta', 'Sihtgrupp'],
    columns='Näitaja',
    values='value',
    aggfunc='first'
).reset_index()

# Flatten the column MultiIndex
df_pivot.columns.name = None

# Step 3: Rename and rearrange columns
df_pivot.rename(columns={
    'Aasta': 'Year',
    'Sihtgrupp': 'Category',
    'Lavastused': 'Lavastused',
    '..uuslavastused': '..uuslavastused',
    'Etendused': 'Etendused',
    '..külalisetendused': '..külalisetendused',
    'Vaatajad, tuhat': 'Vaatajad, tuhat'
}, inplace=True)

df_pivot = df_pivot[['Year', 'Category', 'Lavastused', '..uuslavastused', 'Etendused', '..külalisetendused', 'Vaatajad, tuhat']]

# Step 4: Handle missing values and data types
df_pivot.fillna('..', inplace=True)

# Format numeric columns
numeric_columns = ['Lavastused', '..uuslavastused', 'Etendused', '..külalisetendused', 'Vaatajad, tuhat']

for col in numeric_columns:
    df_pivot[col] = df_pivot[col].apply(lambda x: '{:,}'.format(x).replace(',', ' ') if isinstance(x, (int, float)) else x)

# Step 5: Replace decimal dots with commas
df_pivot['Vaatajad, tuhat'] = df_pivot['Vaatajad, tuhat'].apply(lambda x: str(x).replace('.', ',') if isinstance(x, str) else x)

# Step 6: Adjust Year display
df_pivot.sort_values(by=['Year', 'Category'], inplace=True)
df_pivot.reset_index(drop=True, inplace=True)
df_pivot['Year'] = df_pivot['Year'].mask(df_pivot['Year'].duplicated())
df_pivot['Year'].fillna('', inplace=True)

# Step 7: Display the final DataFrame
print(df_pivot.to_string(index=False))


# Step 1: Ensure the directory exists
output_dir = '/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/data/processed'
os.makedirs(output_dir, exist_ok=True)

# Step 2: Define the output file path
output_csv_path = os.path.join(output_dir, 'cleaned_theater_data4.csv')
output_excel_path = os.path.join(output_dir, 'cleaned_theater_data4.xlsx')

# Step 3: Save as CSV with European formatting
df_pivot.to_csv(
    output_csv_path,
    index=False,
    encoding='utf-8',
    sep=';',        # Semicolon separator
    decimal=',',    # Comma decimal
    float_format='%.1f'  # One decimal place
)

# Step 4: Save as Excel file
df_pivot.to_excel(output_excel_path, index=False)

# Step 1: Read the CSV file
df = pd.read_csv("/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/scripts/data/raw/theater_data.csv")

# Step 2: Extract 'id', 'size', and 'value'
ids = ast.literal_eval(df.loc[0, 'id'])
size = ast.literal_eval(df.loc[0, 'size'])
values = ast.literal_eval(df.loc[0, 'value'])

# Step 3: Convert 'value' into an array and reshape
values_array = np.array(values, dtype=object)
values_array = values_array.reshape(size)

# Function to get labels for a dimension
def get_dimension_labels(df, dimension_name, size):
    index_to_label = {}
    for col in df.columns:
        if col.startswith(f'dimension.{dimension_name}.category.index.'):
            idx = col.split('.')[-1]
            index = int(df.loc[0, col])
            label_col = f'dimension.{dimension_name}.category.label.{idx}'
            label = df.loc[0, label_col]
            index_to_label[index] = label
    labels = [index_to_label[i] for i in range(size)]
    return labels

# Step 4: Get labels for each dimension
year_labels = get_dimension_labels(df, 'Aasta', size[0])
category_labels = get_dimension_labels(df, 'Kategooria', size[1])
indicator_labels = get_dimension_labels(df, 'Näitaja', size[2])

# Step 5: Create MultiIndex and DataFrame
index = pd.MultiIndex.from_product([year_labels, category_labels, indicator_labels], names=ids)
data_flat = values_array.flatten()
df_final = pd.DataFrame(data=data_flat, index=index, columns=['value']).reset_index()

# Step 6: Handle missing values
df_final['value'] = df_final['value'].replace({None: np.nan, 'None': np.nan})
df_final['value'] = pd.to_numeric(df_final['value'], errors='coerce')

# Step 7: Pivot the DataFrame
df_pivot = df_final.pivot_table(
    index=['Aasta', 'Kategooria'],
    columns='Näitaja',
    values='value',
    aggfunc='first'
).reset_index()

# Flatten the column MultiIndex
df_pivot.columns.name = None

# Step 8: Rename and reorder columns
df_pivot.rename(columns={
    'Aasta': 'Year',
    'Kategooria': 'Category',
    'Teatrite arv': 'Teatrite arv',
    'Lavastused': 'Lavastused',
    '..uuslavastused': '..uuslavastused',
    'Kultuuriüritused': 'Kultuuriüritused',
    'Etendused': 'Etendused',
    'Vaatajad, tuhat': 'Vaatajad, tuhat',
    'Teatriskäigud 1000 elaniku kohta': 'Teatriskäigud 1000 elaniku kohta',
    'Saalid': 'Saalid',
    'Istekohad': 'Istekohad'
}, inplace=True)

desired_columns = [
    'Year', 'Category', 'Teatrite arv', 'Lavastused', '..uuslavastused',
    'Kultuuriüritused', 'Etendused', 'Vaatajad, tuhat',
    'Teatriskäigud 1000 elaniku kohta', 'Saalid', 'Istekohad'
]
df_pivot = df_pivot[desired_columns]

# Step 9: Handle missing values and format data
df_pivot.fillna('..', inplace=True)

numeric_columns = [
    'Teatrite arv', 'Lavastused', '..uuslavastused',
    'Kultuuriüritused', 'Etendused', 'Vaatajad, tuhat',
    'Teatriskäigud 1000 elaniku kohta', 'Saalid', 'Istekohad'
]

for col in numeric_columns:
    df_pivot[col] = df_pivot[col].apply(lambda x: '{:,}'.format(x).replace(',', ' ') if isinstance(x, (int, float)) else x)

# Replace decimal dots with commas
df_pivot['Vaatajad, tuhat'] = df_pivot['Vaatajad, tuhat'].apply(lambda x: str(x).replace('.', ',') if isinstance(x, str) else x)
df_pivot['Teatriskäigud 1000 elaniku kohta'] = df_pivot['Teatriskäigud 1000 elaniku kohta'].apply(lambda x: str(x).replace('.', ',') if isinstance(x, str) else x)

# Step 10: Adjust Year display
df_pivot.sort_values(by=['Year', 'Category'], inplace=True)
df_pivot.reset_index(drop=True, inplace=True)
df_pivot['Year'] = df_pivot['Year'].mask(df_pivot['Year'].duplicated())
df_pivot['Year'].fillna('', inplace=True)

# Step 11: Display the final DataFrame
print(df_pivot.to_string(index=False))

output_path = '/home/aleksis/PycharmProjects/DatascienceGroupProject/TheaterDataAnalysis/data/processed'
df_pivot.to_csv(output_path, index=False, encoding='utf-8')
