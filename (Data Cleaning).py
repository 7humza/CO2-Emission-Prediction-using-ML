#libraries:
import pandas as pd
import numpy as np

# The complete Climate Change Dataset is imported into a pandas DataFrame from the downloaded file "climate_change_download_0.xls":


# define the file name and the data sheet
orig_data_file = r"C:\Users\ahmed\OneDrive\Desktop\MINORPROJECT\climate_change_download_0.xls"
data_sheet = "Data"

# read the data from the excel file to a pandas DataFrame
data_orig = pd.read_excel(io=orig_data_file, sheet_name=data_sheet)

# returns (rows, columns)
print("Shape of the original dataset:", data_orig.shape)

# returns column names
print("Available columns:\n",data_orig.columns)

# shows data types of each column
print("Column data types:\n",data_orig.dtypes)

# shows first 7 rows
print("Overview of the first 7 rows:\n",data_orig.head(7))

# mean, count, and standard deviation
print("Descriptive statistics of the columns:\n",data_orig.describe())


# Selects a column and returns its unique values.
print("Unique Series Name Values\n ", data_orig['Series name'].unique(), "\n")

print("Unique Seriese Code Values\n", data_orig['Series code'].unique(), "\n")

print("Unique SCALE Values\n", data_orig['SCALE'].unique(), "\n")

print("Unique Decimal VAlues\n", data_orig['Decimals'].unique(), "\n")


# Show *'Text'* in the *'SCALE'* and *'Decimals'* columns.
print(data_orig[data_orig['SCALE']=='Text'])

print(data_orig[data_orig['Decimals']=='Text'])


# #### Removing rows marked as "Text" in the "SCALE" and "Decimals" columns

# new DataFrame 
data_clean = data_orig

print("Original number of rows:")
print(data_clean.shape[0])

# remove rows which has "Text" in the SCALE column
data_clean = data_clean[data_clean['SCALE']!='Text']

print("Current number of rows:")
print(data_clean.shape[0]) # [0] is row [1] is coloums

#Removing the unnecessary columns "Country name", "Series code", "SCALE", "Decimals"
print("Original number of columns:")
print(data_clean.shape[1])

data_clean = data_clean.drop(['Country name', 'Series code', 'SCALE', 'Decimals'], axis='columns')

print("Current number of columns:")
print(data_clean.shape[1])

# #### Transform the ".." strings and emplty cells ("") into NaN (easy to recog.)
data_clean.iloc[:,2:] = data_clean.iloc[:,2:].replace({'':np.nan, '..':np.nan})

#Transform all data columns into a numerical data type
data_clean2 = data_clean.applymap(lambda x: pd.to_numeric(x, errors='ignore'))
# Errors are ignored in order to avoid error messages about the first two columns, which don't need to be transformed
# into numeric type anyway

print("Print the column data types after transformation:")
print(data_clean2.dtypes)

# #### Rename the features in column "Series name"
# define shorter names corresponding to most relevant variables in a dictionary
chosen_vars = {'Cereal yield (kg per hectare)': 'cereal_yield',
               'Foreign direct investment, net inflows (% of GDP)': 'fdi_perc_gdp',
               'Access to electricity (% of total population)': 'elec_access_perc',
               'Energy use per units of GDP (kg oil eq./$1,000 of 2005 PPP $)': 'en_per_gdp',
               'Energy use per capita (kilograms of oil equivalent)': 'en_per_cap',
               'CO2 emissions, total (KtCO2)': 'co2_ttl',
               'CO2 emissions per capita (metric tons)': 'co2_per_cap',
               'CO2 emissions per units of GDP (kg/$1,000 of 2005 PPP $)': 'co2_per_gdp',
               'Other GHG emissions, total (KtCO2e)': 'other_ghg_ttl',
               'Methane (CH4) emissions, total (KtCO2e)': 'ch4_ttl',
               'Nitrous oxide (N2O) emissions, total (KtCO2e)': 'n2o_ttl',
               'Droughts, floods, extreme temps (% pop. avg. 1990-2009)': 'nat_emerg',
               'Population in urban agglomerations >1million (%)': 'pop_urb_aggl_perc',
               'Nationally terrestrial protected areas (% of total land area)': 'prot_area_perc',
               'GDP ($)': 'gdp',
               'GNI per capita (Atlas $)': 'gni_per_cap',
               'Under-five mortality rate (per 1,000)': 'under_5_mort_rate',
               'Population growth (annual %)': 'pop_growth_perc',
               'Population': 'pop',
               'Urban population growth (annual %)': 'urb_pop_growth_perc',
               'Urban population': 'urb_pop'
                }

# rename all variables in the column "Series name" with comprehensible shorter versions
data_clean2['Series name'] = data_clean2['Series name'].replace(to_replace=chosen_vars)
 
# Transformed dataframe
print(data_clean2.head(7))

# To prepare the data for analysis, the values in the *'Series name'* column are pivoted into separate feature columns, while years are combined into a single column. This involves melting the dataset to align values with their corresponding country and year, and then merging them into a unified, structured DataFrame.
# save the short feature names into a list of strings
chosen_cols = list(chosen_vars.values())

# define an empty list, where sub-dataframes for each feature will be saved
frame_list = []

# iterate over all chosen features
for variable in chosen_cols:

    # pick only rows corresponding to the current feature
    frame = data_clean2[data_clean2['Series name'] == variable]

    # melt all the values for all years into one column and rename the columns correspondingly
    frame = frame.melt(id_vars=['Country code', 'Series name']).rename(columns={'Country code': 'country', 'variable': 'year', 'value': variable}).drop(['Series name'], axis='columns')
    #id_vars makes points fixed
    # add the melted dataframe for the current feature into the list
    frame_list.append(frame)


# merge all sub-frames into a single dataframe, making an outer binding on the key columns 'country','year'
from functools import reduce
all_vars = reduce(lambda left, right: pd.merge(left, right, on=['country','year'], how='outer'), frame_list)

# After this transformation, the new data frame has the following layout:

print(all_vars.head(7))

# Remove the remaining missing values in an optimal way
# Although some columns and rows with empty cells have already been deleted, there are still remaining missing values:

print("check the amount of missing values in each column")
print(all_vars.isnull().sum()) # there are still missing values

# Handling Missing Values

# Filtering the years by missing values
# Checking the amount of missing values for each year:
all_vars_clean = all_vars

#define an array with the unique year values
years_count_missing = dict.fromkeys(all_vars_clean['year'].unique(), 0)
for ind, row in all_vars_clean.iterrows():
    years_count_missing[row['year']] += row.isnull().sum()

# sort the years by missing values
years_missing_sorted = dict(sorted(years_count_missing.items(), key=lambda item: item[1]))

# print the missing values for each year
print("missing values by year:")
for key, val in years_missing_sorted.items():
    print(key, ":", val)

# ### Filtering by Year
# 
# remove countries with excessive missing data while preserving the time span as much as possible. To achieve this, a threshold for allowed NaN values per year is applied.
# Based on earlier analysis, the years 1991 to 2008 offer a good balance and are selected for further processing.
# 


print("number of missing values in the whole dataset before filtering the years:")
print(all_vars_clean.isnull().sum().sum())
print("number of rows before filtering the years:")
print(all_vars_clean.shape[0])

# filter only rows for years between 1991 and 2008 (having less missing values)
all_vars_clean = all_vars_clean[(all_vars_clean['year'] >= 1991) & (all_vars_clean['year'] <= 2008)]

print("number of missing values in the whole dataset after filtering the years:")
print(all_vars_clean.isnull().sum().sum())
print("number of rows after filtering the years:")
print(all_vars_clean.shape[0])

# #### Filtering the countries by missing values
# 
# The same procedure is applied to the filtering of countries with missing values. The following snippet shows the number of NaNs for each country.


# check the amount of missing values by country

# define an array with the unique country values
countries_count_missing = dict.fromkeys(all_vars_clean['country'].unique(), 0)

# iterate through all rows and count the amount of NaN values for each country
for ind, row in all_vars_clean.iterrows():
    countries_count_missing[row['country']] += row.isnull().sum()

# sort the countries by missing values
countries_missing_sorted = dict(sorted(countries_count_missing.items(), key=lambda item: item[1]))

# print the missing values for each country
print("missing values by country:")
for key, val in countries_missing_sorted.items():
    print(key, ":", val)

# This output would suggest to remove rows for countries with more than 90 missing values:


print("number of missing values in the whole dataset before filtering the countries:")
print(all_vars_clean.isnull().sum().sum())
print("number of rows before filtering the countries:")
print(all_vars_clean.shape[0])


# filter only rows for countries with less than 90 missing values
countries_filter = []
for key, val in countries_missing_sorted.items():
    if val<90:
        countries_filter.append(key)

all_vars_clean = all_vars_clean[all_vars_clean['country'].isin(countries_filter)]

print("number of missing values in the whole dataset after filtering the countries:")
print(all_vars_clean.isnull().sum().sum())
print("number of rows after filtering the countries:")
print(all_vars_clean.shape[0])

# #### Checking the features (columns) for missing values
# 
# The NaN values count in each column is:


all_vars_clean.isnull().sum()

# ### Dropping High-NaN Features
# 
# Even after filtering countries and years, some features like *elec_access_perc*, *other_ghg_ttl*, *ch4_ttl*, *n20_ttl*, and *nat_emerg* still have many missing values. Keeping them would significantly reduce the dataset size, so these columns are dropped to preserve more complete observations.


# remove features with more than 20 missing values

from itertools import compress

# create a boolean mapping of features with more than 20 missing values
vars_bad = all_vars_clean.isnull().sum()>20

# remove the columns corresponding to the mapping of the features with many missing values
all_vars_clean2 = all_vars_clean.drop(compress(data = all_vars_clean.columns, selectors = vars_bad), axis='columns')

print("Remaining missing values per column:")
print(all_vars_clean2.isnull().sum())

# Removing the rows with the remainin missing values will not impair the size of the dataset significantly, so these rows will be deleted:


# delete rows with any number of missing values
all_vars_clean3 = all_vars_clean2.dropna(axis='rows', how='any')

print("Remaining missing values per column:")
print(all_vars_clean3.isnull().sum())

print("Final shape of the cleaned dataset:")
print(all_vars_clean3.shape)

# ***
# 
# ### Export of the cleaned data frame to a csv file
all_vars_clean3.to_csv('data_cleaned.csv', index=False)