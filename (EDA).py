import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


# import the cleaned dataset from a csv file
data = pd.read_csv(r'data_cleaned.csv')


print("Shape of the dataset:", data.shape)

print("available columns and their data types:", data.dtypes)


print("Overview of the first 7 rows:", data.head(7))

print("Descriptive statistics:", data.describe().T)

# Global Average CO₂ Emissions per Capita Over Time (in metric tons) from **1991 to 2008**.

# Group by Year and calculate mean CO₂ per capita
df_yearly_avg = data.groupby('year')['co2_per_cap'].mean().reset_index()


plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='year', y='co2_per_cap', marker='o')
plt.title('Global Average CO2 Emissions per Capita Over Time')
plt.ylabel('CO2 per Capita (metric tons)')
plt.xlabel('Year')
plt.grid(True)
plt.tight_layout()
plt.show()

# Blue Line is the **mean CO₂ emissions per capita** globally for each year.
# 
# Shaded Blue Area is the Confidence Band 
#   Indicates the variation or uncertainty (e.g., standard deviation or confidence interval) around the average.  
#   A wider band means greater variability in countries in that year.
# 


# Total CO₂ Emissions vs Population – Chart relationship between a country's **total CO₂ emissions** and its **population**.
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='pop', y='co2_ttl', marker='^')
plt.title('Total CO2 Emissions against Population')
plt.ylabel('Total CO2 emissions')
plt.xlabel('Population')
plt.grid(True)
plt.tight_layout()
plt.show()


# Create a column for the total energy use:


# create a column for the total energy use
data['en_ttl'] = data['en_per_gdp'] * data['gdp'] /1000

# Choosing the Best Unit for CO₂ Emissions & Energy Use
# 
# To determine the most meaningful unit for *CO₂ emissions* and *energy use*, we analyze their **correlations** with other variables.
#  by usingcorrelation matrix of all variables.
# 

# select all features
features_all = data[['country','year','cereal_yield','fdi_perc_gdp','gni_per_cap',
                     'en_per_gdp', 'en_per_cap', 'en_ttl', 'co2_ttl', 'co2_per_cap',
                     'co2_per_gdp', 'pop_urb_aggl_perc', 'prot_area_perc', 'gdp',
                     'pop_growth_perc', 'pop', 'urb_pop_growth_perc']]

# plot a correlation of all features
# correlation matrix
sns.set_theme(font_scale=2)
f,ax=plt.subplots(figsize=(30,20))
sns.heatmap(features_all.drop(['country'], axis=1).corr(), annot=True, cmap='coolwarm', fmt = ".2f", 
            center=0, vmin=-1, vmax=1)
plt.title('Correlation between features', fontsize=25, weight='bold' )
plt.show()

sns.set_theme(font_scale=1)


# - When comparing the dependencies of co2_ttl, co2_per_cap, and co2_per_gdp with other features, **co2_per_cap** shows stronger correlations with more variables.
# 
# - CO₂ emissions independent of population size, making it more useful for comparing countries of different sizes and populations.
# 
# - Since most chosen variables are already linked to population, including pop (population count) adds little value — this is also supported by its low correlation.
# 
# Based on this, the following features will be excluded from further analysis due to weak correlations: pop = -0.1, en_per_gdp = -0.1, en_ttl = 0.09, co2_per_gdp, and co2_ttl = 0.03.


features_for_vif = data[['cereal_yield','fdi_perc_gdp','gni_per_cap', 'en_per_cap', 'co2_per_cap',
                     'pop_urb_aggl_perc', 'prot_area_perc', 'gdp',  'pop_growth_perc', 'urb_pop_growth_perc']]


vif_data = pd.DataFrame()
vif_data["feature"] = features_for_vif.columns # Use the columns from the features_for_vif DataFrame
vif_data["VIF"] = [variance_inflation_factor(features_for_vif.values, i)
                   for i in range(features_for_vif.shape[1])] # Use the values and number of columns from features_for_vif
print(vif_data)

# VIF values measure how much a feature's variance is inflated due to multicollinearity with other features:
# 
# - VIF < 5 → Low multicollinearity (generally safe)
# 
# - VIF 5–10 → Moderate multicollinearity (potential issue)
# 
# - VIF > 10 → High multicollinearity (problematic, should consider dropping or combining)


features = features_all[['cereal_yield','fdi_perc_gdp','gni_per_cap', 'en_per_cap', 'co2_per_cap',
                     'pop_urb_aggl_perc', 'prot_area_perc', 'gdp',  'pop_growth_perc', 'urb_pop_growth_perc']]

# ***
# 
# ## 6. Prepare the visualizations
# 
# ### Plotting preparation
# 
# #### Ensure easier labeling of the plots
# In order to make the labeling of the variables within plots easier in the code, a dictionary with the column names and variable labels to use on axes is defined:


# a dictionary with feature labels
labels_dict = {'gni_per_cap':'GNI per capita [Atlas $]',
               'gdp':'Gross Domestic Product [$]',
               'cereal_yield':'Cereal yield [kg/ha]',
               'prot_area_perc': 'Nationally terrestrial protected areas [% of total land area]',
               'fdi_perc_gdp': 'Foreign Direct Investment [% of GDP]',
               'pop_urb_aggl_perc': 'Population in urban agglomerations > 1mln [%]',
               'urb_pop_growth_perc':'Urban population growth [annual %]',
               'pop_growth_perc': 'Population growth [annual %]',
               'co2_per_cap':'CO2 emissions per capita [t]',
               'en_per_cap':'Energy use per capita [kg oil eq]' }

# #### Choose a subset of countries to plot
# The big amount of data points will result in slower processing of the plot and in a less clear representation. This can be avoided by choosing roughly half of the countries just for the paired scatter plot:


# get unique values in country column

unique_countries = data['country'].unique()
unique_countries

# #### CO₂ Emissions per Capita Over Time (Selected Countries)
# 
# This line chart illustrates the **CO₂ emissions per capita** (in metric tons) over time for five selected countries: **India (IND), New Zealand (NZL), Pakistan (PAK), USA, and Russia (RUS)**, spanning from the early 1990s to 2008.
# 


# Select countries to compare
selected_countries = ['IND', 'USA', 'PAK', 'RUS', 'NZL']

# Use the correct DataFrame variable 'data' and column names 'country', 'year', and 'co2_per_cap'
df_selected = data[data['country'].isin(selected_countries)]

# Plot
plt.figure(figsize=(12, 6))
# Use the correct DataFrame variable 'df_selected' and column names 'year' and 'co2_per_cap' for plotting
sns.lineplot(data=df_selected, x='year', y='co2_per_cap', hue='country', marker='o')
plt.title('CO2 Emissions per Capita Over Time (Selected Countries)')
plt.ylabel('CO2 per Capita (metric tons)')
plt.xlabel('Year')
plt.legend(title='Country')
plt.grid(True)
plt.tight_layout()
plt.show()

# Key Observations:
# - **USA** consistently records the highest per capita emissions, hovering around **19–20 metric tons**, though it shows a slight decline toward the end.
# - **Russia (RUS)** shows a significant **decline in the early 1990s**, likely due to economic restructuring, and then stabilizes around **11–12 metric tons**.
# - **New Zealand (NZL)** maintains a mid-level range, peaking around **9 metric tons**, then gradually declining.
# - **India (IND)** and **Pakistan (PAK)** have the **lowest emissions per capita**, though both show **gradual upward trends**, indicating growing emissions with development.
# 
# #### Insight:
# This comparison highlights stark differences in per capita emissions across developed and developing nations, and underscores the increasing trend in emissions for emerging economies like India and Pakistan.
# 


# select only rows for half of the countries chosen randomly in order to ensure better visibility
chosen_countries=['IND', 'LMC', 'LMY', 'MAR', 'MEX', 'MIC', 'MNA', 'MOZ', 'MYS',
'NGA', 'NLD', 'NZL', 'PAK', 'PAN', 'PER', 'PHL', 'PRT', 'PRY',
'ROM', 'SAS', 'SAU', 'SDN', 'SEN', 'SLV', 'SSA', 'SWE', 'SYR',
'TGO', 'THA', 'TUR', 'TZA', 'UMC', 'URY', 'USA', 'VEN', 'VNM',
'WLD', 'ZAF', 'ZAR', 'ZMB', 'ECA', 'POL', 'RUS', 'UKR', 'YEM',
'ETH', 'BEL']

features_chosen = features_all[features_all['country'].isin(chosen_countries)]

# Create plots and visualizations
# 
# The visualization is organized in a way that global overview of the data and dependencies is presented first, followed by more and more detailed representations of the more relevant relationships.
# 
# ### A global look onto all relationships
# 
# Scatter plots of all chosen variables and countries will give a first impression of possible trends:


sns.set_theme(font_scale=1.3)
pair = sns.pairplot(data=features_chosen, hue='country')
pair.fig.savefig("pairplot_output.png", dpi=300, bbox_inches='tight')

# The most obvious linear dependency of co2_per_cap is with en_per_cap. Apparent hints for nonlinear relationships can be observed in the plots of *co2_per_cap* versus *gni*, *pop_urb_aggl_perc*, *pop_growth_perc*, *urb_pop_growth_perc*.
# 
# 


# 
# #### Notable Outliers: United Arab Emirates
# 
# Another significant observation is the cluster of orange outlier points with CO₂ emissions per capita ranging between **25 and 40 metric tons**, all attributed to the **United Arab Emirates (ARE)**. While there are other country-specific outliers, they do not substantially impact the overall trend.
# 
# To enhance clarity, **ARE data points were removed**, and the updated plots are presented below:
# 


# choose features and label columns
feature_cols = ['country', 'cereal_yield','fdi_perc_gdp','gni_per_cap', 'en_per_cap', 'pop_urb_aggl_perc',
                    'prot_area_perc', 'gdp',  'pop_growth_perc', 'urb_pop_growth_perc', 'co2_per_cap']

# Keep as DataFrame for filtering
features_for_plot = data[feature_cols].copy() # Create a copy to avoid SettingWithCopyWarning

# remove the ARE outliers from the DataFrame used for plotting
features_for_plot = features_for_plot[features_for_plot['country']!='ARE']
