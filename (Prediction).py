import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import feature_selection as fs
import numpy.random as nr

# import the cleaned dataset from a csv file
data = pd.read_csv(r'data_cleaned.csv')
data.head(

)

# define a random state number for all random state 
random_state_num = 0



print("Shape of the dataset:", data.shape)


print("available columns and their data types:\n", data.dtypes)



# remove the ARE outliers
data = data[data['country']!='ARE']

# print shape after removing the outliers
print("Shape of the dataset after removing the ARE outliers:", data.shape)

 
# ### Selection of dependent and independent variables
# 
# During the previous stage of the project the features which have the biggest potential to build as many dependencies as possible. The features were chosen as follows:
# 
# * Dependent variable (DV) to be predicted:
#     - co2_percap*: CO2 emissions per capita (metric tons)
# * Independent variables:
#     - 'cereal_yield': Cereal yield (kg per hectare)
#     - 'fdi_perc_gdp': Foreign direct investment, net inflows (% of GDP)
#     - 'gni_per_cap': GNI per capita (Atlas \\$)
#     - 'en_per_cap': Energy use per capita (kilograms of oil equivalent)
#     - 'pop_urb_aggl_perc': Population in urban agglomerations >1million (\%)
#     - 'prot_area_perc': Nationally terrestrial protected areas (\% of total land area)
#     - 'gdp': GDP (\\$)
#     - 'pop_growth_perc': Population growth (annual \%)
#     - 'urb_pop_growth_perc': Urban population growth (annual \%)


# choosing features and label columns
feature_cols = ['cereal_yield','fdi_perc_gdp','gni_per_cap', 'en_per_cap',
                'pop_urb_aggl_perc', 'prot_area_perc', 'pop_growth_perc', 'urb_pop_growth_perc']
label_col = ['co2_per_cap']

# convert into nympy arrays (required for scikit-learn models)
features = np.array(data[feature_cols])
labels =  np.array(data[label_col])

print(features)

# ***
# 
# ### Train-Test Split and Cross-Validation
# 
# To validate the model on unseen data, the dataset is split into training and testing sets with an 80:20 ratio. 
# To improve generalization, cross-validation is applied on the training set for feature selection, hyperparameter tuning, and performance evaluation.
# The dataset split is performed in the following code snippet:
# 


# split into training and testing subsets
nr.seed(1)
features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.2,
                                                                            random_state=random_state_num)


# Feature selection is performed using `RFECV` from `sklearn.feature_selection`, 
# which applies Recursive Feature Elimination with Cross-Validation (based on R² score) to rank and retain top features. 
# The resulting reduced sets are used for training and testing (`features_train_reduced` and `features_test_reduced`).


# Set folds for cross validation for the feature selection
nr.seed(1)
feature_folds = ms.KFold(n_splits=4, shuffle = True, random_state=random_state_num)

# Define the model
rf_selector = RandomForestRegressor(random_state=random_state_num)

# Define an objects for a model for recursive feature elimination with CV
nr.seed(1)
selector = fs.RFECV(estimator = rf_selector, cv = feature_folds, scoring = 'r2', n_jobs=-1)

selector = selector.fit(features_train, np.ravel(labels_train))

selector.support_
print("Feature ranking after RFECV:")
print(selector.ranking_)

# print the important features
ranks_transform = list(np.transpose(selector.ranking_))
chosen_features = [i for i,j in zip(feature_cols,ranks_transform) if j==1]
print("Chosen important features:")
print(chosen_features)


# Reduce features
features_train_reduced = selector.transform(features_train)
features_test_reduced = selector.transform(features_test)

print("Training subset shape before the recursive feature elimination: ",features_train.shape)
print("Training subset array shape after the recursive feature elimination: ", features_train_reduced.shape)
print("Test subset array shape after the recursive feature elimination: ",features_test_reduced.shape)

# ***
# 
# ### Hyperparameter tuning
# 
# Random Forest is an algorithm with multiple hyperparameters which can have a range of values. In order to find the hyperparameters which would be most suitable for the current data, it is necessary to conduct hyperparameter tuning. The parameters which will be tuned in this case are:
# 
# * n_estimators - number of decision trees in the random forest
# * max_features - number of features to consider at every split
# * max_depth - maximum number of levels in a tree
# * min_samples_split - minimum number of samples required to split a node
# * min_samples_leaf - minimum number of samples required at each leaf node
# 
# The tuning is executed by applying a cross-validated evaluation of the model for different combinations of preliminary defined ranges for the parameters. The output is the model with the hyperparameters which exhibits the best R2 score compared to other parameter combinations.
# 
# Define the hyperparameter ranges to be investigated as a parameter grid (dictionary *param_grid*):


# Define value ranges for each hyperparameter
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Keep the rest unchanged
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)

# Create the hyperparameter grid
param_grid = {
    'n_estimators': n_estimators,
    'max_features': ['sqrt', 'log2', None] ,
    'max_depth': max_depth,
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Define the RandomizedSearchCV object, which will evaluate the R2 scores of models with randomly picked combinations from the defined hyperparameter grid:


# define the cross-validation folds for the hyperparameter tuning
nr.seed(1)
inside_folds = ms.KFold(n_splits=5, shuffle = True, random_state=random_state_num)

# Define the random forest regressor model object
rf_tuner = RandomForestRegressor(random_state=random_state_num)

# Perform a randomized search on the grid
nr.seed(1)
rf_model = ms.RandomizedSearchCV(estimator=rf_tuner, param_distributions = param_grid,
                                 random_state=random_state_num, refit=True,
                                 cv=inside_folds, scoring = 'r2', return_train_score=True, n_jobs=-1)

# Once fitted to the training data, the `RandomizedSearchCV` will return the best Random Forest model in the best_estimator_ attribute—this model uses the hyperparameter combination that gave the highest R² score during testing.
# 
# It's important to note that after finding the best combination, the model is automatically retrained on the full training set (thanks to refit=True in the `RandomizedSearchCV` setup).
# 
# Finally, the best hyperparameters are printed.


# Fit the grid search model object with cross-validation on the data
rf_model.fit(features_train_reduced, np.ravel(labels_train))

# Corrected variable name
print("Best Parameters:", rf_model.best_params_)

# Assign the best model to the model variable *rf_best_model* to be used further:

# pick the model with the best hyperparameter combination for further analysis
rf_best_model = rf_model.best_estimator_


# ### Evaluation of the model with the best hyperparameters on the training subset with cross-validation
# 
# The random forest model object *rf_best_model* with the most important features and the most suitable hyperparameters will be now evaluated on the training subset with cross-validation first:


# define the cross-validation folds for the evaluation
nr.seed(1)
outside_folds = ms.KFold(n_splits=10, shuffle = True, random_state=random_state_num)

# Evaluate the model on the training subset with cross-validation
nr.seed(1)
cv_eval = cross_val_score(rf_best_model, features_train_reduced, labels_train, cv = outside_folds, n_jobs=-1)

print('Mean R2 score of all CV folds = %4.3f' % np.mean(cv_eval))
print('Standard deviation of the R2 score over all folds = %4.3f' % np.std(cv_eval))
print('R2 score for each fold:')

# print the R2 score for each fold
for i, j in enumerate(cv_eval):
    print('Fold %2d    %4.3f' % (i+1, j))

# - **Mean R² score (10 folds):** 0.986  
# - **Standard deviation:** 0.004  


# Finally, the model should be validated on previously unseen data.


# make predictions from the features of the testing subset
predictions = rf_best_model.predict(features_test_reduced)

# calculate the metrics basing on the predicted and true values for the test subset
r2 = r2_score(y_true=labels_test, y_pred=predictions)
mse = mean_squared_error(y_true=labels_test, y_pred=predictions)
rmse = np.sqrt(mse)

print(f"R2 = {r2}, Mean Squared Error (MSE) = {mse}, Root Mean Squared Error (RMSE) = {rmse}")

# ### Model Evaluation & Prediction Quality
# 
# The obtained R² score of **0.986** may seem optimistic at first. However, several steps were taken to reduce overfitting and improve generalization:
# 
# - Dataset split with **80% used for training**
# - **Cross-validation** during feature selection, hyperparameter tuning, and model training
# 
# Note: **MSE and RMSE** values are not directly comparable to the dependent variable (CO₂ per capita), which ranges from **0 to 20 tons**.
# 
# To further evaluate performance, a **regression plot** is used to visualize predicted vs. actual CO₂ emissions per capita from the test set.
# 


import seaborn as sns
# plot predicted vs true values of the test subset

f,ax=plt.subplots(figsize=(20,15))
sns.set_theme(font_scale=2)

sns.regplot(x=predictions, y=np.transpose(labels_test)[0,:], fit_reg=True)
plt.xlabel("CO2 emissions per capita [t] - predicted")
plt.ylabel("CO2 emissions per capita [t] - true")
plt.title("Correlation coefficient R="+str(round(np.corrcoef(predictions,np.transpose(labels_test)[0,:])[0,1],2)))
plt.show()


import joblib

# Save the trained model to file
joblib.dump(rf_best_model, 'forecasting_co2_emmision.pkl')
print("Model saved to 'forecasting_co2_emmision.pkl")

# Load model from file
loaded_model = joblib.load('forecasting_co2_emmision.pkl')
print(" Model loaded successfully.")

# ### Calculating Compound Annual Growth Rates (CAGR) for Key Features
# 
# The process involves:
# - Filtering the dataset for a predefined list of selected countries.
# - Sorting the data by year for each country.
# - Calculating the CAGR for each feature based on its value in the earliest and latest available years.
# - Skipping entries where data is missing, invalid, or where the time range is insufficient.
# 
# This analysis helps identify whether each feature has increased or decreased over time in each country, offering insights into national trends that may impact CO₂ emissions.
# 
# The final output displays feature-wise growth rates (in percentages) for each country in the form:
# 
# 


# List of selected features
selected_features = ['cereal_yield', 'gni_per_cap', 'en_per_cap',
                     'pop_urb_aggl_perc', 'prot_area_perc',
                     'pop_growth_perc', 'urb_pop_growth_perc']


selected_countries = ['IND', 'USA', 'PAK', 'RUS', 'NZL']

# Filter the dataset to include only the selected countries
df_filtered = data[data['country'].isin(selected_countries)]

# Dictionary to store the growth rates for each country
growth_rates = {} #dictionary {}, List[]

# Loop over each selected country
for country in selected_countries:
    # Get data for the current country and sort it by year
    country_data = data[(data['country'] == country)].sort_values('year')

    # Identify the start and end year for the country
    start_year = country_data['year'].min()
    end_year = country_data['year'].max()
    years = end_year - start_year  # Total number of years between start and end

    # Dictionary to store growth rates of all features for the current country
    country_growth = {}

    # Skip this country if the time span is not valid (e.g., only one year of data)
    if years <= 0:
        print(f"Skipping {country} due to insufficient year range.")
        continue

    # Loop through each selected feature
    for feature in selected_features:
        # Get the feature value in the start year
        start_value = country_data[country_data['year'] == start_year][feature].values
        # Get the feature value in the end year
        end_value = country_data[country_data['year'] == end_year][feature].values

        # Skip if either value is missing
        if len(start_value) == 0 or len(end_value) == 0:
            continue

        # Extract scalar values from arrays
        start_value = start_value[0]
        end_value = end_value[0]

        # Skip if values are non-positive or not finite (e.g., NaN, inf)
        if start_value <= 0 or end_value <= 0 or not np.isfinite(start_value) or not np.isfinite(end_value):
            continue

        # Compute the Compound Annual Growth Rate (CAGR)
        cagr = (end_value / start_value) ** (1 / years) - 1
        # Store the result in the country-specific dictionary
        country_growth[feature] = cagr

    # Save the growth rates of all features for the current country
    growth_rates[country] = country_growth

# Display the calculated growth rates in a readable format
print("\nGrowth Rates (CAGR) from {} to {}:\n".format(start_year, end_year))

# Loop through each country and its corresponding growth rate dictionary
for country, features in growth_rates.items():
    print(f"{country}")  # Print the country name with an icon

    # Loop through each feature and its CAGR value
    for feature, rate in features.items():
        # Determine whether to display a plus or minus sign
        sign = '+' if rate >= 0 else '−'

        # Print the feature name and its growth rate percentage (formatted to two decimal places)
        print(f"  • {feature}: {sign}{abs(rate * 100):.2f}%")

    # Add a line break between countries for readability
    print()



# Forecasting CO₂ Emissions per Capita (Next 20 Years)

# Define the range of years to forecast (next 20 years beyond the last available year)
last_year = data['year'].max()
future_years = list(range(last_year + 1, last_year + 21))

# Initialize a list to store forecasted results
forecast_results = []

# Loop through each selected country for prediction
for country in selected_countries:
    country_data = data[(data['country'] == country)].sort_values('year')

    # Skip countries with missing feature values
    if country_data[selected_features].dropna().empty:
        print(f"Skipping {country} due to missing values.")
        continue

    # Take the latest complete record (most recent year) for the country
    latest_row = country_data[selected_features].dropna().iloc[-1].copy()

    # Forecast for each year into the future
    for year in future_years:
        # Apply the previously calculated CAGR to each feature
        for feature in selected_features:
            growth_rate = growth_rates.get(country, {}).get(feature, 0.0)
            latest_row[feature] *= (1 + growth_rate)

        # Use trained model to predict CO₂ emissions per capita
        input_features = latest_row.values.reshape(1, -1)
        predicted_co2 = loaded_model.predict(input_features)[0]

        # Store the forecast result
        forecast_results.append({
            'country': country,
            'year': year,
            'co2_percap': predicted_co2
        })

# Convert the list of predictions into a DataFrame
df_forecast = pd.DataFrame(forecast_results)

# Plot forecasted CO₂ per capita for all countries over the next 20 years
print('📈 Forecasted CO2 Emissions per Capita (Next 20 Years)')
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_forecast, x='year', y='co2_percap', hue='country', marker='o')
plt.title('Forecasted CO2 Emissions per Capita (Next 20 Years)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO2 per Capita (metric tons)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the forecast values for the last 5 years for India
print("\n📊 Forecasted CO₂ per Capita for Last 5 Years in Forecast Period (India):\n")
print(df_forecast[df_forecast['country'] == 'IND'].sort_values(by='year').tail(5))

# ***
# 
# This plot illustrates the projected trends of CO₂ emissions per capita for five countries — USA, RUS, NZL, IND, and PAK — over a 20-year period.
#