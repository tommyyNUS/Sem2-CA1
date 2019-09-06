# 1) import Packages
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error, median_absolute_error, accuracy_score
from math import sqrt
import statsmodels.regression.linear_model as sm
from sklearn.pipeline import make_pipeline

# 2) Define functions
def backwardElimination(x, Y, sl, columns):
    numVars = len(columns.columns)
    numVars2 = len(columns.columns)
    print("\nStarting backward elimination...")
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = Y.astype(float), exog = columns.astype(float)).fit()
        maxVar = float(max(regressor_OLS.pvalues))
        print("Highest P value is: "+str(maxVar))
        if maxVar > sl:
            for j in range(0, numVars2):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    print("Removing column "+str(j))
                    columns = columns.drop(columns.columns[j], axis=1)
                    numVars2 -= 1
        
    print(regressor_OLS.summary())
    return columns

#----------------------Start of code implemention -------------------------------
data = pd.read_csv('Detail_listings.csv')

# 3) Generate Summary Statistics
print("-----------------------")
print("Data Dimensions:  ", data.shape)
sumry = data.describe().transpose()
print("Summary Statistics:\n",sumry,'\n')

# 4) Remove irrelevant columns
data = data.drop(['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space',
            'description', 'experiences_offered', 'neighborhood_overview', 'notes', 
            'transit', 'access', 'interaction', 'house_rules', 'thumbnail_url',
            'medium_url', 'picture_url', 'picture_url', 'xl_picture_url',
            'host_id', 'host_url', 'host_name', 'host_since', 'host_location',
            'host_about', 'host_acceptance_rate', 'host_thumbnail_url',
            'host_picture_url', 'host_neighbourhood', 'host_verifications', 'street',
            'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
            'state', 'zipcode', 'market', 'smart_location', 'country_code', 'country',
            'latitude', 'longitude', 'is_location_exact', 'amenities', 'square_feet',
            'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee',
            'calendar_updated', 'has_availability', 'calendar_last_scraped', 'first_review',
            'last_review', 'requires_license', 'license', 'jurisdiction_names',
            'maximum_nights', 'host_response_rate',
            'number_of_reviews', 'review_scores_communication', 'host_total_listings_count',
            'host_listings_count', 'bed_type', 'city'], axis = 1)

# 5) Handle missing data
data[data.isnull().any(axis=1)]
data = data.dropna()
data = data.replace(to_replace="[$,%]", value='', regex=True)

# 6) Shuffle the data
data = data.sample(frac=1)

# 7) Set independant and dependant variables
X = data.drop("price", axis = 1)
Y = data['price']
Y = pd.to_numeric(Y)

# 8) Encode Categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[['host_response_time']]= le.fit_transform(X[['host_response_time']].astype(str))
X[['room_type']]= le.fit_transform(X[['room_type']].astype(str))
X[['cancellation_policy']]= le.fit_transform(X[['cancellation_policy']].astype(str))

# 9) One hot encode categories
X = pd.concat([X, pd.get_dummies(X['host_is_superhost'], prefix='host_is_superhost')],axis=1)
X = pd.concat([X, pd.get_dummies(X['host_has_profile_pic'], prefix='host_has_profile_pic')],axis=1)
X = pd.concat([X, pd.get_dummies(X['host_identity_verified'], prefix='host_identity_verified')],axis=1)
X = pd.concat([X, pd.get_dummies(X['property_type'], prefix='property_type')],axis=1)
X = pd.concat([X, pd.get_dummies(X['instant_bookable'], prefix='instant_bookable')],axis=1)
X = pd.concat([X, pd.get_dummies(X['require_guest_profile_picture'], prefix='require_guest_profile_picture')],axis=1)
X = pd.concat([X, pd.get_dummies(X['require_guest_phone_verification'], prefix='require_guest_phone_verification')],axis=1)

X.drop(['host_is_superhost'], axis=1, inplace=True)
X.drop(['host_has_profile_pic'], axis=1, inplace=True)
X.drop(['host_identity_verified'], axis=1, inplace=True)
X.drop(['property_type'], axis=1, inplace=True)
X.drop(['instant_bookable'], axis=1, inplace=True)
X.drop(['require_guest_profile_picture'], axis=1, inplace=True)
X.drop(['require_guest_phone_verification'], axis=1, inplace=True)

# 10) Build optimal model using Backward elimination
col_one = pd.DataFrame(index = np.arange(31253), columns = np.arange(1))
col_one[0] = 1
X.insert(0, 'one', col_one)
X_opt = X
SL = 0.000000001
X_opt = backwardElimination(X, Y, SL, X_opt)

# 11) Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.01, random_state = 0)

# 12) Feature Scaling
sc = MinMaxScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

# 13) Train models
#Fitting multiple linear regression model to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Fitting polynomial regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(X, Y)

#Fitting decision tree model to the dataset
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state = 0)
tree_reg.fit(X_train, y_train)

#Fitting random forest to the dataset. Ensemble Learning.
from sklearn.ensemble import RandomForestRegressor
rdm_reg = RandomForestRegressor(n_estimators = 1, random_state = 0)
#rdm_reg.fit(X_train, y_train)

#Applying grid search to find best parameters for random forest
print("\nConducting grid search for random forest... This will take some time...")
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [50,100,150,200,250,300]}]
grid_search = GridSearchCV(estimator=rdm_reg, param_grid = parameters, scoring='neg_mean_squared_error', cv = 10, n_jobs=2)
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    grid_search = grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
nEstimators = grid_search.best_params_['n_estimators']
print("\nBest number of estimators for random forest: "+str(nEstimators))
rdm_reg = RandomForestRegressor(n_estimators = nEstimators, random_state = 0)
#rdm_reg = RandomForestRegressor(n_estimators = 100, random_state = 0)
print("\nTraining random forest...")
rdm_reg.fit(X_train, y_train)

# 14) Predict results
#Predicting results with multiple linear model
y_pred_mlin = regressor.predict(X_test)

#Evaluating multiple linear model
print("\nModel Summary (Multiple Linear Regression)")
print("-----------------------------------------------")
print("Explained Variance: "+str(explained_variance_score(y_test, y_pred_mlin)))
print("Mean absolute error: "+str(mean_absolute_error(y_test, y_pred_mlin)))
print("Root mean sq error: "+str(sqrt(mean_squared_error(y_test, y_pred_mlin))))
print("Median absolute error: "+str(median_absolute_error(y_test, y_pred_mlin)))
print("R2 score: "+str(r2_score(y_test, y_pred_mlin)))

#Predicting results with polynomial regression model
y_pred_poly = poly_model.predict(X_test)

#Evaluating polynomial model
print("\nModel Summary (Polynomial Regression)")
print("-----------------------------------------------")
print("Explained Variance: "+str(explained_variance_score(y_test, y_pred_poly)))
print("Mean absolute error: "+str(mean_absolute_error(y_test, y_pred_poly)))
print("Root mean sq error: "+str(sqrt(mean_squared_error(y_test, y_pred_poly))))
print("Median absolute error: "+str(median_absolute_error(y_test, y_pred_poly)))
print("R2 score: "+str(r2_score(y_test, y_pred_poly)))

#Predicting results with decision tree model
y_pred_dtree = tree_reg.predict(X_test)

#Evaluating decision tree model
print("\nModel Summary (Decision Tree)")
print("-----------------------------------------------")
print("Explained Variance: "+str(explained_variance_score(y_test, y_pred_dtree)))
print("Mean absolute error: "+str(mean_absolute_error(y_test, y_pred_dtree)))
print("Root mean sq error: "+str(sqrt(mean_squared_error(y_test, y_pred_dtree))))
print("Median absolute error: "+str(median_absolute_error(y_test, y_pred_dtree)))
print("R2 score: "+str(r2_score(y_test, y_pred_dtree)))

#Predicting results with random forests model
y_pred_rtree = rdm_reg.predict(X_test)

#Evaluating random forest model
print("\nModel Summary (Random forest)")
print("-----------------------------------------------")
print("Explained Variance: "+str(explained_variance_score(y_test, y_pred_rtree)))
print("Mean absolute error: "+str(mean_absolute_error(y_test, y_pred_rtree)))
print("Root mean sq error: "+str(sqrt(mean_squared_error(y_test, y_pred_rtree))))
print("Median absolute error: "+str(median_absolute_error(y_test, y_pred_rtree)))
print("R2 score: "+str(r2_score(y_test, y_pred_rtree)))

# 15) Visualizing Results

#Consolidate actual Y values and predictions into 1 dataframe and display head(Top 30 rows)
y_values = pd.DataFrame()
y_values['Actual'] = y_test
y_values['Multiple Linear'] = y_pred_mlin
y_values['Polynomial'] = y_pred_poly
y_values['Decision Tree'] = y_pred_dtree
y_values['Random Forest'] = y_pred_rtree
print('\n---------- Actual VS All Prediction Values Comparison ----------\n')
print(y_values.head(n=30))

#--- Multiple Linear model ---
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

ax[0].scatter(y_test, y_pred_mlin)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[0].set_xlabel('actual')
ax[0].set_ylabel('predicted')
ax[0].set_title("Multiple Linear Regression")

ax[1].scatter(y_test, y_pred_mlin)
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[1].set_xlabel('actual')
ax[1].set_ylabel('predicted')
ax[1].set_xlim([0,1000])
ax[1].set_ylim([0,1000])
ax[1].set_title("Multiple Linear Regression (Zoomed In)")

plt.show()

#--- Polynomial model ---
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax[0].scatter(y_test, y_pred_poly)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[0].set_xlabel('actual')
ax[0].set_ylabel('predicted')
ax[0].set_title("Polynomial Regression")

ax[1].scatter(y_test, y_pred_poly)
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[1].set_xlabel('actual')
ax[1].set_ylabel('predicted')
ax[1].set_xlim([0,1000])
ax[1].set_ylim([0,1000])
ax[1].set_title("Polynomial Regression (Zoomed In)")

plt.show()

#--- Decision Tree model ---
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax[0].scatter(y_test, y_pred_dtree)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[0].set_xlabel('actual')
ax[0].set_ylabel('predicted')
ax[0].set_title("Decision Tree Regression")

ax[1].scatter(y_test, y_pred_dtree)
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[1].set_xlabel('actual')
ax[1].set_ylabel('predicted')
ax[1].set_xlim([0,1000])
ax[1].set_ylim([0,1000])
ax[1].set_title("Decision Tree Regression (Zoomed In)")

plt.show()

#--- Random Forest model ---
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax[0].scatter(y_test, y_pred_rtree)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[0].set_xlabel('actual')
ax[0].set_ylabel('predicted')
ax[0].set_title("Random Forest Regression")

ax[1].scatter(y_test, y_pred_rtree)
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax[1].set_xlabel('actual')
ax[1].set_ylabel('predicted')
ax[1].set_xlim([0,1000])
ax[1].set_ylim([0,1000])
ax[1].set_title("Random Forest Regression (Zoomed In)")

plt.show()

import seaborn as sns
sns.distplot(y_test, hist=False, color='r', label='Actual Value')
sns.distplot(y_pred_mlin, hist=False, color='b', label='Multiple Linear')
plt.title('Multiple Linear Regression')
plt.show()

sns.distplot(y_test, hist=False, color='r', label='Actual Value')
sns.distplot(y_pred_poly, hist=False, color='b', label='Polynomial')
plt.title('Polynomial Regression')
plt.show()

sns.distplot(y_test, hist=False, color='r', label='Actual Value')
sns.distplot(y_pred_dtree, hist=False, color='b', label='Decision Tree')
plt.title('Decision Tree Regression')
plt.show()

sns.distplot(y_test, hist=False, color='r', label='Actual Value')
sns.distplot(y_pred_rtree, hist=False, color='b', label='Random Forest')
plt.title('Random Forest Regression')
plt.show()
