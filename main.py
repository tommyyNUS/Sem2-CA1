# (0) import Packages
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

#Define functions

data = pd.read_csv('Detail_listings.csv')

#Generate Summary Statistics
print("-----------------------")
print("Data Dimensions:  ", data.shape)
sumry = data.describe().transpose()
print("Summary Statistics:\n",sumry,'\n')

#Remove irrelevant columns
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
            'maximum_nights', 'city', 'host_response_rate', 'property_type',
            'number_of_reviews', 'review_scores_communication', 'host_total_listings_count',
            'host_listings_count', 'bed_type', 'host_has_profile_pic', 'review_scores_accuracy',
            'minimum_nights'], axis = 1)

#Handle missing data
data[data.isnull().any(axis=1)]
data = data.dropna()
data = data.replace(to_replace="[$,%]", value='', regex=True)

#Set independant and dependant variables
X = data.drop("price", axis = 1)
Y = data.iloc[:, 8].values
Y = pd.to_numeric(Y)

#Encode Categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[['host_response_time']]= le.fit_transform(X[['host_response_time']].astype(str))
X[['room_type']]= le.fit_transform(X[['room_type']].astype(str))
#X[['bed_type']]= le.fit_transform(X[['bed_type']].astype(str))
X[['cancellation_policy']]= le.fit_transform(X[['cancellation_policy']].astype(str))

#One hot encode categories
X = pd.concat([X, pd.get_dummies(X['host_is_superhost'], prefix='host_is_superhost')],axis=1)
#X = pd.concat([X, pd.get_dummies(X['host_has_profile_pic'], prefix='host_has_profile_pic')],axis=1)
X = pd.concat([X, pd.get_dummies(X['host_identity_verified'], prefix='host_identity_verified')],axis=1)
#X = pd.concat([X, pd.get_dummies(X['city'], prefix='city')],axis=1)
#X = pd.concat([X, pd.get_dummies(X['property_type'], prefix='property_type')],axis=1)
X = pd.concat([X, pd.get_dummies(X['instant_bookable'], prefix='instant_bookable')],axis=1)
X = pd.concat([X, pd.get_dummies(X['require_guest_profile_picture'], prefix='require_guest_profile_picture')],axis=1)
X = pd.concat([X, pd.get_dummies(X['require_guest_phone_verification'], prefix='require_guest_phone_verification')],axis=1)

X.drop(['host_is_superhost'], axis=1, inplace=True)
#X.drop(['host_has_profile_pic'], axis=1, inplace=True)
X.drop(['host_identity_verified'], axis=1, inplace=True)
#X.drop(['city'], axis=1, inplace=True)
#X.drop(['property_type'], axis=1, inplace=True)
X.drop(['instant_bookable'], axis=1, inplace=True)
X.drop(['require_guest_profile_picture'], axis=1, inplace=True)
X.drop(['require_guest_phone_verification'], axis=1, inplace=True)

#Build optimal model using Backward elimination, SL = 0.05
#Dropped columns: city, host_response_rate, property_type, number_of_reviews, review_scores_communication,
#  host_total_listings_count, host_listings_count, availability, availability_90, availability_30
# bed_type, host_has_profile_pic_f
#import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm
col_one = pd.DataFrame(index = np.arange(31253), columns = np.arange(1))
col_one[0] = 1
X.insert(0, 'one', col_one)
X_opt = X
X_opt = X_opt.drop(['availability_90', 'availability_30'], axis = 1)
regressor_OLS = sm.OLS(endog = Y.astype(float), exog = X_opt.astype(float)).fit()
regressor_OLS.summary()

#Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
sc = MinMaxScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

#Fitting Ridge linear regression model to dataset
from sklearn.linear_model import Ridge
lrid = Ridge(alpha=20.0).fit(X_train, y_train)

#Fitting Ridge linear regression model to dataset
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=2.0, max_iter=10000).fit(X_train, y_train)

#Fitting multiple linear regression model to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Fitting polynomial regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.fit_transform(X_test)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

#Fitting decision tree model to the dataset
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state = 0)
tree_reg.fit(X_train, y_train)

#Fitting random forest to the dataset. Ensemble Learning.
from sklearn.ensemble import RandomForestRegressor
rdm_reg = RandomForestRegressor(n_estimators = 100, random_state = 0)
rdm_reg.fit(X_train, y_train)

#Applying grid search to find best parameters for random forest
#from sklearn.model_selection import GridSearchCV
#parameters = [{'n_estimators': [10,50,100,150,200,250,300]}]
#grid_search = GridSearchCV(estimator=rdm_reg, param_grid = parameters, scoring='neg_mean_squared_error', cv = 10, n_jobs=2)
#if __name__ == '__main__':
#    grid_search = grid_search.fit(X_train, y_train)

#accuracy1 = grid_search.best_score_
#best_params = grid_search.best_params_
#nEstimators = grid_search.best_params_['n_estimators']
#rdm_reg = RandomForestRegressor(n_estimators = nEstimators, random_state = 0)
#rdm_reg.fit(X_train, y_train)

#Predicting results with multiple linear model
y_pred_rlin = lrid.predict(X_test)

#Evaluating ridge linear model
print("\nModel Summary (Ridge Linear Regression)")
print("-----------------------------------------------")
print("Explained Variance: "+str(explained_variance_score(y_test, y_pred_rlin)))
print("Mean absolute sq error: "+str(mean_absolute_error(y_test, y_pred_rlin)))
print("Root mean sq error: "+str(sqrt(mean_squared_error(y_test, y_pred_rlin))))
print("Median absolute error: "+str(median_absolute_error(y_test, y_pred_rlin)))
print("Ridge score (Train): "+str(lrid.score(X_train, y_train)))
print("Ridge score (Test): "+str(lrid.score(X_test, y_test)))
print("Model Intercept: "+str(lrid.intercept_))

#Predicting results with multiple linear model
y_pred_las = lasso.predict(X_test)

#Evaluating lasso linear model
print("\nModel Summary (Lasso Linear Regression)")
print("-----------------------------------------------")
print("Explained Variance: "+str(explained_variance_score(y_test, y_pred_rlin)))
print("Mean absolute sq error: "+str(mean_absolute_error(y_test, y_pred_rlin)))
print("Root mean sq error: "+str(sqrt(mean_squared_error(y_test, y_pred_rlin))))
print("Median absolute error: "+str(median_absolute_error(y_test, y_pred_rlin)))
print("Lasso score (Train): "+str(lasso.score(X_train, y_train)))
print("Lasso score (Test): "+str(lasso.score(X_test, y_test)))
print("Model Intercept: "+str(lasso.intercept_))

#Predicting results with multiple linear model
y_pred_mlin = regressor.predict(X_test)

#Evaluating multiple linear model
print("\nModel Summary (Multiple Linear Regression)")
print("-----------------------------------------------")
print("Explained Variance: "+str(explained_variance_score(y_test, y_pred_mlin)))
print("Mean absolute sq error: "+str(mean_absolute_error(y_test, y_pred_mlin)))
print("Root mean sq error: "+str(sqrt(mean_squared_error(y_test, y_pred_mlin))))
print("Median absolute error: "+str(median_absolute_error(y_test, y_pred_mlin)))
print("Multiple Linear score (Train): "+str(regressor.score(X_train, y_train)))
print("Multiple Linear score (Test): "+str(regressor.score(X_test, y_test)))
print("Model Intercept: "+str(regressor.intercept_))

#Predicting results with polynomial regression model
y_pred_poly = lin_reg_2.predict(X_poly_test)

#Evaluating polynomial model
print("\nModel Summary (Polynomial Regression)")
print("-----------------------------------------------")
print("Explained Variance: "+str(explained_variance_score(y_test, y_pred_poly)))
print("Mean absolute sq error: "+str(mean_absolute_error(y_test, y_pred_poly)))
print("Root mean sq error: "+str(sqrt(mean_squared_error(y_test, y_pred_poly))))
print("Median absolute error: "+str(median_absolute_error(y_test, y_pred_poly)))
print("Polynomial Linear score (Train): "+str(lin_reg_2.score(X_poly, y_train)))
print("Polynomial Linear score (Test): "+str(lin_reg_2.score(X_poly_test, y_test)))
print("Model Intercept: "+str(lin_reg_2.intercept_))

#Predicting results with decision tree model
y_pred_dtree = tree_reg.predict(X_test)

#Evaluating decision tree model
print("\nModel Summary (Decision Tree)")
print("-----------------------------------------------")
print("Explained Variance: "+str(explained_variance_score(y_test, y_pred_dtree)))
print("Mean absolute sq error: "+str(mean_absolute_error(y_test, y_pred_dtree)))
print("Root mean sq error: "+str(sqrt(mean_squared_error(y_test, y_pred_dtree))))
print("Median absolute error: "+str(median_absolute_error(y_test, y_pred_dtree)))
print("R2 score: "+str(r2_score(y_test, y_pred_dtree)))

#Predicting results with random forests model
y_pred_rtree = rdm_reg.predict(X_test)

#Evaluating random forest model
print("\nModel Summary (Random forest)")
print("-----------------------------------------------")
print("Explained Variance: "+str(explained_variance_score(y_test, y_pred_dtree)))
print("Mean absolute sq error: "+str(mean_absolute_error(y_test, y_pred_dtree)))
print("Root mean sq error: "+str(sqrt(mean_squared_error(y_test, y_pred_dtree))))
print("Median absolute error: "+str(median_absolute_error(y_test, y_pred_dtree)))
print("R2 score: "+str(r2_score(y_test, y_pred_rtree)))

#Visualizing Polynomial Results
#plt.scatter(X_test['accommodates'], y_test, color = 'red')
#plt.plot(X_test['accommodates'], y_pred_mlin, color = 'blue')
#plt.title('Polynomial Regression')
#plt.xlabel('Number of bedrooms')
#plt.ylabel('Salary')
#plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_mlin)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
plt.show()

import seaborn as sns
ax1 = sns.distplot(y_test, hist=False, color='r', label='Actual Value')
sns.distplot(y_pred_mlin, hist=False, color='b', label='Fitted Values', ax = ax1)

#ax1 = sns.distplot(y_test, hist=False, color='r', label='Actual Value')
#sns.distplot(y_pred_poly, hist=False, color='b', label='Fitted Values', ax = ax1)

#ax1 = sns.distplot(y_test, hist=False, color='r', label='Actual Value')
#sns.distplot(y_pred_dtree, hist=False, color='b', label='Fitted Values', ax = ax1)

#ax1 = sns.distplot(y_test, hist=False, color='r', label='Actual Value')
#sns.distplot(y_pred_rtree, hist=False, color='b', label='Fitted Values', ax = ax1)
