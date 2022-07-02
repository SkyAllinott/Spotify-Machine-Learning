import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Importing data: ###########################################################
data = pd.read_csv('G:/My Drive/Python Projects/Spotify API/spotify.csv')

#############################################################################

################ Exploratory Visualization: ################################

histogram('popularity', data)
#histogram("release_year", data)
#histogram("danceability", data)
#histogram("length", data)
#histogram("valence", data)


# Visualizing by grouping on year:
data['year'] = data['release_year'].astype(int)
datayear = data.groupby('year').mean().reset_index()
#graph('year', 'popularity', datayear)
graph('year', 'danceability', datayear)
#graph('year', 'length', datayear)
graph('year', 'valence', datayear)

#################### Modelling: ##############################################
# Simple linear regression to determine popularity
y = data['popularity']

X = data.drop(['Unnamed: 0', "name", "track_id",  "album", "artist", "artist_id", "type", "time_signature", "key", "year", "popularity"], axis=1)
X['constant'] = np.repeat(1, len(X['length']))
X['release_month'] = X['release_month'].astype(str)
Xdummy = pd.get_dummies(X)
Xdummy = Xdummy.drop(['release_month_1'], axis=1)
Xdummy = Xdummy.drop(['genre_8-bit'], axis=1)

train_features, test_features, train_labels, test_labels = train_test_split(Xdummy, y, test_size=0.25, random_state=7)

sm.OLS(train_labels, train_features).fit().summary()
model = sm.OLS(train_labels, train_features).fit()
predictions_OLS = model.predict(test_features)
errors_OLS = abs(predictions_OLS - test_labels)

plt.scatter(test_labels, predictions_OLS)
plt.title('OLS: True Values vs Fitted Values')
plt.xlabel('True Values')
plt.ylabel('Fitted Values')
MAE = np.zeros([3,1])
MAE[0,:] = (round(np.mean(errors_OLS), 2))
# Typical OLS model has trouble dealing with all the 0 rating

# Random Forest
feature_list = list(Xdummy.columns)


rf = RandomForestRegressor(n_estimators=1000, bootstrap=True, oob_score=True, n_jobs=-1, random_state=9)
rf.fit(train_features, train_labels)
rf.feature_importances_
plt.barh(feature_list, rf.feature_importances_)

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)

plt.scatter(test_labels, predictions)
plt.title('RandomForest: True Values vs Fitted Values')
plt.xlabel('True Values')
plt.ylabel('Fitted Values')
MAE[1,:] = (round(np.mean(errors), 2))

# Randomforest improves MAE from 16.45 to 14.13

# Boosting:
trees = np.arange(1,526, 25)
errors_trees = np.empty([21, 2])
errors_trees[:, 1] = np.arange(1, 526, 25)
for i in range(1, 526, 25):
    boost = XGBRegressor(n_estimators=i, learning_rate=0.1, max_depth=8, n_jobs=-1, random_state=9)
    boost.fit(train_features, train_labels)
    boost_predict = boost.predict(test_features)
    errors_boost = abs(boost_predict - test_labels)
    errors_trees[i,0] = (round(np.mean(errors_boost), 2))

boost = XGBRegressor(n_estimators=105, learning_rate=0.1, max_depth=8, n_jobs=-1, random_state=9)
boost.fit(train_features, train_labels)
boost_predict = boost.predict(test_features)
errors_boost = abs(boost_predict - test_labels)
MAE[2,:] = (round(np.mean(errors_boost), 2))


# OLS has MAE of 14.62, random forest of 12.97 and boosting of 12.92
# OLS does not pick followers as useful, the others do.
MAE

plt.scatter(test_labels, boost_predict)
plt.title('XGBoost: True Values vs Fitted Values')
plt.xlabel('True Values')
plt.ylabel('Fitted Values')