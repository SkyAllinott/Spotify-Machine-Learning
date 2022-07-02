import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import sys
sys.path.append('G:/My Drive/Python Projects/Spotify API')
import Spotify_Functions as sf

# Set-up: ##################################################################
data = pd.read_csv('G:/My Drive/Python Projects/Spotify API/spotify.csv')
seed = 9

#############################################################################

################ Exploratory Visualization: ################################

sf.histogram('popularity', data)


# Visualizing by grouping on year:
data['year'] = data['release_year'].astype(int)
datayear = data.groupby('year').mean().reset_index()
sf.graph('year', 'danceability', datayear)
sf.graph('year', 'valence', datayear)

#################### Modelling: ##############################################
# Splitting data into train/test sets:
y = data['popularity']

X = data.drop(['Unnamed: 0', "name", "track_id",  "album", "artist", "artist_id", "type", "time_signature", "key", "year", "popularity"], axis=1)
X['constant'] = np.repeat(1, len(X['length']))
X['release_month'] = X['release_month'].astype(str)
Xdummy = pd.get_dummies(X)
Xdummy = Xdummy.drop(['release_month_1'], axis=1)
Xdummy = Xdummy.drop(['genre_8-bit'], axis=1)

train_features, test_features, train_labels, test_labels = train_test_split(Xdummy, y, test_size=0.25, random_state=seed)

# OLS Model:
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
# MAE of 15.54

# Random Forest:
feature_list = list(Xdummy.columns)

rf = RandomForestRegressor(n_jobs=-1, random_state=seed)
rf.fit(train_features, train_labels)
sigs = np.array(rf.feature_importances_)
ind = np.argpartition(sigs, -3)[-3:]
topvars = np.array(feature_list)[ind]
toptotal = np.vstack([topvars, np.array(sigs[ind])])
toptotal


predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)

plt.scatter(test_labels, predictions)
plt.title('RandomForest: True Values vs Fitted Values')
plt.xlabel('True Values')
plt.ylabel('Fitted Values')
MAE[1,:] = (round(np.mean(errors), 2))



# Boosting:
p = np.arange(0, 526,25)
errors_trees = np.zeros([526, 2])
errors_trees[:, 1] = np.arange(1, 527)
for i in range(1, 527, 25):
    boost = XGBRegressor(n_estimators=(i), learning_rate=0.05, max_depth=12, n_jobs=-1, random_state=seed, base_score=0)
    boost.fit(train_features, train_labels)
    boost_predict = boost.predict(test_features)
    errors_boost = abs(boost_predict - test_labels)
    errors_trees[(i-1),0] = (round(np.mean(errors_boost), 2))
errors = errors_trees[p,:]
indexes = np.where(errors[:,0] == np.amin(errors[:,0]))
print(errors[indexes])

boost = XGBRegressor(n_estimators=301, learning_rate=0.05, max_depth=12, n_jobs=-1, random_state=seed, base_score=0)
boost.fit(train_features, train_labels)
sigsBoost = np.array(rf.feature_importances_)
indBoost = np.argpartition(sigsBoost, -3)[-3:]
topvarsBoost = np.array(feature_list)[indBoost]
toptotalBoost = np.vstack([topvarsBoost, np.array(sigsBoost[ind])])
toptotal

boost_predict = boost.predict(test_features)
errors_boost = abs(boost_predict - test_labels)
MAE[2,:] = (round(np.mean(errors_boost), 2))


MAE
# As expected, trained XGBoost > untrained RF > untrained OLS

plt.scatter(test_labels, boost_predict)
plt.title('XGBoost: True Values vs Fitted Values')
plt.xlabel('True Values')
plt.ylabel('Fitted Values')
