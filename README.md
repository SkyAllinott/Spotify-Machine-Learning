# Spotify Machine Learning
## Overview
This is my first forray into the Spotify API. Using a playlist of all my music, I pull artist and track information from the API and assemble a dataset. Next I use the following three machine learning methods to predict song popularity:
1. Untrained OLS
2. Untrained RandomForest
3. Trained XGBoost

## Results
As expected, the XGBoost (only trained model), performed the best. The following table provides the mean squared error (MAE) for each model type in the testing set:

| Model | MAE |
| :----:  | --- |
| OLS   | 15.54|
| RandomForest | 14.11 |
| XGBoost | 13.36 |

With popularity being bounded between 0 and 100, these errors are relatively large. The models struggled with the binomial distribution of the dependant variable; popularity.

![distribution_y](https://user-images.githubusercontent.com/52394699/177019386-e07ef75a-f605-473d-b508-5c6417eec91a.png)

This lead to all of the models (XGBoost displayed) performing poorly on songs with a 0 popularity:

![XGBoost](https://user-images.githubusercontent.com/52394699/177019421-ef7748f1-1d82-429f-a262-8573a46de32c.png)

Disregarding all popularity=0 in the testing set lowers the MAE's fairly significantly:

| Model | MAE |
| :----:  | --- |
| OLS   | 13.54|
| RandomForest | 12.02 |
| XGBoost | 11.63 |

## Variable Importance
The variables for each song fall into 3 categories:
1. **Meta**: this includes release date, length, and whether or not it is explicit.
2. **Features**: this includes measures provided by Spotify, such as 'valence' (measure of the songs happiness), 'energy', danceability, etc.
3. **Artist features**: this includes information about the artist, such as their followers and primary genre.

For OLS, the most significant variables were instrumentalness, release year, length, month of June, and various genres such as alternative country, hip hop, and metal.

For randomForest and XGBoost, the 3 most important variables were release year, instrumentalness, and artist followers. Artist followers in particular was extremely important in randomForest and XGBoost, but was insigificant in the OLS model.





