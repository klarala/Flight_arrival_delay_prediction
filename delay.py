import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import joblib
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector


### Reading the dataset
df = pd.read_csv('flights.csv', low_memory=False)

#print(df.head())

### Removing diverted and cancelled flights, checking for missing values
#print(df.isnull().sum())
df = df[df.DIVERTED != 1]
df = df[df.CANCELLED != 1]
#print(df.isnull().sum())

### Features with departure information
df = df[["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF", "SCHEDULED_TIME", "DISTANCE", "SCHEDULED_ARRIVAL", "ARRIVAL_DELAY"]]

### Features without departure information
#df = df[["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "SCHEDULED_TIME", "DISTANCE", "SCHEDULED_ARRIVAL", "ARRIVAL_DELAY"]]
#df.head()

### fixing skew
#df.DEPARTURE_DELAY = df.DEPARTURE_DELAY ** (0.5)
df.DISTANCE = np.log(df.DISTANCE)
#print(df.skew())

### Encoding categorical data
le = LabelEncoder()
df.AIRLINE = le.fit_transform(df.AIRLINE)
df.ORIGIN_AIRPORT = le.fit_transform(df.ORIGIN_AIRPORT)
df.DESTINATION_AIRPORT = le.fit_transform(df.DESTINATION_AIRPORT)
df.FLIGHT_NUMBER = le.fit_transform(df.FLIGHT_NUMBER)
df.TAIL_NUMBER = le.fit_transform(df.TAIL_NUMBER)
print(df.head())

### Scaling data and creating training and target datasets
sc = StandardScaler()
ds_x = df.drop('ARRIVAL_DELAY', axis=1)
y = df.ARRIVAL_DELAY
dataset = sc.fit_transform(ds_x)
x = pd.DataFrame(dataset, columns = ds_x.columns)
#print(x)

### Splitting the dataset into train and test sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state=42)



            ##### Feature selection methods I tried #### I'm not sure they work with no errors anymore, because I have changed the code after trying them ###


# fs = SelectKBest(score_func=f_regression, k=5) # OR
# #fs = SelectKBest(score_func=mutual_info_regression, k='all')
# fs.fit(train_x, train_y)

# X_train_fs = fs.transform(train_x)
# X_test_fs = fs.transform(test_x)

# for i in range(len(fs.scores_)):
#     print('Feature %d: %f' % (i, fs.scores_[i]))

# plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
# plt.show()

# print(X_train_fs)


# from sklearn.feature_selection import RFE
# print("Recursive feature elimination")
# rfe = RFE(LinearRegression(), n_features_to_select = 5)
# rfe.fit(x,y)
# status = rfe.support_ #rfe.get_support()
# print(status)
# features = np.array(x.columns)
# print(features[status])
# print(rfe.transform(x))
# x_new = rfe.transform(x)


### The same as the one after model fitting
# model = SelectFromModel(gd, prefit=True)
# X_new = model.transform(x)
# selected_columns = selected_features.columns[selected_features.var() != 0]
# print(selected_columns)


# from sklearn.linear_model import LassoCV
# from sklearn.feature_selection import SelectFromModel
# clf = LassoCV().fit(x, y)
# importance = np.abs(clf.coef_)
# idx_third = importance.argsort()[-3]
# threshold = importance[idx_third] + 0.01
# idx_features = (-importance).argsort()[:10]
# feature_names = np.array(x.columns)
# name_features = np.array(feature_names)[idx_features]
# print('Selected features: {}'.format(name_features))



                                                    ##### Regression models I tried #####



# # DECISION TREE REGRESSOR
# print("Decision tree regressor")

# dt = DecisionTreeRegressor()
# dt.fit(train_x, train_y)
# dt_pred = dt.predict(test_x)

# print("R2_test: ", r2_score(test_y, dt_pred))
# print("R2_train: ", r2_score(train_y, dt.predict(train_x)))
# print("MAE: ", mean_absolute_error(test_y, dt_pred))
# print("MSE: ", mean_squared_error(test_y, dt_pred))
# print("RMSE: ", mean_squared_error(test_y, dt_pred, squared=False))


# # K-neighbors regressor
# print("K neighbors regressor")

# knn = KNeighborsRegressor(n_jobs=3)
# knn.fit(train_x, train_y)
# knn_pred = knn.predict(test_x)

# print("R2_test: ", r2_score(test_y, knn_pred))
# print("R2_train: ", r2_score(train_y, knn.predict(train_x)))
# print("MAE: ", mean_absolute_error(test_y, knn_pred))
# print("MSE: ", mean_squared_error(test_y, knn_pred))
# print("RMSE: ", mean_squared_error(test_y, knn_pred, squared=False))


#Linear regression
print("Linear regression")

lr = LinearRegression()
lr.fit(train_x, train_y)
lr_pred = lr.predict(test_x)

print("R2_test: ", r2_score(test_y, lr_pred))
print("R2_train: ", r2_score(train_y, lr.predict(train_x)))
print("MAE: ", mean_absolute_error(test_y, lr_pred))
print("MSE: ", mean_squared_error(test_y, lr_pred))
print("RMSE: ", mean_squared_error(test_y, lr_pred, squared=False))



# Random forest regressor
# print("Random forest regressor")

# rfr = RandomForestRegressor()
# rfr.fit(train_x, train_y)
# rfr_pred = rfr.predict(test_x)

# print("R2_test: ", r2_score(test_y, rfr_pred))
# print("R2_train: ", r2_score(train_y, rfr.predict(train_x)))
# print("MAE: ", mean_absolute_error(test_y, rfr_pred))
# print("MSE: ", mean_squared_error(test_y, rfr_pred))
# print("RMSE: ", mean_squared_error(test_y, rfr_pred, squared=False))


#Ada Boost regressor
# print("Ada boost regressor")

# ad = AdaBoostRegressor()
# ad.fit(train_x, train_y)
# ad_pred = ad.predict(test_x)

# print("R2_test: ", r2_score(test_y, ad_pred))
# print("R2_train: ", r2_score(train_y, ad.predict(train_x)))
# print("MAE: ", mean_absolute_error(test_y, ad_pred))
# print("MSE: ", mean_squared_error(test_y, ad_pred))
# print("RMSE: ", mean_squared_error(test_y, ad_pred, squared=False))


#Gradient Boosting Regressor
# print("Gradient boosting regressor")

# gd = GradientBoostingRegressor(learning_rate=0.3, alpha=0.3, n_estimators=100, verbose=1)
# gd.fit(train_x, train_y)
# gd_pred = gd.predict(test_x)

# print("R2_test: ", r2_score(test_y, gd_pred))
# print("R2_train: ", r2_score(train_y, gd.predict(train_x)))
# print("MAE: ", mean_absolute_error(test_y, gd_pred))
# print("MSE: ", mean_squared_error(test_y, gd_pred))
# print("RMSE: ", mean_squared_error(test_y, gd_pred, squared=False))

### Saving / loading model
joblib.dump(lr, "lr_model")
gd = joblib.load("lr_model")
#print("model loaded")

### cross-validation
#cv = cross_val_score(gd, x, y)
#print(gd, cv.mean())


### Selection and extraction of features
model = SelectFromModel(gd, prefit=True)
status = model.get_support()
print(status)
features = np.array(x.columns)
print(features[status])
print(model.transform(x))
x_new = model.transform(x)

### Creating new test and train sets based on the extracted features
train_x, test_x, train_y, test_y = train_test_split(x_new, y, test_size = 0.2, random_state=42)


##### Testing the performance of the models with selected features


# print("Gradient boosting regressor")

# gd = GradientBoostingRegressor(learning_rate=0.8, alpha=0.15, n_estimators=100, verbose=1)
# gd.fit(train_x, train_y)
# gd_pred = gd.predict(test_x)

# print("R2_test: ", r2_score(test_y, gd_pred))
# print("R2_train: ", r2_score(train_y, gd.predict(train_x)))
# print("MAE: ", mean_absolute_error(test_y, gd_pred))
# print("MSE: ", mean_squared_error(test_y, gd_pred))
# print("RMSE: ", mean_squared_error(test_y, gd_pred, squared=False))


print("Linear regression")

lr = LinearRegression()
lr.fit(train_x, train_y)
lr_pred = lr.predict(test_x)

print("R2_test: ", r2_score(test_y, lr_pred))
print("R2_train: ", r2_score(train_y, lr.predict(train_x)))
print("MAE: ", mean_absolute_error(test_y, lr_pred))
print("MSE: ", mean_squared_error(test_y, lr_pred))
print("RMSE: ", mean_squared_error(test_y, lr_pred, squared=False))

### loading / saving model
joblib.dump(lr, "linear_features")
gd = joblib.load("linear_features")
# print("model loaded")


### Cross-validation
# cv = cross_val_score(gd, x_new, y)
# print(gd, cv.mean())
    

### parameter tuning with grid search -- only for gradient boosting regressor!
# param_grid = {'learning_rate': [0.1, 0.5, 0.8, 0.85], 'alpha':[0.9, 0.4, 0.15, 0.1], 'loss':['mean', 'huber'], 'n_estimators': [100, 200, 500, 1000]}
# gcv_gd = GridSearchCV(gd, param_grid, cv=2, verbose=5, n_jobs=5)
# res = gcv_gd.fit(train_x, train_y)
# print(res.best_params_)
### learning_rate=0.8, alpha=0.15, n_estimators=1000, loss = mean


### Testing the performance of models after parameter tuning
# print("Gradient boosting regressor")

# gd = GradientBoostingRegressor(learning_rate=0.8, alpha=0.15, n_estimators=1000, verbose=1)
# gd.fit(train_x, train_y)
# gd_pred = gd.predict(test_x)

# print("R2_test: ", r2_score(test_y, gd_pred))
# print("R2_train: ", r2_score(train_y, gd.predict(train_x)))
# print("MAE: ", mean_absolute_error(test_y, gd_pred))
# print("MSE: ", mean_squared_error(test_y, gd_pred))
# print("RMSE: ", mean_squared_error(test_y, gd_pred, squared=False))

### Cross-validation
#cv = cross_val_score(gd, x, y, n_jobs=2)
#print(gd, cv.mean())
