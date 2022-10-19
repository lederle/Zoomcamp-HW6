import hw6
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt


df = hw6.load_housing_data()
housing = df.copy()
hw6.fill_na(housing)
housing["median_house_value"] = hw6.transform_label(housing)
(train, y_train), (val, y_val), (test, y_test) = hw6.prepare_data(housing)
dv = DictVectorizer(sparse = False)
dv.fit(hw6.rows_as_dict(housing))
X_train = hw6.dict_vectorize(dv, train)
X_val = hw6.dict_vectorize(dv, val)
X_test = hw6.dict_vectorize(dv, test)

# Question 1
dtr = DecisionTreeRegressor(max_depth = 1)
hw6.tree_regression(dtr, dv, X_train, y_train)

# Question 2
rf = RandomForestRegressor(n_estimators = 10, random_state = 1, n_jobs = -1)
hw6.random_forest_regression(rf, X_train, y_train, X_val, y_val)

# Question 3
if False:
    scores_df = hw6.optimize_N_estimators(X_train, y_train, X_val, y_val)
    print(scores_df)

# Question 4
df_scores = hw6.optimize_max_depth(X_train, y_train, X_val, y_val)

for d in [10, 15, 20, 25]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plot(df_subset["n_estimator"], df_subset["rmse"], label = d)

plt.legend()
hw6.save_fig("max_depth_plot")
