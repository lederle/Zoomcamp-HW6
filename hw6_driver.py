import hw6
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor

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

