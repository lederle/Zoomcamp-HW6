import hw6
from sklearn.feature_extraction import DictVectorizer

housing = hw6.load_housing_data()
hw6.fill_na(housing)
housing["median_house_value"] = hw6.transform_label(housing)
(train, y_train), (val, y_val), (test, y_test) = hw6.prepare_data(housing)
dv = DictVectorizer(sparse = False)
X_train = hw6.dict_vectorize(dv, train)
X_val = hw6.dict_vectorize(dv, val)
X_test = hw6.dict_vectorize(dv, test)
