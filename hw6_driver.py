import hw6

housing = hw6.load_housing_data()
hw6.fill_na(housing)
housing["median_house_value"] = hw6.transform_label(housing)
(train, y_train), (val, y_val), (test, y_test) = hw6.split_data(housing)
