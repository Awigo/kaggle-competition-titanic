import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("data/train.csv")

features = ['Pclass', 'Title', 'Sex']

X = df[features]
y = df.Survived

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)
model_5 = RandomForestRegressor(n_estimators=2000, criterion='absolute_error', random_state=0)
models = [model_1, model_2, model_3, model_4, model_5]


def calculate_mae(model):
    model.fit(train_X, train_y)
    predicted = pd.Series(model.predict(val_X).round().astype('int'))
    return mean_absolute_error(val_y, predicted)


for m in models:
    print(calculate_mae(m))
