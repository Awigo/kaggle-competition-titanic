import pandas as pd
from sklearn.ensemble import RandomForestRegressor
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

features = ['Pclass', 'Title', 'Sex']

X = df[features]
test_X = test_df[features]
y = df.Survived

model = RandomForestRegressor(random_state=0)
model.fit(X, y)

predicted = pd.Series(model.predict(test_X).round().astype('int'))

output = pd.DataFrame({'PassengerId': test_df.PassengerId,
                       'Survived': predicted})

output.to_csv('output.csv', index=False)
