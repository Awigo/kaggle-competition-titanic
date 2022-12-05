import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('train.csv')

# variables
target = ['Survived']

names = pd.Series(train_data['Name'])

for i, name in enumerate(names):
    title = name.split(' ')[1]
    names[i] = int(hash(title))

features = ['Pclass', 'Fare', 'SibSp', 'Parch', 'Name']

X = train_data[features]
y = train_data[target]

train_X, val_X, train_y, val_y = train_test_split(X, y)

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

predicted = model.predict(val_X)

mae = mean_absolute_error(val_y, predicted)

print(mae)

