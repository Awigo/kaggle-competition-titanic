import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('train.csv')

# variables
target = ['Survived']
features = ['Pclass', 'Fare', 'SibSp', 'Parch', 'Name', 'Ticket']

X = train_data[features]
y = train_data[target]

train_X, val_X, train_y, val_y = train_test_split(X, y)

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

predicted = model.predict(val_X)

avrMae = 0
for i in range(100000):
    avrMae += mean_absolute_error(val_y, predicted)
avrMae /= 100000
print(avrMae)

# Without ticket number
#0.214
#0.257
#0.201

# With ticket number
#0.185
#0.205
#0.183