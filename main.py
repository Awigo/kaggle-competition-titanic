import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

train_data = pd.read_csv('train.csv')
validation_data = pd.read_csv('test.csv')

# target setup
train_y = train_data.Survived

# column names prediction will base on
features = ['Pclass', 'Fare', 'SibSp', 'Parch', 'Name', 'Ticket']
train_X = train_data[features]
val_X = validation_data[features]

model = RandomForestClassifier(random_state=1)
model.fit(train_X, train_y)

prediction = model.predict(val_X)
# prediction = [round(num) for num in prediction]

df = pd.DataFrame({'PassengerId' : validation_data['PassengerId'],
    'Survived' : prediction})

df.to_csv('out.csv',  index=False)

from sklearn import tree
tree.export_graphviz(model, out_file='visualization.dot',
                     feature_names=features,
                     label='all',
                     rounded=True,
                     filled=True)
