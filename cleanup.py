import pandas as pd

df = pd.read_csv('train.csv')

names = pd.Series(df['Name'])

for i, name in enumerate(names):
    title = name.split(' ')[1]
    names[i] = int(hash(title))


