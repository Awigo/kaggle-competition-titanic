import pandas as pd
from sklearn.impute import SimpleImputer
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

csv = 'data/train.csv'
df = pd.read_csv(csv)

# print(df)

imputer = SimpleImputer()

imputed_age = pd.DataFrame(imputer.fit_transform(pd.DataFrame(df.Age)))
print(imputed_age)

imputed_age.to_csv('output.csv', index=False)