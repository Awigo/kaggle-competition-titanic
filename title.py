import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

csv = 'data/test.csv'
df = pd.read_csv(csv)

# {'Mr.': 0, 'Mrs.': 1, 'Miss.': 2, 'Master.': 3, 'Don.': 4, 'Rev.': 5, 'Dr.': 6, 'Mme.': 7, 'Ms.': 8, 'Major.': 9, 'Lady.': 10, 'Sir.': 11, 'Mlle.': 12, 'Col.': 13, 'Capt.': 14, 'Countess.': 15, 'Jonkheer.': 16}

def title_to_number(row):
    title_map = {'Mr.': 0, 'Mrs.': 1, 'Miss.': 2, 'Master.': 3, 'Don.': 4, 'Rev.': 5, 'Dr.': 6, 'Mme.': 7, 'Ms.': 8, 'Major.': 9, 'Lady.': 10, 'Sir.': 11, 'Mlle.': 12, 'Col.': 13, 'Capt.': 14, 'Countess.': 15, 'Jonkheer.': 16, 'Dona.': 17}
    title = row.Title
    row.Title = title_map[title]
    return row


df = df.apply(title_to_number, axis='columns')

df.to_csv(csv, index=False)