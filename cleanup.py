import pandas as pd

df = pd.read_csv('train.csv')

tickets = pd.Series(df['Ticket'])
cleaned = []

for ticket in tickets:
    ticket_number = ticket.split(' ')[-1]
    cleaned.append(int(ticket_number))

df['Ticket'] = cleaned

df.to_csv('train.csv')



