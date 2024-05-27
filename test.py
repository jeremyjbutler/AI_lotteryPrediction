import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import beta

#url = 'https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD'

#powerball
url = 'https://data.ny.gov/resource/d6yy-54nr.csv'

# 1. Data Reading and Cleaning
df = pd.read_csv(url)
print(df.head())

df['winning_numbers'] = df['winning_numbers'].str.split()

for i in range(1, 6):  # Assuming there are 5 numbers in each winning set
    df[f'b{i}'] = df['winning_numbers'].str[i - 1]
for i in range(1, 6):
    df[f'b{i}'] = pd.to_numeric(df[f'b{i}'], errors='coerce')
print(df.head())

#df.drop(columns=['Unnamed: 0', 'dp'], inplace=True)
df['date'] = pd.to_datetime(df['draw_date'])

#print(df.head())
#Powerball
ball_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'multiplier']
df[ball_cols] = df[ball_cols].astype(int)
print(df[ball_cols].head())
print(df.head())
#print(df['Winning Numbers'].head())
#MegaMillions
#ball_cols = ['Winning Numbers', 'Mega Ball', 'Multiplier']
#df[['b1', 'b2', 'b3', 'b4', 'b5']] = df['Winning Numbers'].apply(lambda x: pd.Series(list(x)))
#df['Winning Numbers'].apply(lambda x: pd.Series(list(str(x))))
#new_df = df[['date', 'b1', 'b2', 'b3', 'b4', 'b5', 'Mega Ball', 'Multiplier']]

#df[ball_cols] = df[ball_cols].astype(int)

