
import pandas as pd
import ssl
#powerball
ssl._create_default_https_context = ssl._create_unverified_context

class LottoDataFrame():
    def __init__(self):
      pass
    def get_powerball(self):
        self.url = 'https://data.ny.gov/resource/d6yy-54nr.csv'

        # 1. Data Reading and Cleaning
        self.df = pd.read_csv(self.url)
        self.df['winning_numbers'] = self.df['winning_numbers'].str.split()
        self.df['pb'] = self.df['multiplier'].astype(int)

        for i in range(1, 6):  # Assuming there are 5 numbers in each winning set
            self.df[f'b{i}'] = self.df['winning_numbers'].str[i - 1]
        for i in range(1, 6):
            self.df[f'b{i}'] = pd.to_numeric(self.df[f'b{i}'], errors='coerce')
        print(self.df.head())
        self.df['date'] = pd.to_datetime(self.df['draw_date'])

        #Powerball
        ball_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'pb']
        self.df[ball_cols] = self.df[ball_cols].astype(int)
        return self.df
        
new_powerball = LottoDataFrame()

#print(new_powerball.get_powerball())