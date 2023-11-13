#!/usr/bin/env python3
import pandas as pd

# Read the CSV file
df = pd.read_csv('UGA.csv')

# Select only the 'Date' and 'CLOSE' columns
df = df[['Date', 'Close']]

#convert to csv
#df.to_csv('UGA.csv', index=False)

df.head()  # Show the first few rows of the filtered DataFrame