#!/usr/bin/env python3

#imports
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler




def main():
    print("Hello World!")
    # Dictionary to store dataframes
    dataframes = {}

    # List of file names for 'vix' structured files
    vix_files = ['vix_crude_oil_history.csv', 'vix_emerging_markets_history.csv', 'vix_history.csv']

    # List of file names for 'weekly' structured files
    weekly_files = ['weekly_gas_stocks.csv', 'weekly_iraq_imports_to_us.csv', 'weekly_saudi_imports_to_us.csv']

    # Read 'vix' structured files
    for file in vix_files:
        dataframes[file] = pd.read_csv(file)


    # Read 'weekly' structured files
    for file in weekly_files:
        dataframes[file] = pd.read_csv(file)


    # Read UGA.csv which follows a different structure
    dataframes['UGA.csv'] = pd.read_csv('UGA.csv')
    dataframes['nyh_spot_price.csv'] = pd.read_csv('nyh_spot_price.csv')

    # Convert all date columns to datetime
    for df in dataframes.values():
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    #Change data with daily frequency to weekly frequency
    dataframes['UGA.csv'] = dataframes['UGA.csv'].resample('W-FRI').mean()

    for vix in vix_files:
        dataframes[vix] = dataframes['UGA.csv'].resample('W-FRI').mean()

    # Now, align all other weekly datasets to the same weekly frequency as VIX
    # We will use an inner join to only keep rows with matching dates across all datasets
    aligned_data = dataframes[vix]
    for file, df in dataframes.items():
        aligned_data = aligned_data.join(df, how='inner', rsuffix=f'_{file}')

    # Reset index to bring the Date back as a column
    aligned_data.reset_index(inplace=True)

    # Now, 'aligned_data' is a DataFrame that contains weekly data with all features aligned by date

    print(aligned_data.head())


    #Print heads just to make sure everything looks good
    #for file, df in dataframes.items():
       # print(f"\nFirst few rows of {file}:")
        #print(df.head())



    

if __name__ == "__main__":
    main()