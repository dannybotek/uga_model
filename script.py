#!/usr/bin/env python3

#imports
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from matplotlib import pyplot as plt




def main():
    print("Hello World!")
    # Dictionary to store dataframes
    dataframes = {}

    # List of file names for 'vix' structured files
    vix_files = ['vix_crude_oil_history.csv', 'vix_emerging_markets_history.csv', 'vix_history.csv']

    # List of file names for 'weekly' structured files
    weekly_files = ['weekly_gas_stocks.csv', 'weekly_iraq_imports_to_us.csv', 'weekly_saudi_imports_to_us.csv']

    # Read 'vix' structured files and resample to weekly frequency
    for file in vix_files:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        dataframes[file] = df.resample('W-FRI').mean()

    # Read 'weekly' structured files
    for file in weekly_files:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        dataframes[file] = df  # Already weekly frequency, so no resampling needed

    # Read UGA.csv which has the target variable and resample to weekly frequency
    uga_df = pd.read_csv('UGA.csv')
    uga_df['Date'] = pd.to_datetime(uga_df['Date'])
    uga_df.set_index('Date', inplace=True)
    dataframes['UGA.csv'] = uga_df.resample('W-FRI').last()  # Assuming last value is the price

    # Align and join the dataframes
    # Start with the UGA dataframe as it's our target
    aligned_data = dataframes['UGA.csv']

    # Join all other dataframes onto the UGA dataframe
    for file, df in dataframes.items():
        if file != 'UGA.csv':  # Avoid joining UGA.csv onto itself
            aligned_data = aligned_data.join(df, how='inner', rsuffix=f'_{file}')

    # Reset index to bring the Date back as a column
    aligned_data.reset_index(inplace=True)

    # Handle missing values, feature engineering, etc.
    # ...

    # Prepare the data for modeling
    # X = aligned_data.drop(['Date', 'Price'], axis=1)  # Assuming 'Price' is the target variable
    # y = aligned_data['Price']
    # ...

    # Modeling (split data, scale features, train model, evaluate model)
    # ...

    # Print aligned data to verify
    print(aligned_data.head())
    #aligned_data.to_csv('aligned_data.csv', index=False)

    # Define window sizes for rolling calculations
    rolling_windows = [4, 8, 12]  # 4, 8, and 12 weeks for example

    # Create rolling window features for Gasoline Stock and Crude Imports
    for window in rolling_windows:
        aligned_data[f'Gasoline_Stock_mean_{window}w'] = aligned_data['US Gasoline Stock KB'].rolling(window=window).mean()
        aligned_data[f'Iraq_Imports_mean_{window}w'] = aligned_data['Iraq->US Crude Imports KB/D'].rolling(window=window).mean()
        aligned_data[f'Saudi_Imports_mean_{window}w'] = aligned_data['Saudi->US Crude Imports KB/D'].rolling(window=window).mean()
        aligned_data[f'Gasoline_Stock_std_{window}w'] = aligned_data['US Gasoline Stock KB'].rolling(window=window).std()
        aligned_data[f'Iraq_Imports_std_{window}w'] = aligned_data['Iraq->US Crude Imports KB/D'].rolling(window=window).std()
        aligned_data[f'Saudi_Imports_std_{window}w'] = aligned_data['Saudi->US Crude Imports KB/D'].rolling(window=window).std()

    # Momentum indicators for VIX data
    momentum_periods = [1, 2, 4]  # 1, 2, and 4 weeks for example

    # Create momentum features (e.g., 1,2,4-week percent change)
    for period in momentum_periods:
        aligned_data[f'OVX_momentum_{period}w'] = aligned_data['OVX'].pct_change(periods=period) * 100
        aligned_data[f'EMVIX_momentum_{period}w'] = aligned_data['EMVIX'].pct_change(periods=period) * 100
        aligned_data[f'VIX_momentum_{period}w'] = aligned_data['VIX'].pct_change(periods=period) * 100

    # Drop NaN values that were created by rolling window and momentum calculations
    aligned_data.dropna(inplace=True)

    # Now split the data into features (X) and the target variable (y)
    # Prepare the data for modeling
    y = aligned_data['UGA']  # Target variable
    X = aligned_data.drop(['Date', 'UGA'], axis=1)  # Drop 'Date' and the target variable 'UGA'

    # Initialize the TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=5)

    # Create a pipeline that scales the features and then applies linear regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regression', LinearRegression())
    ])

    # Perform cross-validation
    mse_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = pipeline.predict(X_test)

        # Calculate and store the mean squared error
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    # Calculate root mean squared error (RMSE) from the MSE scores
    rmse_scores = np.sqrt(mse_scores)

    print("Cross-validation RMSE scores:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print("Standard deviation of RMSE:", rmse_scores.std())

    # Prepare the performance data for export
    performance_data = {
        'Fold': range(1, len(rmse_scores) + 1),
        'RMSE': rmse_scores
    }
    performance_df = pd.DataFrame(performance_data)

    # Add mean and standard deviation to the DataFrame
    performance_df.loc['mean'] = ['Mean', rmse_scores.mean()]
    performance_df.loc['std_dev'] = ['Std Dev', rmse_scores.std()]

    # Print the performance DataFrame
    print(performance_df)

    # Export the performance DataFrame to a CSV file
    performance_df.to_csv('model_performance.csv', index=False)


if __name__ == "__main__":
    main()


    '''
    # Plot actual vs. predicted prices for the entire dataset
    plt.figure(figsize=(12, 6))
    plt.plot(aligned_data['Date'], y, label='Actual UGA Price')
    plt.plot(aligned_data['Date'], y_pred, label='Predicted UGA Price', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('UGA Price')
    plt.title('Actual vs. Predicted UGA Prices')
    plt.legend()

    # Save the figure
    plt.savefig('uga_model/uga_price_prediction.png', dpi=300)

    plt.close()  # Close the plot to avoid displaying in the terminal
    '''
if __name__ == "__main__":
   main()