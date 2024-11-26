%% Main script for the machine learning assignment

% Add the scripts folder where task functions are located
addpath('scripts');  

% Task 1: Get data
[turkishStockData, mtCarsData] = task1_getdata(); % Call the function to get data

% Task 2: Fit a linear regression model
task2_fitmodel(turkishStockData, mtCarsData); % Call the regression fitting function

% Task 3: Test regression model
task3_testmodel(turkishStockData, mtCarsData); % Call the function to test the regression model