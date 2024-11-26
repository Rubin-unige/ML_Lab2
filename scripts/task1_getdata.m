%% Task 1: Get Data

function [turkishStockData, mtCarsData] = task1_getdata()
    
    % Load turkish stock exchange data
    turkishStockData = readtable('data/turkish-se-SP500vsMSCI.csv');
    turkishStockData.Properties.VariableNames = {'SP500', 'MSCI'}; % Assign variable names 

    % Load mt cars data with 4 features
    mtCarsData = readtable('data/mtcarsdata-4features.csv'); 
    
end
