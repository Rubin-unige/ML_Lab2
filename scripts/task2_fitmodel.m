%% Task 2: Fit a Linear Regression Model

function task2_fitmodel(turkishStockData, mtCarsData)
    %% 1. One-dimensional problem without intercept on the Turkish stock exchange data

    % Prepare Turkish stock data
    x_turkish = turkishStockData.SP500; % Input (predictor)
    t_turkish = turkishStockData.MSCI;  % Actual target (response)
    % Prediction using all the given data
    w_all = sum(x_turkish .* t_turkish) / sum(x_turkish .^ 2); % Slope using all data
    y_all = w_all * x_turkish; % Predicted target using all the data

    % Plot the Results
    figure;
    scatter(x_turkish, t_turkish, 'blue', 'x', 'DisplayName', 'Actual Data'); % Scatter plot of actual data
    hold on;
    % Plot the regression line
    plot(x_turkish, y_all, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Linear Regression (All Data)');
    xlabel('Standard and Poors 500 Return Index');
    ylabel('MSCI Europe Index');
    title('Linear Regression Without Intercept: SP500 vs MSCI');
    legend('show');
    grid on;
    hold off;
    saveas(gcf, 'result/turkish_stock_regression.png');

    %% 2. Compare graphically the solution obtained on different random subsets (10%) of the whole data set

    % Set the size of the subset to use
    m = floor(0.1 * height(turkishStockData)); % 10% of the data size
   
    % Extract the first and last subsets of the data
    x_start = x_turkish(1:m); % First 10% of the data
    t_start = t_turkish(1:m); 

    x_end = x_turkish(end-m+1:end); % Last 10% of the data
    t_end = t_turkish(end-m+1:end); 

    % Generate a random subset
    randomIndices = randperm(height(turkishStockData), m); % Randomly select m unique indices
    x_random = x_turkish(randomIndices); % Random subset of x
    t_random = t_turkish(randomIndices); % Random subset of t

    % Calculate slope for the first subset
    w_start = sum(x_start .* t_start) / sum(x_start .^ 2); % Slope for starting subset
    % Calculate slope for the last subset
    w_end = sum(x_end .* t_end) / sum(x_end .^ 2); % Slope for ending subset
    % Calculate slope for the random subset
    w_random = sum(x_random .* t_random) / sum(x_random .^ 2); % Slope for random subset

    % Generate a full range for plotting the regression lines
    x_full_range = linspace(min(x_turkish), max(x_turkish), 100); % Full range of x values

    % Calculate the predicted values for the full range using the slopes
    y_start_full = w_start * x_full_range; % Predicted target for starting subset
    y_end_full = w_end * x_full_range;     % Predicted target for ending subset
    y_random_full = w_random * x_full_range; % Predicted target for random subset

    % Plot the result
    figure;
    scatter(x_turkish, t_turkish, 'blue', 'x', 'DisplayName', 'Actual Data'); % Scatter plot of actual data
    hold on;
    % Plot regression lines for the starting, ending, and random subsets
    plot(x_full_range, y_start_full, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Linear Regression (Start of Data)');
    plot(x_full_range, y_end_full, 'm-', 'LineWidth', 1.5, 'DisplayName', 'Linear Regression (End of Data)');
    plot(x_full_range, y_random_full, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Linear Regression (Random Subset)');
    % Plot the full dataset regression line for comparison
    plot(x_turkish, y_all, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Linear Regression (All Data)');
    % Customize plot
    xlabel('Standard and Poors 500 Return Index');
    ylabel('MSCI Europe Index');
    title('Comparison of Linear Regression Models from Different Subsets');
    legend('show');
    grid on;
    hold off;
    saveas(gcf, 'result/turkish_stock_subsets_comparison.png');

    %% 3. One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight

    % Prepare Motor trend data 
    t_mpg = mtCarsData.mpg;          % Actual MPG data
    x_weight = mtCarsData.weight;    % Weight of the cars

    % Calculate mean of mpg and weight
    xbar_weight = mean(x_weight);    % Mean of weights
    tbar_mpg = mean(t_mpg);          % Mean of MPG

    % Calculate slope (w1) and intercept (w0)
    w1 = sum((x_weight - xbar_weight) .* (t_mpg - tbar_mpg)) / sum((x_weight - xbar_weight) .^ 2); % Slope
    w0 = tbar_mpg - w1 * xbar_weight; % Intercept

    % Make predictions using the model
    y_mpg_pred_weight = w1 * x_weight + w0;  % Predicted MPG values

    % Create scatter plot to show the result
    figure;
    scatter(x_weight, t_mpg, 'blue', 'x', 'DisplayName', 'Motor Trend Data'); % Scatter plot of actual data
    xlabel('Car Weight (lbs)');  
    ylabel('Fuel Efficiency (mpg)'); 
    title('Motor Trends Survey: Car MPG as a Function of Weight');  
    grid on;  
    hold on;  
    % Plot linear regression line
    plot(x_weight, y_mpg_pred_weight, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Linear Regression'); % Regression line
    legend('Motor Trend Data', 'Linear Regression'); 
    hold off;  
    saveas(gcf, 'result/mtcars_weight_regression.png');

    %% 4. Multi-dimensional problem on the complete MTcars data, using all four columns 

    % Prepare the data for predictions
    X_input_mat = [ones(height(mtCarsData), 1), mtCarsData.weight, mtCarsData.disp, mtCarsData.hp]; 
    t_mpg = mtCarsData.mpg;

    % Calculate weights
    w_slope_mat = (X_input_mat' * X_input_mat) \ (X_input_mat' * t_mpg);

    % Create predictions
    y_pred_mpg = X_input_mat * w_slope_mat;

    % Create a table for actual vs predicted including individual columns
    resultsTable = table(mtCarsData.Model, t_mpg, y_pred_mpg, ...
        'VariableNames', {'Car_Model', 'Actual_MPG', 'Predicted_MPG_all_input'});
    
    % Save the results table to a CSV file
    writetable(resultsTable, 'result/mtcars_actual_vs_predicted.csv');

    % Create a figure to visualize actual vs predicted MPG
    figure;
    plot(1:height(mtCarsData), t_mpg, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Actual MPG');
    hold on;
    plot(1:height(mtCarsData), y_pred_mpg, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Predicted MPG');
    xlabel('Car Index');
    ylabel('MPG');
    title('Actual vs Predicted MPG');
    legend('show');
    grid on;
    hold off;
    saveas(gcf, 'result/mtcars_actual_vs_predicted.png');
end
