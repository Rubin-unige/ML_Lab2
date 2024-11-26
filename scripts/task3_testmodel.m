%% Task 3: Test regression model for Turkish Stock Data only

function task3_testmodel(turkishStockData, mtCarsData)
    % Set random seed for reproducibility
    rng(1);

    % Define number of trials and training percentages
    num_trials = 10;
    train_percent_turkish = 0.05; % 5% for Turkish Stock Data
    % 5% for mt car data is too low, it will not train proper model
    train_percent_cars = 0.15; % 15% for mtCars Data

    % Initialize arrays to store MSE results
    mse_turkish_train = zeros(num_trials, 1);
    mse_turkish_test = zeros(num_trials, 1);
    mse_cars_train = zeros(num_trials, 1);
    mse_cars_test = zeros(num_trials, 1);
    mse_multi_train = zeros(num_trials, 1);
    mse_multi_test = zeros(num_trials, 1);

    % Get number of observations in each dataset
    n_turkish = height(turkishStockData);
    n_cars = height(mtCarsData);

    % Run trials
    for trial = 1:num_trials    
    %% Model 1: One-dimensional regression without intercept (Turkish Stock Data)
        num_samples_turkish = round(train_percent_turkish * n_turkish);  % 5% training samples
        idx_turkish_train = randperm(n_turkish, num_samples_turkish); % Random training indices
        idx_turkish_test = setdiff(1:n_turkish, idx_turkish_train); % Remaining indices for testing

        % Split training and testing data
        x_train_turkish = turkishStockData.SP500(idx_turkish_train);
        t_train_turkish = turkishStockData.MSCI(idx_turkish_train);
        x_test_turkish = turkishStockData.SP500(idx_turkish_test);
        t_test_turkish = turkishStockData.MSCI(idx_turkish_test);

        % Slope calculation (no intercept)
        w_train_turkish = sum(x_train_turkish .* t_train_turkish) / sum(x_train_turkish .^ 2);
        y_train_turkish = w_train_turkish * x_train_turkish;
        y_test_turkish = w_train_turkish * x_test_turkish;

        % Calculate MSE
        mse_turkish_train(trial) = mean((y_train_turkish - t_train_turkish).^2);
        mse_turkish_test(trial) = mean((y_test_turkish - t_test_turkish).^2);

    %% Model 2: One-dimensional regression with intercept (Motor Trends Car Data)
        num_samples_cars = round(train_percent_cars * n_cars);  % 15% training samples
        idx_cars_train = randperm(n_cars, num_samples_cars); % Random training indices
        idx_cars_test = setdiff(1:n_cars, idx_cars_train); % Remaining indices for testing

        % Split training and testing data
        x_train_cars = mtCarsData.weight(idx_cars_train);
        t_train_cars = mtCarsData.mpg(idx_cars_train);
        x_test_cars = mtCarsData.weight(idx_cars_test);
        t_test_cars = mtCarsData.mpg(idx_cars_test);

        % Calculate slope (w1) and intercept (w0)
        xbar = mean(x_train_cars);
        tbar = mean(t_train_cars);
        w1 = sum((x_train_cars - xbar) .* (t_train_cars - tbar)) / sum((x_train_cars - xbar).^2);
        w0 = tbar - w1 * xbar;

        % Predictions
        y_train_cars = w1 * x_train_cars + w0;
        y_test_cars = w1 * x_test_cars + w0;

        % Calculate MSE
        mse_cars_train(trial) = mean((y_train_cars - t_train_cars).^2);
        mse_cars_test(trial) = mean((y_test_cars - t_test_cars).^2);

    %% Model 3: Multi-dimensional regression (mpg ~ weight + disp + hp)
        X_train_multi = [ones(num_samples_cars, 1), ...
                         mtCarsData.weight(idx_cars_train), ...
                         mtCarsData.disp(idx_cars_train), ...
                         mtCarsData.hp(idx_cars_train)];
        y_train_multi = mtCarsData.mpg(idx_cars_train);

        % Calculate regression weights
        w_multi = (X_train_multi' * X_train_multi) \ (X_train_multi' * y_train_multi);

        % Predictions on training data
        y_train_pred_multi = X_train_multi * w_multi;
        mse_multi_train(trial) = mean((y_train_pred_multi - y_train_multi).^2);

        % Predictions on testing data
        X_test_multi = [ones(length(idx_cars_test), 1), ...
                        mtCarsData.weight(idx_cars_test), ...
                        mtCarsData.disp(idx_cars_test), ...
                        mtCarsData.hp(idx_cars_test)];
        y_test_multi = mtCarsData.mpg(idx_cars_test);

        y_test_pred_multi = X_test_multi * w_multi;
        mse_multi_test(trial) = mean((y_test_pred_multi - y_test_multi).^2);
    end

    %% Display result using table and plots
    
    % Table for Turkish Stock Data Model
    turkish_table = table((1:num_trials)', mse_turkish_train, mse_turkish_test, ...
                          'VariableNames', {'Trial', 'MSE_Turkish_Train', 'MSE_Turkish_Test'});
    disp('Turkish Stock Data Model Results:');
    disp(turkish_table);
    
    % Table for MT Cars One-Dimensional Model (weight vs mpg)
    cars_table = table((1:num_trials)', mse_cars_train, mse_cars_test, ...
                       'VariableNames', {'Trial', 'MSE_Cars_Train', 'MSE_Cars_Test'});
    disp('MT Cars One-Dimensional Model Results (weight vs mpg):');
    disp(cars_table);
    
    % Table for MT Cars Multi-Dimensional Model
    multi_table = table((1:num_trials)', mse_multi_train, mse_multi_test, ...
                        'VariableNames', {'Trial', 'MSE_Multi_Train', 'MSE_Multi_Test'});
    disp('MT Cars Multi-Dimensional Model Results:');
    disp(multi_table);

    % Save the results table to a CSV file
    writetable(turkish_table, 'result/turkishstockexchange_iterations.csv');
    writetable(cars_table, 'result/weightvsmpg_iterations.csv');
    writetable(multi_table, 'result/multidimensionalproblem_iterations.csv');

    fprintf('Results saved in table".\n');


    % Plot results for Turkish Stock Data Model
    figure;
    plot(1:num_trials, mse_turkish_train, 'b-o', 'DisplayName', 'MSE Turkish Train');
    hold on;
    plot(1:num_trials, mse_turkish_test, 'g-o', 'DisplayName', 'MSE Turkish Test');
    xlabel('Trial Number');
    ylabel('Mean Squared Error');
    title('MSE for Turkish Stock Data Model Across Trials');
    legend('Location', 'Best');
    grid on;
    hold off;
    saveas(gcf, 'result/mse_turkish_stock_data.png');
    
    % Plot results for MT Cars One-Dimensional Model (weight vs mpg)
    figure;
    plot(1:num_trials, mse_cars_train, 'r-o', 'DisplayName', 'MSE Cars Train');
    hold on;
    plot(1:num_trials, mse_cars_test, 'm-o', 'DisplayName', 'MSE Cars Test');
    xlabel('Trial Number');
    ylabel('Mean Squared Error');
    title('MSE for MT Cars One-Dimensional Model Across Trials');
    legend('Location', 'Best');
    grid on;
    hold off;
    saveas(gcf, 'result/mse_cars_weight.png');
    
    % Plot results for MT Cars Multi-Dimensional Model
    figure;
    plot(1:num_trials, mse_multi_train, 'c-o', 'DisplayName', 'MSE Multi Train');
    hold on;
    plot(1:num_trials, mse_multi_test, 'k-o', 'DisplayName', 'MSE Multi Test');
    xlabel('Trial Number');
    ylabel('Mean Squared Error');
    title('MSE for MT Cars Multi-Dimensional Model Across Trials');
    legend('Location', 'Best');
    grid on;
    hold off;
    saveas(gcf, 'result/mse_cars_multidimensional.png');
end
