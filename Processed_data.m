% PINN - Data Processing Script
folder_path = '/Users/professional_Soham/Documents/project_Matlab/kepsiloncsv_cleaned';
output_path = '/Users/professional_Soham/Documents/project_Matlab/kepsiloncsv_summary'; % Output folder for summaries
mkdir(output_path); % Create output folder if not existing

csv_files = dir(fullfile(folder_path, '*.csv'));

for i = 1:length(csv_files)
    file_path = fullfile(folder_path, csv_files(i).name);
    data = readtable(file_path); % Read CSV as a table   
    % Initialize cell array for summary statistics
    summary_stats = {};
    
    for j = 1:width(data)
        param_name = data.Properties.VariableNames{j};
        param_values = data{:, j}; % Extract column data
        
        % Check if column is numeric
        if ~isnumeric(param_values)
            fprintf('  Skipping non-numeric column: %s\n', param_name);
            continue;
        end
        
        % Compute statistics
        min_value = min(param_values);
        max_value = max(param_values);
        mean_value = mean(param_values);
        std_value = std(param_values);
        median_value = median(param_values);
        p25 = prctile(param_values, 25);
        p75 = prctile(param_values, 75);

        % Append results to summary_stats cell array
        summary_stats = [summary_stats; {param_name, min_value, max_value, mean_value, std_value, median_value, p25, p75}];
        
        fprintf('  %s: Min = %.6f, Max = %.6f, Mean = %.6f, Std = %.6f, Median = %.6f, P25 = %.6f, P75 = %.6f\n', ...
                param_name, min_value, max_value, mean_value, std_value, median_value, p25, p75);
    end
    
    % Convert summary_stats to a table
    summary_table = cell2table(summary_stats, 'VariableNames', ...
        {'Parameter', 'Min', 'Max', 'Mean', 'Std', 'Median', 'P25', 'P75'});

    % Save summary statistics to a CSV file
    summary_filename = fullfile(output_path, ['summary_' csv_files(i).name]);
    writetable(summary_table, summary_filename);
    
    fprintf('Saved summary: %s\n\n', summary_filename);
end

fprintf('Batch processing complete!\n');