% PINN - Data Cleaning Script
folder_path = '/Users/professional_Soham/Documents/kepsiloncsv'; % Input folder
output_path = '/Users/professional_Soham/Documents/kepsiloncsv_cleaned'; % Output folder for cleaned data
mkdir(output_path); % Create output folder if not existing

csv_files = dir(fullfile(folder_path, '*.csv'));

for i = 1:length(csv_files)
    file_path = fullfile(folder_path, csv_files(i).name);
    data = readtable(file_path); % Read CSV as a table
    % Remove missing values (NaN or empty cells)
    data = rmmissing(data);
    
    % Convert non-numeric columns to numeric if possible
    for j = 1:width(data)
        if iscell(data{:, j}) % Check if column is cell array
            data{:, j} = str2double(data{:, j}); % Convert to numeric
        end
    end
    
    % Remove any remaining non-numeric columns
    numeric_data = varfun(@isnumeric, data, 'OutputFormat', 'uniform');
    data = data(:, numeric_data);
    
    % Save the cleaned dataset
    cleaned_filename = fullfile(output_path, ['cleaned_' csv_files(i).name]);
    writetable(data, cleaned_filename);    
end

fprintf('Data cleaning complete!\n');