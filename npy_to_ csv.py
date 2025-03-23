import numpy as np
import os

# Define input and output folders
input_folder = "/Users/professional_Soham/Downloads/kepsilon"
output_folder = "/Users/professional_Soham/Documents/kepsiloncsv"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all .npy files in the folder
for file in os.listdir(input_folder):
    if file.endswith(".npy"):  
        file_path = os.path.join(input_folder, file)

        # Try loading the .npy file
        try:
            data = np.load(file_path, allow_pickle=True)  # Allow pickled data
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue  # Skip this file if loading fails

        # Check if data is an array or an object
        if isinstance(data, np.ndarray):
            if data.dtype == object:  # Convert object arrays to strings first
                try:
                    data = data.astype(float)  # Try converting to numbers
                except ValueError:
                    print(f"Skipping {file}: Contains non-numeric data.")
                    continue  # Skip non-numeric files

            if data.ndim == 0:  # If scalar (0D), reshape to 1×1 (2D)
                data = np.array([[data]])
            elif data.ndim == 1:  # If 1D, reshape to 2D column
                data = data.reshape(-1, 1)
            elif data.ndim == 3:  # If 3D, flatten last two dimensions
                data = data.reshape(data.shape[0], -1)
            elif data.ndim > 3:  # Skip higher dimensions
                print(f"Skipping {file}: Unsupported {data.ndim}D array")
                continue
        else:
            print(f"Skipping {file}: Not a NumPy array (Type: {type(data)})")
            continue

        # Convert file extension to .csv
        csv_filename = file.replace(".npy", ".csv")
        csv_path = os.path.join(output_folder, csv_filename)

        # Save as CSV
        np.savetxt(csv_path, data, delimiter=",", fmt="%.6f")

        print(f"Converted: {file_path} -> {csv_path}")

print("Batch conversion complete! ✅")