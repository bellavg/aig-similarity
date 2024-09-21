import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import os

# Directory containing the CSV files
directory = '../data/results/'

# Load the reference CSV (this will always be used as Y data)
y_data = pd.read_csv(os.path.join(directory, 'relative_size_diff_metric_scores.csv'), index_col=0)

# Confidence level and critical z-value for 95% confidence intervals
confidence_level = 0.95
z_critical = norm.ppf((1 + confidence_level) / 2)

# Fisher transformation and inverse transformation functions
def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

# Get all CSV files in the directory, excluding the reference file itself
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') and not f.startswith("relative_size_diff_metric_scores")]

# Process each CSV file and compute correlations against the reference file
for csv_file in csv_files:
    file_path = os.path.join(directory, csv_file)

    # Load the X data (current CSV file)
    try:
        x_data = pd.read_csv(file_path, index_col=0)

        # Ensure that the columns between x_data and y_data are aligned and of equal length
        common_columns = x_data.columns.intersection(y_data.columns)

        if common_columns.empty:
            print(f"No common columns found for {csv_file} and reference file.")
            continue

        # Initialize lists to store statistics for combined analysis
        all_x_vals = []
        all_y_vals = []

        # Loop through each common column to compute the correlation
        for column in common_columns:
            x_vals = x_data[column]
            y_vals = y_data[column]

            # Collect all x and y values for the combined dataset
            all_x_vals.extend(x_vals)
            all_y_vals.extend(y_vals)

        # Convert collected data to numpy arrays for combined analysis
        all_x_vals = np.array(all_x_vals)
        all_y_vals = np.array(all_y_vals)

        # Compute overall statistics for the combined data
        combined_slope, combined_intercept, combined_r_value, combined_p_value, combined_std_err = stats.linregress(all_x_vals, all_y_vals)
        combined_rho = np.corrcoef(all_x_vals, all_y_vals)[0, 1]

        # Fisher transformation and confidence interval for combined correlation
        n_combined = len(all_x_vals)
        fisher_z_combined = fisher_z(combined_rho)
        SE_z_combined = 1 / np.sqrt(n_combined - 3)

        z_lower_combined = fisher_z_combined - z_critical * SE_z_combined
        z_upper_combined = fisher_z_combined + z_critical * SE_z_combined

        # Transform back to correlation scale for combined data
        rho_lower_combined = inverse_fisher_z(z_lower_combined)
        rho_upper_combined = inverse_fisher_z(z_upper_combined)

        # Output the overall statistics for the current file
        print(f'File: {csv_file}')
        print(f'Overall Pearson Correlation (ρ): {combined_rho:.2f}')
        print(f'Overall p-value: {combined_p_value:.4f}')
        print(f'Overall Correlation Confidence Interval: [{rho_lower_combined:.2f}, {rho_upper_combined:.2f}]')
        print('--------------------------------------------------')

    except Exception as e:
        print(f"Failed to process {csv_file}: {str(e)}")
