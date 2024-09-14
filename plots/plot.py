import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# Load both CSVs
y_data = pd.read_csv('/Users/bellavg/AIG_SIM/data/results/relative_size_diff_metric_scores.csv', index_col=0)  # CSV containing Y values
x_data = pd.read_csv('/Users/bellavg/AIG_SIM/data/results/rel_level_count_scores.csv', index_col=0)  # CSV containing X values

# Initialize lists to store statistics
all_x_vals = []
all_y_vals = []

# Confidence level and critical z-value for 95% confidence intervals
confidence_level = 0.95
z_critical = norm.ppf((1 + confidence_level) / 2)


# Fisher transformation and inverse transformation functions
def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z):
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


# Set up subplots: number of rows and columns, adjust based on your data
num_columns = len(x_data.columns)
rows = (
                   num_columns // 3) + 1 if num_columns % 3 != 0 else num_columns // 3  # Adjust grid size based on the number of columns
fig, axes = plt.subplots(nrows=rows, ncols=3,
                         figsize=(20, 4 * rows))  # Adjust figure size based on the number of subplots
axes = axes.flatten()  # Flatten to easily index subplots

# Define colors for each column (updated method)
colormap = plt.get_cmap('tab20')  # Use the new get_cmap method
num_colors = min(20, num_columns)  # Ensure you only access valid indices

# Loop through each column and create a subplot
for idx, column in enumerate(x_data.columns):
    x_vals = x_data[column]
    y_vals = y_data[column]

    # Collect all x and y values for the combined dataset
    all_x_vals.extend(x_vals)
    all_y_vals.extend(y_vals)

    # Use color cycling for cases where you have more than 20 columns
    color_idx = idx % num_colors  # Recycle colors if num_columns > 20

    # Scatter plot for each column
    axes[idx].scatter(x_vals, y_vals, marker='x', color=colormap(color_idx), label=column)

    # Fit a linear trendline for each subplot
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
    trendline = slope * x_vals + intercept

    # Compute the correlation coefficient
    correlation = np.corrcoef(x_vals, y_vals)[0, 1]

    # Fisher transformation and confidence interval for Pearson correlation
    n = len(x_vals)
    fisher_z_val = fisher_z(correlation)
    SE_z = 1 / np.sqrt(n - 3)

    z_lower = fisher_z_val - z_critical * SE_z
    z_upper = fisher_z_val + z_critical * SE_z

    # Transform back to correlation scale
    rho_lower = inverse_fisher_z(z_lower)
    rho_upper = inverse_fisher_z(z_upper)

    # Plot the trendline
    axes[idx].plot(x_vals, trendline, color='black', linestyle='--')

    # Add the correlation coefficient (ρ) and confidence interval as text inside the plot
    axes[idx].text(0.05, 0.8, f'$\\rho$ = {correlation:.2f}\nCI: [{rho_lower:.2f}, {rho_upper:.2f}]',
                   transform=axes[idx].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Set title with typewriter style
    axes[idx].set_title(column, fontsize=16, fontfamily='monospace')  # Title with larger font

    # Set axis labels with larger font size
    axes[idx].set_xlabel('Relative Level Count', fontsize=14)  # Larger x-axis label
    axes[idx].set_ylabel('Relative Optimizability Difference', fontsize=14)  # Larger y-axis label

    # Fix the axis limits
    axes[idx].set_xlim(0, 0.55)  # Fix x-axis from 0 to 0.55
    axes[idx].set_ylim(0, 0.65)  # Fix y-axis from 0 to 0.65

# Hide any unused subplots if the number of subplots doesn't fill the grid exactly
for i in range(num_columns, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout to avoid overlap
plt.tight_layout()

plt.savefig("./levels.png")
# Show the plot
plt.show()

# Convert collected data to numpy arrays for combined analysis
all_x_vals = np.array(all_x_vals)
all_y_vals = np.array(all_y_vals)

# Compute overall statistics for the combined data
combined_slope, combined_intercept, combined_r_value, combined_p_value, combined_std_err = stats.linregress(all_x_vals,
                                                                                                            all_y_vals)
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

# Output the overall statistics
print(f'Overall Pearson Correlation (ρ): {combined_rho:.2f}')
print(f'Overall p-value: {combined_p_value:.4f}')
print(f'Overall Correlation Confidence Interval: [{rho_lower_combined:.2f}, {rho_upper_combined:.2f}]')