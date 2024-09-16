import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from scipy.stats import norm

# Load your data
y_data = pd.read_csv('../data/results/relative_size_diff_metric_scores.csv', index_col=0)  # CSV containing Y values
x_data = pd.read_csv('../data/results/relative_resub_metric_scores.csv', index_col=0)  # CSV containing X values


# Fisher transformation and inverse transformation functions
def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

# Confidence level and critical z-value for 95% confidence intervals
confidence_level = 0.95
z_critical = norm.ppf((1 + confidence_level) / 2)

# Select columns 5 and 6
selected_columns = [x_data.columns[4], x_data.columns[5]]

# Initialize subplots: 2 columns for the two selected columns
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # Adjust figure size for two plots

# Define colors for the two plots
colormap = plt.get_cmap('tab10')  # Color map

colors = [colormap(0), colormap(2)]
# Loop through selected columns and plot
for idx, column in enumerate(selected_columns):
    x_vals = x_data[column]
    y_vals = y_data[column]

    # Scatter plot for each column
    axes[idx].scatter(x_vals, y_vals, marker='x', color=colors[idx], label=column)

    # Fit a linear trendline
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
    axes[idx].text(0.05, 0.875, f'$\\rho$ = {correlation:.2f}\nCI: [{rho_lower:.2f}, {rho_upper:.2f}]',
                   transform=axes[idx].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Set title with typewriter style
    axes[idx].set_title(column, fontsize=16, fontfamily='monospace')  # Title with larger font

    # Set axis labels
    axes[idx].set_xlabel('Resub Score Value', fontsize=14)
    axes[idx].set_ylabel('Relative Optimizability Difference', fontsize=14)

    # Set axis limits (optional, can be adjusted based on your data)
    axes[idx].set_xlim(0, 0.5)
    axes[idx].set_ylim(0, 0.65)

plt.tight_layout()

# Save the figure
plt.savefig("./paper_fig_resub.png")

# Show the plot
plt.show()
