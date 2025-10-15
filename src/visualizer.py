import numpy as np
import matplotlib.pyplot as plt

# Define the theoretical Benford probabilities (P(d)) as a global constant
# 1 through 9
BENFORD_PROPORTIONS = [
    0.30103, 0.17609, 0.12494, 0.09691, 0.07918,
    0.06695, 0.05799, 0.05115, 0.04576
]
DIGITS = list(range(1, 10))

def plot_benford_histogram(observed_frequencies, mad_score, title):
    """
    Generates a professional histogram comparing observed vs. theoretical Benford frequencies.

    Args:
        observed_frequencies (list): The list of observed proportions for digits 1-9.
        mad_score (float): The calculated Mean Absolute Deviation score.
        title (str): The descriptive title for the plot (e.g., "C_2,3,1 Baseline Test").
    """
    
    # Set up the plot area
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(DIGITS)) 
    width = 0.8 # Width of the bars

    # 1. Plot the Observed Frequencies (Your Collatz Data)
    ax.bar(x_pos, observed_frequencies, width, color='#0077B6', label='Observed Sequence Frequency', alpha=0.7)

    # 2. Plot the Theoretical Benford Frequencies as a Line
    ax.plot(x_pos, BENFORD_PROPORTIONS, color='red', marker='o', linestyle='--', linewidth=2, label='Theoretical Benford Frequency')

    # Add the MAD Score to the plot for context
    ax.text(8.5, 0.33, f"MAD: {mad_score:.4f}", ha='right', va='top', fontsize=12, color='black', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Set Labels and Title (CRITICAL for research papers)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('First Digit ($d$)', fontsize=14)
    ax.set_ylabel('Proportion / Frequency', fontsize=14)
    
    # Format the X-axis to show the digits 1 through 9
    ax.set_xticks(x_pos)
    ax.set_xticklabels(DIGITS)
    
    # Set a consistent y-limit for easier visual comparison across different plots
    ax.set_ylim(0, 0.35) 
    ax.grid(axis='y', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure directly to your results folder
    plt.savefig(f"results/histogram_{title.replace(' ', '_').replace(':', '')}.png")
    plt.close(fig) # Close the plot to save memory