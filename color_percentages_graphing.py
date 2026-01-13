import pandas as pd
import matplotlib.pyplot as plt
from corner_detection.corner_detection import RobotCornerDetection

def makeGraph():
    # Read the CSV file
    df = pd.read_csv("ColorPercentageData.csv")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define color pairs for each column (two distinct colors per column)
    color_pairs = [
        ('#1f77b4', '#aec7e8'),  # Blue pair
        ('#ff7f0e', '#ffbb78'),  # Orange pair
        ('#2ca02c', '#98df8a'),  # Green pair
        ('#d62728', '#ff9896'),  # Red pair
        ('#9467bd', '#c5b0d5'),  # Purple pair
        ('#8c564b', '#c49c94'),  # Brown pair
        ('#e377c2', '#f7b6d2'),  # Pink pair
        ('#7f7f7f', '#c7c7c7'),  # Gray pair
    ]

    # Plot each column with two distinct lines
    for i, column in enumerate(df.columns):
        # Get color pair (cycle through if more columns than color pairs)
        color1, color2 = color_pairs[i % len(color_pairs)]
        
        # Create x-axis values (row indices)
        x = range(len(df))
        y = df[column]
        
        # Plot two lines for this column with different styles
        ax.plot(x, y, color=color1, linewidth=2, label=f'{column} - Line 1', marker='o', markersize=4)
        ax.plot(x, y, color=color2, linewidth=1.5, linestyle='--', label=f'{column} - Line 2', alpha=0.7)

    # Customize the plot
    ax.set_xlabel('Row Index', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Color Percentage Data Visualization', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([i * 0.1 for i in range(11)])  # 0, 0.1, 0.2, ..., 1.0
    ax.grid(True, alpha=0.3)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig('color_percentage_graph.png', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

    print(f"Graph created successfully with {len(df.columns)} columns and {len(df)} data points per column.")