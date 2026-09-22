import matplotlib.pyplot as plt
import pandas as pd
from data import master_df

def verify_data_alignment():
    print("--- Diagnostic Readout ---")
    print(f"Dataset Range: {master_df.index.min()} to {master_df.index.max()}")
    print(f"Total Rows: {len(master_df)}")
    print(f"Missing Values:\n{master_df.isna().sum()}\n")

    # Select the first 2 weeks of the dataset to check volatility and alignment
    start_date = master_df.index.min().strftime('%Y-%m-%d')
    end_date = (master_df.index.min() + pd.Timedelta(days=14)).strftime('%Y-%m-%d')
    
    subset = master_df.loc[start_date:end_date]
    
    if subset.empty:
        print(f"Error: No data found between {start_date} and {end_date}.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot Load on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Building Load (kW)', color=color, fontweight='bold')
    ax1.plot(subset.index, subset['Load_kW'], color=color, linewidth=1.5, label='Load (kW)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Temperature on secondary y-axis to verify temporal alignment
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Temperature (°C)', color=color, fontweight='bold')  
    ax2.plot(subset.index, subset['Temperature_2m'], color=color, linewidth=1.5, alpha=0.7, linestyle='--', label='Temperature')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Building 59 Data Verification ({start_date} to {end_date})')
    fig.tight_layout()
    plt.savefig('latrobe_data_verification.png', dpi=300)
    print("Verification plot saved to 'latrobe_data_verification.png'.")
    plt.show()

if __name__ == "__main__":
    verify_data_alignment()