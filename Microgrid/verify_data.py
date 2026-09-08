import matplotlib.pyplot as plt
from data import master_df

def verify_data_alignment():
    print("--- Diagnostic Readout ---")
    print(f"Dataset Range: {master_df.index.min()} to {master_df.index.max()}")
    print(f"Total Rows: {len(master_df)}")
    print(f"Missing Values:\n{master_df.isna().sum()}\n")

    # Select a 2-week window in the summer of 2018 to check volatility and alignment
    start_date = '2018-08-01'
    end_date = '2018-08-14'
    
    subset = master_df.loc[start_date:end_date]
    
    if subset.empty:
        print(f"Error: No data found between {start_date} and {end_date}. Check your dataset dates.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot Load on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Microgrid Load (MW)', color=color, fontweight='bold')
    ax1.plot(subset.index, subset['Load_MW'], color=color, linewidth=1.5, label='Load (MW)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Temperature on secondary y-axis to verify temporal alignment
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Temperature (°C)', color=color, fontweight='bold')  
    ax2.plot(subset.index, subset['Temperature_2m'], color=color, linewidth=1.5, alpha=0.7, linestyle='--', label='Temperature')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Austin Microgrid Data Verification ({start_date} to {end_date})')
    fig.tight_layout()
    plt.savefig('austin_data_verification.png', dpi=300)
    print("Verification plot saved to 'austin_data_verification.png'.")
    plt.show()

if __name__ == "__main__":
    verify_data_alignment()