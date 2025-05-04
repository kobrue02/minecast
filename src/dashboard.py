import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Set the style for all visualizations
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16

def dashboard(df: pd.DataFrame, title: str = "Mining Company Dashboard"):
    """
    Create a dashboard for parsed mining data using matplotlib and seaborn.

    Args:
        df (pd.DataFrame): DataFrame containing project data.
    """
    # use underscore for column names instead of spaces
    df.columns = df.columns.str.replace(' ', '_')
    # Create a figure with a grid layout
    plt.figure(figsize=(20, 16))
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'{title} Gold Project Dashboard', fontsize=24, fontweight='bold', y=0.98)

    # Create GridSpec layout
    gs = GridSpec(4, 6, figure=fig)

    # Function to format large numbers
    def format_numbers(x, pos):
        if x >= 1e9:
            return f'{x*1e-9:.1f}B'
        elif x >= 1e6:
            return f'{x*1e-6:.1f}M'
        elif x >= 1e3:
            return f'{x*1e-3:.1f}K'
        else:
            return f'{x:.1f}'

    formatter = ticker.FuncFormatter(format_numbers)

    # 1. Project Financial Overview
    ax1 = fig.add_subplot(gs[0, :3])
    financial_data = [df['post_tax_net_present_value'].values[0], 
                    df['market_cap'].values[0]]
    financial_labels = ['Post-Tax NPV', 'Market Cap']
    colors = sns.color_palette("viridis", 2)
    bars = ax1.bar(financial_labels, financial_data, color=colors)
    ax1.set_title('Project Financial Overview', fontweight='bold')
    ax1.yaxis.set_major_formatter(formatter)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'${height/1e6:.1f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    # 2. Production Economics
    ax2 = fig.add_subplot(gs[0, 3:])
    production_labels = ['Gold Price ($/oz)', 'All-in Sustaining Cost ($/oz)', 'Margin ($/oz)']
    margin = df['long_term_gold_price_per_ounce_base_case'].values[0] - df['all_in_sustaining_cost_per_ounce'].values[0]
    production_data = [df['long_term_gold_price_per_ounce_base_case'].values[0], 
                    df['all_in_sustaining_cost_per_ounce'].values[0],
                    margin]
    colors = sns.color_palette("magma", 3)
    bars = ax2.bar(production_labels, production_data, color=colors)
    ax2.set_title('Gold Production Economics', fontweight='bold')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'${height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    # 3. Balance Sheet Overview
    ax3 = fig.add_subplot(gs[1, :3])
    balance_sheet_data = [df['Cash_And_Cash_Equivalents'].values[0],
                        df['Total_Debt'].values[0],
                        df['Current_Debt'].values[0],
                        df['Working_Capital'].values[0]]
    balance_sheet_labels = ['Cash & Equivalents', 'Total Debt', 'Current Debt', 'Working Capital']
    colors = sns.color_palette("muted", 4)
    bars = ax3.bar(balance_sheet_labels, balance_sheet_data, color=colors)
    ax3.set_title('Balance Sheet Overview', fontweight='bold')
    ax3.yaxis.set_major_formatter(formatter)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'${height/1e6:.1f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    # 4. Ownership Structure Pie Chart - FIXED VERSION
    ax4 = fig.add_subplot(gs[1, 3:])
    insiders_percent = max(0, df['insidersPercentHeld'].values[0] * 100)  # Ensure non-negative
    institutions_percent = max(0, df['institutionsPercentHeld'].values[0] * 100)  # Ensure non-negative
    other_percent = max(0, 100 - (insiders_percent + institutions_percent))  # Ensure non-negative

    # Make sure we have at least some positive values for the pie chart
    if insiders_percent <= 0 and institutions_percent <= 0:
        # If both main values are zero or negative, create a simple placeholder
        labels = ['Other Shareholders']
        sizes = [100]
        colors = sns.color_palette("pastel", 1)
        explode = None
    else:
        # Only include segments with positive values
        labels = []
        sizes = []
        colors_list = []
        explode_list = []
        
        if insiders_percent > 0:
            labels.append('Insiders')
            sizes.append(insiders_percent)
            colors_list.append(sns.color_palette("pastel")[0])
            explode_list.append(0.1)
            
        if institutions_percent > 0:
            labels.append('Institutions')
            sizes.append(institutions_percent)
            colors_list.append(sns.color_palette("pastel")[1])
            explode_list.append(0)
            
        if other_percent > 0:
            labels.append('Other')
            sizes.append(other_percent)
            colors_list.append(sns.color_palette("pastel")[2])
            explode_list.append(0)
            
        colors = colors_list
        explode = explode_list if explode_list else None

    # Create pie chart with sanitized data
    ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax4.set_title('Ownership Structure', fontweight='bold')

    # Add a note about the ownership data if there were issues
    if insiders_percent <= 0 or institutions_percent <= 0:
        ax4.annotate('Note: Ownership data may be incomplete',
                    xy=(0.5, -0.1),
                    xycoords='axes fraction',
                    ha='center', va='center',
                    fontsize=10, fontstyle='italic')

    # 5. Project Timeline
    ax5 = fig.add_subplot(gs[2, :2])
    timeline_labels = ['Release Date', 'Commercial Production']
    timeline_dates = ['Feb 2022', 'July 2024']

    # Convert to number of months since Jan 2022
    months_since_start = [1, 30]  # Feb 2022 -> 1, July 2024 -> 30

    ax5.plot(months_since_start, [1, 1], 'o-', markersize=15, linewidth=3, color='darkblue')
    ax5.set_yticks([])
    ax5.set_xticks(months_since_start)
    ax5.set_xticklabels(timeline_dates)
    ax5.grid(axis='x')
    ax5.set_xlim(0, max(months_since_start) + 5)

    # Add event labels above points
    for i, label in enumerate(timeline_labels):
        ax5.annotate(label, 
                    xy=(months_since_start[i], 1),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax5.set_title('Project Timeline', fontweight='bold')

    # 8. Key Financial Metrics Table - FIXED
    ax8 = fig.add_subplot(gs[3, 1:5])
    ax8.axis('tight')
    ax8.axis('off')

    metrics_data = [
        ['Post-Tax NPV', f"${df['post_tax_net_present_value'].values[0]/1e6:.1f}M"],
        ['Gold Price (Base Case)', f"${df['long_term_gold_price_per_ounce_base_case'].values[0]:.0f}/oz"],
        ['All-in Sustaining Cost', f"${df['all_in_sustaining_cost_per_ounce'].values[0]:.0f}/oz"],
        ['Annual Gold Production', f"{df['average_annual_gold_production_in_ounces'].values[0]:,.0f} oz"],
        ['Mine Life', f"{df['total_life_of_mine'].values[0]:.0f} years"],
        ['Head Grade', f"{df['average_diluted_head_grade'].values[0]:.2f} g/t"],
        ['Market Cap', f"${df['market_cap'].values[0]/1e6:.1f}M {df['currency'].values[0]}"],
        ['Current Share Price', f"${df['current_price'].values[0]:.2f} {df['currency'].values[0]}"],
        ['Cash Position', f"${df['Cash_And_Cash_Equivalents'].values[0]/1e6:.1f}M"],
        ['Total Debt', f"${df['Total_Debt'].values[0]/1e6:.1f}M"]
    ]

    data_colors = [['ghostwhite', 'ghostwhite'] for _ in range(len(metrics_data))]
    table = ax8.table(cellText=metrics_data, 
                    colLabels=['Metric', 'Value'],
                    cellLoc='center',
                    loc='center',
                    cellColours=data_colors) 

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Add title for the table
    ax8.set_title('Key Project Metrics', fontweight='bold', pad=20)

    # Add company and project info at the bottom
    footer_text = f"{df['company_name'].values[0]} ({df['ticker'].values[0]}) - {df['project_name'].values[0]}, {df['project_location'].values[0]}"
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=12, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('mining_company_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()