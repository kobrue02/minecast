import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap # For wrapping long text
import os
import logging
import matplotlib.gridspec as gridspec # Import gridspec

logger = logging.getLogger(__name__)

# Helper function to safely get data from DataFrame row
def get_value(data_row, col_name, default=None):
    if col_name in data_row and pd.notna(data_row[col_name]):
        return data_row[col_name]
    return default

def format_large_number(num, default_if_none="N/A"):
    if num is None or pd.isna(num):
        return default_if_none
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if abs(num) >= 1_000:
        return f"{num / 1_000:.2f}K"
    return f"{num:.2f}" if isinstance(num, float) else str(num)
    

def plot_mineral_summary(ax, data_row):
    ax.set_title("Mineral Resource/Reserve Summary", fontsize=10, loc='left', pad=10)
    ax.axis('off') # Turn off axis for text display first

    resource_data = []
    # TechnicalReportData.mineral_resources and .mineral_reserves are lists of ResourceEstimate
    # pd.json_normalize flattens this. Example: 'mineral_resources.0.category', 'mineral_resources.0.tonnes'
    
    for i in range(5): # Check for up to 5 resource/reserve items
        base_col_resource = f"mineral_resources.{i}"
        base_col_reserve = f"mineral_reserves.{i}"

        cat_res = get_value(data_row, f"{base_col_resource}.category")
        if cat_res:
            resource_data.append({
                "type": "Resource",
                "category": get_value(data_row, f"{base_col_resource}.category", "N/A"),
                "tonnes_M": get_value(data_row, f"{base_col_resource}.tonnes"), # Assuming in Millions from schema
                "grade": get_value(data_row, f"{base_col_resource}.grade"),
                "grade_unit": get_value(data_row, f"{base_col_resource}.grade_unit","N/A"),
                "mineral_type": get_value(data_row, f"{base_col_resource}.mineral_type", "N/A"),
                "contained_amount": get_value(data_row, f"{base_col_resource}.contained_amount"),
                "contained_unit": get_value(data_row, f"{base_col_resource}.contained_unit","N/A")
            })
        
        cat_resv = get_value(data_row, f"{base_col_reserve}.category")
        if cat_resv:
             resource_data.append({
                "type": "Reserve",
                "category": get_value(data_row, f"{base_col_reserve}.category", "N/A"),
                "tonnes_M": get_value(data_row, f"{base_col_reserve}.tonnes"), # Assuming in Millions
                "grade": get_value(data_row, f"{base_col_reserve}.grade"),
                "grade_unit": get_value(data_row, f"{base_col_reserve}.grade_unit","N/A"),
                "mineral_type": get_value(data_row, f"{base_col_reserve}.mineral_type", "N/A"),
                "contained_amount": get_value(data_row, f"{base_col_reserve}.contained_amount"),
                "contained_unit": get_value(data_row, f"{base_col_reserve}.contained_unit","N/A")
            })

    if not resource_data:
        ax.text(0.5, 0.5, "No Mineral Resource/Reserve data found.", ha='center', va='center', fontsize=9, wrap=True)
        return

    res_df = pd.DataFrame(resource_data)
    res_df = res_df.dropna(subset=['tonnes_M', 'grade']) # Ensure key plotting values are present

    if res_df.empty:
        ax.text(0.5, 0.5, "Resource/Reserve data incomplete for plotting.", ha='center', va='center', fontsize=9, wrap=True)
        return
    
    # Simplified bar chart: Tonnes by Category, colored by Type (Resource/Reserve)
    # For multiple minerals or more complex grades, this would need more sophistication
    
    # Reactivate axis for plotting
    ax.axis('on')
    sns.barplot(x="category", y="tonnes_M", hue="type", data=res_df, ax=ax, palette={"Resource": "skyblue", "Reserve": "steelblue"}, dodge=True)
    ax.set_ylabel("Tonnes (M)", fontsize=8)
    ax.set_xlabel("Category", fontsize=8)
    ax.tick_params(axis='x', rotation=15, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(title="Type", fontsize=7, title_fontsize=8)
    
    # Annotate with grade if possible (simplified: assuming one primary mineral type for display)
    # This is a simplified annotation. For a real report, you'd want to be very specific about which grade.
    primary_mineral = res_df['mineral_type'].mode()[0] if not res_df['mineral_type'].empty else 'N/A'
    primary_grade_unit = res_df['grade_unit'].mode()[0] if not res_df['grade_unit'].empty else ''
    
    for p in ax.patches:
        height = p.get_height()
        if pd.notna(height) and height > 0:
            # Find corresponding grade - this is tricky due to hue
            # This simplistic annotation takes average grade for the category for now
            try:
                category = p.get_x() + p.get_width() / 2.
                # This requires more robust logic to link bar to exact data point for grade
                # For now, let's skip grade annotation directly on bars to avoid complexity with hue
                # Instead, we add a note about primary mineral.
            except:
                pass # best effort
    
    ax.text(0.99, 0.01, f"Primary Mineral (visualized): {primary_mineral}", ha='right', va='bottom', transform=ax.transAxes, fontsize=6, alpha=0.7)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def plot_economic_indicators(subplot_spec_container, data_row):
    # subplot_spec_container is the SubplotSpec for the cell where economic indicators will be plotted.
    # ax.remove() is not needed as we are building on the SubplotSpec directly.

    # Create a nested GridSpec for the economic plots within the provided SubplotSpec
    gs_inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=subplot_spec_container, hspace=0.5, wspace=0.4, 
                                           height_ratios=[2,1], width_ratios=[2,1])

    # Get the main figure object from the subplot_spec_container
    fig = subplot_spec_container.get_gridspec().figure

    ax1 = fig.add_subplot(gs_inner[0, 0])  # Main economic indicators (NPV, CAPEX)
    ax2 = fig.add_subplot(gs_inner[0, 1])  # AISC
    ax3 = fig.add_subplot(gs_inner[1, 0])  # IRR and LOM
    ax4 = fig.add_subplot(gs_inner[1, 1])  # Additional metrics

    # Main economic indicators (NPV, CAPEX)
    ax1.set_title("Major Economic Indicators (USD)", fontsize=9, loc='left', pad=8)
    main_indicators = {
        "NPV": get_value(data_row, "npv_post_tax_usd"),
        "Initial CAPEX": get_value(data_row, "initial_capital_cost_usd"),
        "Sustaining CAPEX": get_value(data_row, "sustaining_capital_cost_usd"),
        "Total CAPEX": get_value(data_row, "total_capital_cost_usd")
    }
    main_indicators = {k: v for k, v in main_indicators.items() if pd.notna(v)}
    
    if main_indicators:
        labels = list(main_indicators.keys())
        values = [float(v) if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit() else 0 for v in main_indicators.values()]
        bars = sns.barplot(x=values, y=labels, hue=labels, ax=ax1, palette="viridis", orient='h', legend=False)
        ax1.set_xlabel("Value (USD)", fontsize=7)
        ax1.tick_params(labelsize=7)
        ax1.set_ylabel("", fontsize=7) # Explicitly set y-label
        
        max_val_for_text = max(values) if values else 1 # Avoid division by zero if values are all zero
        for i, bar in enumerate(bars.patches):
            label_val = list(main_indicators.values())[i]
            formatted_val = format_large_number(label_val)
            text_x = bar.get_width()
            ha = 'left'
            # Adjust text position for better readability
            if bar.get_width() < (max_val_for_text * 0.25):
                 text_x += (max_val_for_text * 0.02) 
            else: 
                text_x -= (max_val_for_text * 0.01)
                ha = 'right'
            ax1.text(text_x, bar.get_y() + bar.get_height() / 2, formatted_val, 
                    va='center', ha=ha, fontsize=6, color='black' if ha == 'left' else 'white')
    else:
        ax1.text(0.5, 0.5, "No major economic data found.", ha='center', va='center', fontsize=8)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # AISC Plot
    ax2.set_title("AISC", fontsize=9, loc='left', pad=8)
    aisc_data = get_value(data_row, "aisc_usd_per_ounce")
    aisc_label = "AISC"
    aisc_value = None

    if isinstance(aisc_data, dict) and aisc_data:
        key = next(iter(aisc_data))
        aisc_label = f"AISC ({key.replace('_', ' ').title()})"
        aisc_value = aisc_data[key]
    else:
        flat_aisc_cols = [col for col in data_row.index if str(col).startswith("aisc_usd_per_ounce.")]
        if flat_aisc_cols:
            aisc_full_col_name = flat_aisc_cols[0]
            aisc_value = get_value(data_row, aisc_full_col_name)
            commodity_key = aisc_full_col_name.split('.')[-1].replace('_', ' ').title()
            aisc_label = f"AISC ({commodity_key})"

    if aisc_value is not None:
        # For a single bar, hue can be the y-variable itself
        bars = sns.barplot(x=[float(aisc_value)], y=[aisc_label], hue=[aisc_label], ax=ax2, palette="viridis", orient='h', legend=False)
        ax2.set_xlabel("USD per ounce", fontsize=7)
        ax2.tick_params(labelsize=7)
        # Text annotation for AISC bar
        bar = bars.patches[0]
        text_x = bar.get_width() - (bar.get_width() * 0.05) # Inside bar, to the right
        ax2.text(text_x, bar.get_y() + bar.get_height() / 2, f"${aisc_value:,.0f}", 
                 va='center', ha='right', fontsize=6, color='white')
    else:
        ax2.text(0.5, 0.5, "No AISC data.", ha='center', va='center', fontsize=8)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    # IRR and LOM Plot
    ax3.set_title("Project Metrics", fontsize=9, loc='left', pad=8)
    project_metrics = {
        "IRR (%)": get_value(data_row, "irr_post_tax_percent"),
        "LOM (Years)": get_value(data_row, "life_of_mine_years"),
        "Payback (Years)": get_value(data_row, "payback_period_years"),
        "Discount Rate (%)": get_value(data_row, "discount_rate_percent")
    }
    project_metrics = {k: v for k, v in project_metrics.items() if pd.notna(v)}
    
    if project_metrics:
        labels = list(project_metrics.keys())
        values = [float(v) if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit() else 0 for v in project_metrics.values()]
        bars = sns.barplot(x=values, y=labels, hue=labels, ax=ax3, palette="viridis", orient='h', legend=False)
        ax3.set_xlabel("Value", fontsize=7)
        ax3.tick_params(labelsize=7)
        ax3.set_ylabel("", fontsize=7)

        max_val_for_text = max(values) if values else 1
        for i, bar in enumerate(bars.patches):
            label_val = list(project_metrics.values())[i]
            formatted_val = f"{label_val:.1f}%" if "IRR" in labels[i] or "Rate" in labels[i] else f"{label_val:.1f}"
            text_x = bar.get_width()
            ha = 'left'
            if bar.get_width() < (max_val_for_text * 0.25):
                 text_x += (max_val_for_text * 0.02) 
            else: 
                text_x -= (max_val_for_text * 0.01)
                ha = 'right'
            ax3.text(text_x, bar.get_y() + bar.get_height() / 2, formatted_val, 
                    va='center', ha=ha, fontsize=6, color='black' if ha == 'left' else 'white')
    else:
        ax3.text(0.5, 0.5, "No project metrics found.", ha='center', va='center', fontsize=8)
    ax3.grid(axis='x', linestyle='--', alpha=0.7)

    # Additional Metrics Plot
    ax4.set_title("Other Metrics", fontsize=9, loc='left', pad=8)
    additional_metrics = {
        "OpCost (USD/t)": get_value(data_row, "operating_cost_usd_per_tonne"),
        "Strip Ratio": get_value(data_row, "strip_ratio"),
        "Recovery (%)": get_value(data_row, "recovery_rate_percent"),
        "Process (tpd)": get_value(data_row, "processing_rate_tonnes_per_day")
    }
    additional_metrics = {k: v for k, v in additional_metrics.items() if pd.notna(v)}
    
    if additional_metrics:
        labels = list(additional_metrics.keys())
        values = [float(v) if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit() else 0 for v in additional_metrics.values()]
        bars = sns.barplot(x=values, y=labels, hue=labels, ax=ax4, palette="viridis", orient='h', legend=False)
        ax4.set_xlabel("Value", fontsize=7)
        ax4.tick_params(labelsize=7)
        ax4.set_ylabel("", fontsize=7)

        max_val_for_text = max(values) if values else 1
        for i, bar in enumerate(bars.patches):
            label_val = list(additional_metrics.values())[i]
            formatted_val = f"{label_val:,.1f}"
            text_x = bar.get_width()
            ha = 'left'
            if bar.get_width() < (max_val_for_text * 0.25):
                 text_x += (max_val_for_text * 0.02) 
            else: 
                text_x -= (max_val_for_text * 0.01)
                ha = 'right'
            ax4.text(text_x, bar.get_y() + bar.get_height() / 2, formatted_val, 
                    va='center', ha=ha, fontsize=6, color='black' if ha == 'left' else 'white')
    else:
        ax4.text(0.5, 0.5, "No additional metrics found.", ha='center', va='center', fontsize=8)
    ax4.grid(axis='x', linestyle='--', alpha=0.7)


def plot_financial_summary(outer_subplot_spec, data_row):
    fig = outer_subplot_spec.get_gridspec().figure

    # Create a 2-row layout: Row 0 for Balance Sheet, Row 1 for Stock Info
    gs_financial_main = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_subplot_spec, hspace=0.5, height_ratios=[1, 1.8])

    # --- Balance Sheet Plot ---
    ax_bs = fig.add_subplot(gs_financial_main[0])
    ax_bs.set_title("Balance Sheet Highlights (USD)", fontsize=9, loc='left', pad=5)
    
    bs_suffix = "_bs" if "Cash And Cash Equivalents_bs" in data_row else ""
    balance_sheet_items = {
        "Cash & Equiv.": get_value(data_row, f"Cash And Cash Equivalents{bs_suffix}"),
        "Total Debt": get_value(data_row, f"Total Debt{bs_suffix}"),
        "Working Capital": get_value(data_row, f"Working Capital{bs_suffix}")
    }
    bs_to_plot = {k: v for k, v in balance_sheet_items.items() if pd.notna(v) and v != 0}

    if bs_to_plot:
        bs_labels = list(bs_to_plot.keys())
        bs_values = [float(v) for v in bs_to_plot.values()]
        bs_bars = sns.barplot(x=bs_values, y=bs_labels, hue=bs_labels, ax=ax_bs, palette="Blues_r", orient='h', legend=False)
        ax_bs.set_xlabel("Value (USD)", fontsize=7)
        ax_bs.tick_params(labelsize=7)
        max_bs_val = max(bs_values) if bs_values else 1
        for i, bar in enumerate(bs_bars.patches):
            val = bs_values[i]
            formatted_val = format_large_number(val)
            text_x = bar.get_width()
            ha = 'left'
            if bar.get_width() < (max_bs_val * 0.3):
                 text_x += (max_bs_val * 0.02)
            else:
                text_x -= (max_bs_val * 0.01)
                ha = 'right'
            ax_bs.text(text_x, bar.get_y() + bar.get_height() / 2, formatted_val, va='center', ha=ha, fontsize=6, color='black' if ha == 'left' else 'white')
        ax_bs.grid(axis='x', linestyle='--', alpha=0.7)
    else:
        ax_bs.text(0.5, 0.5, "No Balance Sheet data available.", ha='center', va='center', fontsize=8)
        ax_bs.axis('off')

    # --- Stock Info Section (Subdivided) ---
    stock_info_outer_spec = gs_financial_main[1]
    gs_stock_details = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=stock_info_outer_spec, hspace=0.6, wspace=0.4)

    # Plot 1: Market Cap & Enterprise Value
    ax_sm_large = fig.add_subplot(gs_stock_details[0, 0])
    ax_sm_large.set_title("Valuation (USD)", fontsize=8, loc='left', pad=3)
    stock_large_values = {
        "Market Cap": get_value(data_row, "market_cap"),
        "Ent. Value": get_value(data_row, "enterprise_value")
    }
    sl_to_plot = {k: v for k, v in stock_large_values.items() if pd.notna(v) and v != 0}
    if sl_to_plot:
        sl_labels = list(sl_to_plot.keys())
        sl_values = [float(v) for v in sl_to_plot.values()]
        sl_bars = sns.barplot(x=sl_values, y=sl_labels, hue=sl_labels, ax=ax_sm_large, palette="Greens_r", orient='h', legend=False)
        max_sl_val = max(sl_values) if sl_values else 1
        for i, bar in enumerate(sl_bars.patches):
            val = sl_values[i]
            formatted_val = format_large_number(val)
            text_x = bar.get_width() - (max_sl_val * 0.01)
            ax_sm_large.text(text_x, bar.get_y() + bar.get_height() / 2, formatted_val, va='center', ha='right', fontsize=6, color='white')
        ax_sm_large.grid(axis='x', linestyle='--', alpha=0.7)
    else:
        ax_sm_large.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=8)
        ax_sm_large.axis('off')
    ax_sm_large.tick_params(labelsize=7)
    ax_sm_large.set_xlabel("", fontsize=7)

    # Plot 2: Current Price & Shares Outstanding (Text or small bars)
    ax_sp_other = fig.add_subplot(gs_stock_details[0, 1])
    ax_sp_other.set_title("Price & Shares", fontsize=8, loc='left', pad=3)
    ax_sp_other.axis('off')
    y_pos_sp = 0.8
    cur_price = get_value(data_row, "current_price")
    currency = get_value(data_row, "currency", "")
    if cur_price is not None:
        ax_sp_other.text(0.05, y_pos_sp, f"Cur. Price: {cur_price:.2f} {currency}", fontsize=7, va='center')
        y_pos_sp -= 0.3
    shares_out = get_value(data_row, "shares_outstanding")
    if shares_out is not None:
        ax_sp_other.text(0.05, y_pos_sp, f"Shares Out: {format_large_number(shares_out)}", fontsize=7, va='center')

    # Plot 3: Ratios (PE, Beta)
    ax_sr = fig.add_subplot(gs_stock_details[1, 0])
    ax_sr.set_title("Ratios & Beta", fontsize=8, loc='left', pad=3)
    stock_ratios = {
        "Trailing PE": get_value(data_row, "trailing_pe"),
        "Forward PE": get_value(data_row, "forward_pe"),
        "Beta": get_value(data_row, "beta")
    }
    sr_to_plot = {k: v for k, v in stock_ratios.items() if pd.notna(v)}
    if sr_to_plot:
        sr_labels = list(sr_to_plot.keys())
        sr_values = [float(v) for v in sr_to_plot.values()]
        sr_bars = sns.barplot(x=sr_values, y=sr_labels, hue=sr_labels, ax=ax_sr, palette="Purples_r", orient='h', legend=False)
        max_sr_val = max(sr_values) if sr_values else 1
        for i, bar in enumerate(sr_bars.patches):
            val = sr_values[i]
            formatted_val = f"{val:.2f}"
            text_x = bar.get_width() - (max_sr_val * 0.01) if max_sr_val > 0 else bar.get_width() * 0.99
            if bar.get_width() < 0: # Handle negative PE ratios if they occur
                 text_x = 0.01
                 ha = 'left'
                 bar_color = 'black'
            else:
                 ha = 'right'
                 bar_color = 'white'

            ax_sr.text(text_x, bar.get_y() + bar.get_height() / 2, formatted_val, va='center', ha=ha, fontsize=6, color=bar_color)
        ax_sr.grid(axis='x', linestyle='--', alpha=0.7)
    else:
        ax_sr.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=8)
        ax_sr.axis('off')
    ax_sr.tick_params(labelsize=7)
    ax_sr.set_xlabel("Value", fontsize=7)
    
    # Plot 4: Institutional Ownership
    ax_si = fig.add_subplot(gs_stock_details[1, 1])
    ax_si.set_title("Institutional Own. %", fontsize=8, loc='left', pad=3)
    inst_own_str = get_value(data_row, "institutional_ownership_percent")
    inst_own_val = None
    if inst_own_str is not None:
        if isinstance(inst_own_str, str):
            try: inst_own_val = float(inst_own_str.replace('%', ''))
            except ValueError: pass
        elif isinstance(inst_own_str, (float, int)): inst_own_val = float(inst_own_str)
    
    if inst_own_val is not None:
        sns.barplot(x=[inst_own_val], y=["Ownership"], hue=["Ownership"], ax=ax_si, palette=["Oranges_r"], orient='h', legend=False)
        ax_si.set_xlim(0, 100)
        ax_si.set_xlabel("%", fontsize=7)
        ax_si.text(inst_own_val - 1, 0, f"{inst_own_val:.1f}%", va='center', ha='right', fontsize=6, color='white')
        ax_si.grid(axis='x', linestyle='--', alpha=0.7)
    else:
        ax_si.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=8)
        ax_si.axis('off')
    ax_si.set_yticks([])
    ax_si.tick_params(axis='x', labelsize=7)


def create_dashboard_visualization(df_row: pd.Series, ticker: str, output_image_path: str):
    if df_row is None or df_row.empty:
        logger.error(f"DataFrame row for ticker {ticker} is empty. Cannot generate dashboard.")
        return

    sns.set_theme(style="whitegrid", palette="pastel")
    
    fig = plt.figure(figsize=(18, 14)) # Slightly adjusted figure size
    
    # Title GridSpec
    gs_title_outer = plt.GridSpec(1, 1, figure=fig, top=0.95, bottom=0.9, left=0.05, right=0.95)
    title_ax = fig.add_subplot(gs_title_outer[0])
    title_ax.axis('off')
    
    report_title = get_value(df_row, "report_title", "N/A")
    project_name = get_value(df_row, "project_name", "N/A")
    company_name = get_value(df_row, "company_name", ticker.upper())
    main_title_text = f"Data Snapshot for: {company_name} ({ticker.upper()})"
    sub_title_text = f"Project: {project_name} | Report: {textwrap.shorten(report_title, width=70, placeholder='...')}"
    title_ax.text(0.5, 0.5, main_title_text + "\n" + sub_title_text, 
                 fontsize=14, weight='bold', ha='center', va='center')

    # Main plots GridSpec: gives more relative height to the top row (mineral & econ)
    # And more relative width to the left column (mineral & financial)
    gs_main = plt.GridSpec(2, 2, figure=fig, top=0.88, bottom=0.08, left=0.07, right=0.95, 
                           hspace=0.35, wspace=0.25, height_ratios=[1.2, 1], width_ratios=[1.2, 1])
    
    ax_mineral = fig.add_subplot(gs_main[0, 0])
    ax_financial = fig.add_subplot(gs_main[1, 0])
    ax_additional_ph = fig.add_subplot(gs_main[1, 1]) # Placeholder
    
    plot_mineral_summary(ax_mineral, df_row)
    plot_economic_indicators(gs_main[0, 1], df_row)
    plot_financial_summary(gs_main[1, 1], df_row)
    
    ax_additional_ph.set_title("Sensitivities / Commodity Prices", fontsize=9, loc='left', pad=8)
    ax_additional_ph.text(0.5, 0.5, "Placeholder for sensitivity analysis or commodity price charts.", 
                           ha='center', va='center', fontsize=8, wrap=True)
    ax_additional_ph.axis('off')

    report_date = get_value(df_row, "report_date", "N/A")
    balance_sheet_date = get_value(df_row, "BalanceSheet_ReportDate", get_value(df_row, "ReportDate_bs"))
    bs_date_str = f"BS Date: {balance_sheet_date}" if balance_sheet_date else "BS Date: N/A"
    footer_text = f"Technical Report Date: {report_date} | {bs_date_str} | Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
    fig.text(0.5, 0.015, footer_text, ha='center', va='bottom', fontsize=7, color='gray')

    try:
        plt.savefig(output_image_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        logger.info(f"Dashboard visualization saved to {output_image_path}")
    except Exception as e:
        logger.error(f"Failed to save dashboard image to {output_image_path}: {e}")
    finally:
        plt.close(fig)

if __name__ == '__main__':
    # This is for testing the dashboard script directly
    # Create a dummy DataFrame row similar to what the pipeline would produce
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing dashboard visualization script...")

    # Create an absolute path for the output directory if it doesn't exist
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = os.path.join(os.path.dirname(current_script_dir), "downloaded_reports") # In MineCast/downloaded_reports
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    
    test_image_path = os.path.join(test_output_dir, "test_dashboard_viz.png")

    # Sample data (simulating a single row from the processed DataFrame)
    sample_data = {
        "ticker": "TESTCO",
        "company_name": "Test Mining Corp",
        "project_name": "El Tesoro Project",
        "report_title": "Feasibility Study for El Tesoro Gold Project",
        "report_date": "2023-10-01",
        "BalanceSheet_ReportDate": "2023-09-30",
        # Mineral Resources
        "mineral_resources.0.category": "Indicated", "mineral_resources.0.tonnes": 50.0, 
        "mineral_resources.0.grade": 1.5, "mineral_resources.0.grade_unit": "g/t", 
        "mineral_resources.0.mineral_type": "Gold",
        "mineral_resources.1.category": "Inferred", "mineral_resources.1.tonnes": 120.0, 
        "mineral_resources.1.grade": 1.1, "mineral_resources.1.grade_unit": "g/t",
        "mineral_resources.1.mineral_type": "Gold",
        # Mineral Reserves
        "mineral_reserves.0.category": "Probable", "mineral_reserves.0.tonnes": 40.0,
        "mineral_reserves.0.grade": 1.4, "mineral_reserves.0.grade_unit": "g/t",
        "mineral_reserves.0.mineral_type": "Gold",
        # Economic Indicators
        "npv_post_tax_usd": 250000000, "irr_post_tax_percent": 22.5,
        "initial_capital_cost_usd": 150000000, "life_of_mine_years": 10,
        "aisc_usd_per_ounce.gold_oz": 950, # Flattened AISC
        # Balance Sheet
        "Cash And Cash Equivalents_bs": 20000000, "Total Debt_bs": 50000000, "Working Capital_bs": 5000000,
        # Stock Info
        "current_price": 10.50, "currency": "USD", "market_cap": 500000000, "shares_outstanding": 50000000,
        "enterprise_value": 530000000, "beta": 1.2, "trailing_pe": 15.0, "forward_pe": 12.0,
        "institutional_ownership_percent": "65.7%"
    }
    sample_df_row = pd.Series(sample_data)

    # Explicitly re-defining this call and context to clear potential hidden characters
    create_dashboard_visualization(sample_df_row, "TESTCO", test_image_path)
    
    # Test with more missing data
    sample_data_missing = {
        "ticker": "NODATCO",
        "company_name": "No Data Mining Co",
        "project_name": "Empty Site",
        "report_title": "Preliminary Assessment of Nothing",
        "report_date": "2024-01-01",
        "current_price": 1.0, "currency": "USD", "market_cap": 1000000
    }
    sample_df_row_missing = pd.Series(sample_data_missing)
    test_image_path_missing = os.path.join(test_output_dir, "test_dashboard_viz_missing.png")
    create_dashboard_visualization(sample_df_row_missing, "NODATCO", test_image_path_missing)

    logger.info("Test dashboard generation complete. Check images in 'downloaded_reports' (relative to project root).") 