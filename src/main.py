import argparse
import logging
import os
import sys
import pandas as pd # Added for type hinting and checking df_row
import warnings # Add this import

# Attempt to ensure 'src' is in path if script is run in certain ways,
# though usually not needed if run as 'python src/main.py' from root or 'python -m src.main'
# Get the absolute path of the 'src' directory
# current_script_path = os.path.dirname(os.path.abspath(__file__))
# src_path = os.path.abspath(os.path.join(current_script_path, '.')) # Assumes main.py is directly in src
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)
# project_root_path = os.path.abspath(os.path.join(src_path, '..'))
# if project_root_path not in sys.path:
#     sys.path.insert(0, project_root_path)

from dotenv import load_dotenv

load_dotenv()

from src.retriever.edgar import Edgar
from src.etl.pipeline import Pipeline
from src.dashboard.viz import create_dashboard_visualization # Import the dashboard function
# from etl.extractor.types import NewsType # Not directly needed here

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
USER_AGENT_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", "your_name@example.com")
DOWNLOAD_DIR = "downloaded_reports"

def run_full_pipeline(ticker_symbol: str):
    """
    Runs the full data retrieval and processing pipeline for a given stock ticker.
    Checks for a pre-existing processed CSV file first.
    1. If CSV exists, loads data from it.
    2. Else, downloads the latest S-K 1300 technical report using the Edgar retriever.
    3. Processes the downloaded report using the ETL pipeline.
    4. Outputs the processed data as a DataFrame and saves it to CSV (if not loaded from cache).
    5. Generates and saves a dashboard visualization from the processed data.
    """
    logger.info(f"--- Starting full pipeline for ticker: {ticker_symbol} ---")

    if USER_AGENT_EMAIL == "your_name@example.com":
        logger.warning("Using default User-Agent email for SEC Edgar. "
                       "Please set the SEC_USER_AGENT_EMAIL environment variable with your actual email for SEC compliance.")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    abs_download_dir = os.path.abspath(DOWNLOAD_DIR)
    
    # Define the expected processed CSV filename based on ticker and a generic report name part
    # This needs to be robust. For now, we might need a placeholder or a way to find the latest.
    # Let's assume the CSV name will be simpler: TICKER_processed_data.csv
    safe_ticker_filename = "".join(c if c.isalnum() else "_" for c in ticker_symbol)
    # The original CSV naming depends on the downloaded file's name.
    # For caching, we need a predictable name *before* downloading.
    # Let's define a cache CSV name format that only depends on the ticker.
    cache_csv_filename = f"{safe_ticker_filename}_processed_dashboard_data.csv"
    cache_csv_path = os.path.join(abs_download_dir, cache_csv_filename)

    processed_data_df = None
    data_row_for_dashboard = None
    processed_data_for_dashboards = [] # Initialize here, outside the cache-check block

    if os.path.exists(cache_csv_path):
        logger.info(f"Found existing processed CSV: {cache_csv_path}. Loading data from CSV.")
        try:
            processed_data_df = pd.read_csv(cache_csv_path)
            if not processed_data_df.empty:
                # Convert all columns to string first to avoid type issues, then infer best types
                # This is important if CSV was saved with mixed types or NaNs that pandas might misinterpret on load.
                # However, for dashboard, specific types might be expected. Best to ensure CSV saving is robust.
                # For now, directly use the loaded df.
                logger.info(f"Successfully loaded data from CSV. Shape: {processed_data_df.shape}")
            else:
                logger.warning(f"Loaded CSV {cache_csv_path} is empty. Proceeding with full pipeline.")
                processed_data_df = None # Reset to ensure full pipeline runs
        except Exception as e:
            logger.error(f"Failed to load or parse CSV {cache_csv_path}: {e}. Proceeding with full pipeline.")
            processed_data_df = None # Reset

    if processed_data_df is None or processed_data_df.empty:
        logger.info(f"No cached data found or cache was invalid for {ticker_symbol}. Running full data retrieval and processing.")
        # Step 1: Retrieve data using Edgar
        edgar_retriever = Edgar(user_agent_email=USER_AGENT_EMAIL)
        logger.info(f"Reports will be downloaded to: {abs_download_dir}")

        # Returns a list of dictionaries, one for each downloaded report
        retrieved_reports_info = edgar_retriever.get_latest_economic_reports(
            ticker=ticker_symbol,
            output_dir=abs_download_dir,
            num_filings_to_check=100 # Or make this configurable
        )

        if not retrieved_reports_info:
            logger.error(f"Failed to download any relevant reports for {ticker_symbol}. Pipeline cannot proceed.")
            logger.info(f"--- Pipeline finished for ticker: {ticker_symbol} ---")
            return

        logger.info(f"Successfully retrieved {len(retrieved_reports_info)} report(s) for {ticker_symbol}.")
        for report_info in retrieved_reports_info:
            logger.info(f"  - Type: {report_info.get('doc_type')}, Path: {report_info.get('filepath')}")


        # Step 2: Process the downloaded file(s) using the ETL Pipeline
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set. The ETL Pipeline requires it to function.")
            # Consider returning or raising an error if API key is crucial and missing
        
        etl_pipeline = Pipeline() 
        logger.info(f"Processing {len(retrieved_reports_info)} retrieved report(s) with ETL pipeline...")
        
        # The pipeline's run_all method expects a list of dicts with 'file_path', 'ticker', and optionally 'doc_type'
        # retrieved_reports_info already contains 'filepath', 'ticker', and 'doc_type' (as 'filepath', 'ticker', 'doc_type' respectively)
        # The key names in retrieved_reports_info are: 'filepath', 'ticker', 'doc_type', 'filing_date', 'accession_number', 'report_title', 'source_url'
        # We need to ensure the pipeline's run_all method gets the correct keys.
        # The `run_all` in pipeline.py expects `files_info` which is List[Dict[str, str]] with 'file_path', 'ticker', 'doc_type'
        # Our `retrieved_reports_info` matches this.
        
        processed_results_list = etl_pipeline.run_all(
            files_info=retrieved_reports_info, # Pass the list of dicts
            output_format="df", 
            include_financial_data=True
            # news_type_override is handled internally by run_all based on 'doc_type' in files_info
        )

        # Logic for handling multiple results and caching the first good one.
        first_successful_df_saved_to_cache = False
        processed_data_for_dashboards = [] # To store (data_row, report_info_dict) for dashboard generation
        all_successful_dataframes_this_run = [] # New list to collect all successful DFs

        for idx, df_result in enumerate(processed_results_list):
            # Add a more stringent check: ensure the DataFrame is not all NaNs
            if df_result is not None and not df_result.empty and not df_result.isna().all().all():
                original_report_meta = retrieved_reports_info[idx]
                logger.info(f"Successfully processed data for report: {original_report_meta.get('filepath')}. DataFrame shape: {df_result.shape}")
                
                all_successful_dataframes_this_run.append(df_result) # Collect all good DFs
                
                # Store data for generating a dashboard for this specific report
                if not df_result.empty:
                    processed_data_for_dashboards.append({'data_row': df_result.iloc[0], 'meta': original_report_meta})
            else:
                if idx < len(retrieved_reports_info):
                    logger.warning(f"ETL Pipeline did not produce data for report: {retrieved_reports_info[idx].get('filepath')}")
                else:
                    logger.warning("ETL Pipeline did not produce data for a report (index out of bounds for retrieved_reports_info).")

        # After processing all reports, if any were successful, concatenate and save them.
        if all_successful_dataframes_this_run:
            if len(all_successful_dataframes_this_run) == 1:
                concatenated_df = all_successful_dataframes_this_run[0].copy()
            else:
                # This specific warning is about future changes in how pandas handles
                # dtype inference during concatenation when some dataframes have all-NA columns
                # that align with data-filled columns in other dataframes.
                # We suppress it as the current behavior is accepted.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
                        category=FutureWarning
                    )
                    concatenated_df = pd.concat(all_successful_dataframes_this_run, ignore_index=True)
            
            logger.info(f"Concatenated {len(all_successful_dataframes_this_run)} processed reports into a single DataFrame. Shape: {concatenated_df.shape}")
            try:
                concatenated_df.to_csv(cache_csv_path, index=False)
                logger.info(f"All processed data from this run saved to cache CSV: {cache_csv_path}")
                processed_data_df = concatenated_df # This is now the main df for this run
            except Exception as e:
                logger.error(f"Failed to save concatenated DataFrame to cache CSV {cache_csv_path}: {e}")
        # If `all_successful_dataframes_this_run` is empty, but `processed_data_for_dashboards` isn't (should not happen with current logic),
        # or more importantly, if no new data was processed, `processed_data_df` remains as loaded from cache or None.

        if not processed_data_for_dashboards and not all_successful_dataframes_this_run:
            # This means no reports were processed successfully in this run, 
            # AND no prior cache was loaded that resulted in dashboard data.
            # (The processed_data_for_dashboards gets populated only from current run data)
            logger.error(f"ETL Pipeline did not produce any processable data for {ticker_symbol} in this run. Dashboards cannot be generated.")
            logger.info(f"--- Pipeline finished for ticker: {ticker_symbol} ---")
            return
        
        # If the main `processed_data_df` is still None (e.g. initial cache load failed or cache was empty, 
        # AND all_successful_dataframes_this_run was empty), but somehow processed_data_for_dashboards has items (edge case, defensive)
        # OR, more likely: ensure processed_data_df reflects the current run if it happened.
        if processed_data_df is None and all_successful_dataframes_this_run: # If cache was empty/not loaded, but we processed new data
             processed_data_df = pd.concat(all_successful_dataframes_this_run, ignore_index=True) # Re-assign if it was missed.

    # End of the block for `if processed_data_df is None or processed_data_df.empty:`
    # Now, generate dashboards.
    # If `processed_data_df` was loaded from the main cache, `processed_data_for_dashboards` will be empty.
    # In this case, we generate one dashboard from `processed_data_df`.
    # Otherwise, `processed_data_for_dashboards` has items from the current run.

    if not processed_data_for_dashboards: # True if data was loaded from main cache
        if processed_data_df is not None and not processed_data_df.empty:
            logger.info("Generating single dashboard from cached main processed data.")
            data_row_for_dashboard = processed_data_df.iloc[0]
            # Generic dashboard name for the cached data
            dashboard_image_filename = f"{safe_ticker_filename}_dashboard.png"
            dashboard_image_path = os.path.join(abs_download_dir, dashboard_image_filename)
            try:
                create_dashboard_visualization(
                    df_row=data_row_for_dashboard, 
                    ticker=ticker_symbol, 
                    output_image_path=dashboard_image_path
                )
            except Exception as e:
                logger.error(f"Failed to generate or save dashboard visualization from cached data: {e}", exc_info=True)
        else:
            logger.error(f"No data available for dashboard generation for ticker {ticker_symbol}.")
    else: # Data was processed in the current run, generate dashboard for each
        logger.info(f"Generating {len(processed_data_for_dashboards)} dashboard(s) from current run...")
        for item in processed_data_for_dashboards:
            data_row = item['data_row']
            meta = item['meta']
            doc_type_str = meta.get('doc_type', 'UnknownType')
            accession_str = meta.get('accession_number', 'UnknownAccession').replace('-','')[:15] # Sanitize accession
            
            dashboard_image_filename_specific = f"{safe_ticker_filename}_{doc_type_str}_{accession_str}_dashboard.png"
            dashboard_image_path_specific = os.path.join(abs_download_dir, dashboard_image_filename_specific)
            
            logger.info(f"Generating dashboard for report: {meta.get('filepath')} -> {dashboard_image_path_specific}")
            try:
                create_dashboard_visualization(
                    df_row=data_row, 
                    ticker=ticker_symbol, 
                    output_image_path=dashboard_image_path_specific
                )
            except Exception as e:
                logger.error(f"Failed to generate or save dashboard for {meta.get('filepath')}: {e}", exc_info=True)

    logger.info(f"--- Pipeline finished for ticker: {ticker_symbol} ---")


def main():
    parser = argparse.ArgumentParser(description="Run the MineCast data processing and dashboard generation pipeline.")
    parser.add_argument("--ticker",type=str, help="Stock ticker symbol (e.g., 'GOLD') to process.")
    args = parser.parse_args()

    run_full_pipeline(args.ticker)

if __name__ == "__main__":
    main()