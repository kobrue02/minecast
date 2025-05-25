import yfinance as yf
import pandas as pd
import logging
from pandas._libs.tslibs.parsing import DateParseError

logger = logging.getLogger(__name__)

class FinancialDataRetriever:
    # Define expected columns for each financial statement to ensure consistency
    EXPECTED_BALANCE_SHEET_COLS = [
        "Cash And Cash Equivalents", "Total Debt", "Current Debt", 
        "Tangible Book Value", "Working Capital", "Net Tangible Assets"
    ]
    # Add other sets of columns if needed for financials, cashflow etc.

    def __init__(self):
        """
        Initialize the FinancialDataRetriever class.
        """
        pass

    def _prep_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the DataFrame by transposing it, resetting the index, and renaming columns.

        Args:
            df (pd.DataFrame): The DataFrame to be prepared.
        
        Returns:
            pd.DataFrame: The prepared DataFrame.
        """
        df = df.iloc[:, :1]
        df = df.transpose()
        df = df.reset_index()
        try:
            df.columns = ['Date'] + list(df.columns[1:])
            df['Date'] = pd.to_datetime(df['Date'])
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df = df.set_index('Date')
        except DateParseError:
            pass
        df = df.sort_index(ascending=False)
        # try to typecast all columns to float except string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass
        return df

    def _prep_financial_statement_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if df is None or df.empty:
            logger.warning(f"Input DataFrame for ticker {ticker} is empty in _prep_financial_statement_df.")
            return pd.DataFrame() # Return an empty DataFrame

        # Attempt to take only the first data column if multiple date columns exist
        # This assumes the latest data is in the first column after yfinance call
        if not df.empty and len(df.columns) > 0:
            df_transposed = df.iloc[:, [0]].T # Select first column, then transpose
        else:
            df_transposed = pd.DataFrame()

        # df_transposed will have the original index (e.g. 'Cash And Cash Equivalents') as columns
        # and the date as the index. Reset index to make date a column.
        df_transposed = df_transposed.reset_index()
        
        if 'index' in df_transposed.columns:
            df_transposed.rename(columns={'index': 'ReportDate'}, inplace=True)
            try:
                df_transposed['ReportDate'] = pd.to_datetime(df_transposed['ReportDate']).dt.strftime('%Y-%m-%d')
            except Exception as e:
                logger.warning(f"Could not parse ReportDate for {ticker}: {e}. Keeping original.")
        
        # Add ticker column
        df_transposed['ticker'] = ticker
        return df_transposed

    def get_ticker_info_data(self, ticker: str) -> pd.DataFrame:
        logger.info(f"Fetching ticker info for {ticker}")
        try:
            data = yf.Ticker(ticker)
            info = data.info
            if not info or info.get('regularMarketPrice') is None: # Check if info is sparse or essentially empty
                logger.warning(f"No substantial info found for ticker {ticker} or 'regularMarketPrice' is missing.")
                # Return DataFrame with expected structure but NaNs
                return pd.DataFrame({
                    "current_price": [None], "currency": [None], "market_cap": [None],
                    "shares_outstanding": [None], "enterprise_value": [None],
                    "beta": [None], "trailing_pe": [None], "forward_pe": [None],
                    "ticker": [ticker]
                })

            current_price = info.get("currentPrice", info.get("regularMarketPrice")) # currentPrice can be None
            df = pd.DataFrame({
                "current_price": [current_price],
                "currency": [info.get("currency")],
                "market_cap": [info.get("marketCap")],
                "shares_outstanding": [info.get("sharesOutstanding")],
                "enterprise_value": [info.get("enterpriseValue")],
                "beta": [info.get("beta")],
                "trailing_pe": [info.get("trailingPE")],
                "forward_pe": [info.get("forwardPE")],
                "ticker": [ticker]
            })
            return df
        except Exception as e:
            logger.error(f"Error fetching ticker info for {ticker}: {e}", exc_info=True)
            return pd.DataFrame({"ticker": [ticker]}) # Minimal DataFrame for merge

    def get_balance_sheet(self, ticker: str) -> pd.DataFrame:
        logger.info(f"Fetching balance sheet for {ticker}")
        try:
            ticker_data = yf.Ticker(ticker)
            balance_sheet_raw = ticker_data.balance_sheet

            if balance_sheet_raw is None or balance_sheet_raw.empty:
                logger.warning(f"Balance sheet data is empty for ticker: {ticker}")
                # Return an empty DataFrame with expected columns + ticker
                df_empty = pd.DataFrame(columns=self.EXPECTED_BALANCE_SHEET_COLS + ['ticker', 'ReportDate'])
                df_empty['ticker'] = ticker
                return df_empty

            # Process the raw balance sheet (assuming it's not empty)
            # yfinance balance_sheet has items as index, dates as columns.
            # We typically want the latest balance sheet (first column).
            
            # Take only the latest available balance sheet data (first column)
            latest_balance_sheet = balance_sheet_raw.iloc[:, [0]]
            
            # Transpose so that items become columns, date becomes index
            df_processed = latest_balance_sheet.T 
            df_processed.reset_index(inplace=True) # Date becomes a column
            df_processed.rename(columns={'index': 'ReportDate'}, inplace=True)
            try:
                df_processed['ReportDate'] = pd.to_datetime(df_processed['ReportDate']).dt.strftime('%Y-%m-%d')
            except Exception as e:
                logger.warning(f"Could not parse ReportDate for balance sheet {ticker}: {e}")

            df_processed['ticker'] = ticker

            # Ensure all expected columns exist, fill with None if not
            for col in self.EXPECTED_BALANCE_SHEET_COLS:
                if col not in df_processed.columns:
                    df_processed[col] = None
            
            # Select only expected columns + ticker and ReportDate
            final_cols = ['ticker', 'ReportDate'] + self.EXPECTED_BALANCE_SHEET_COLS
            return df_processed[final_cols]

        except Exception as e:
            logger.error(f"Error fetching or processing balance sheet for {ticker}: {e}", exc_info=True)
            df_empty = pd.DataFrame(columns=self.EXPECTED_BALANCE_SHEET_COLS + ['ticker', 'ReportDate'])
            df_empty['ticker'] = ticker
            return df_empty

    def get_major_holders(self, ticker: str) -> pd.DataFrame:
        logger.info(f"Fetching major holders for {ticker}")
        try:
            ticker_data = yf.Ticker(ticker)
            major_holders_raw = ticker_data.major_holders

            if major_holders_raw is None or major_holders_raw.empty:
                logger.warning(f"Major holders data is empty for ticker: {ticker}")
                # Return df with just ticker col to allow merge
                return pd.DataFrame({'ticker': [ticker], '% Out': [None], 'Shares': [None], 'Date Reported': [None], 'Holder': [None]}) 
            
            # yfinance major_holders is usually a DataFrame with columns like:
            # 0 % Out | 1 Shares | 2 Date Reported | 3 Holder
            # We want to make 'Holder' more specific if possible (e.g., "Institutions", "Individuals")
            # For now, let's pivot or restructure if needed, or just return as is with ticker
            
            # The structure is typically:
            #                  0              1
            # 0  % of Shares Held by All Insider...         1.23%
            # 1  % of Shares Held by Institutions...       45.67%
            # Let's try to make this more usable, or just pass it through carefully
            # For simplicity now, pass through and let it be handled in merge or later processing
            # The current _prep_df is not suitable.
            
            # Let's try to flatten it or select key pieces if structure is consistent
            # For now, we'll just add a ticker and return.
            # A more robust parsing might be needed if this data is critical and its structure varies.
            
            # A simple approach if it has two columns [Value, Label]
            # Try to rename columns if they are just numbers
            if all(isinstance(col, int) for col in major_holders_raw.columns) and len(major_holders_raw.columns) == 2:
                 major_holders_raw.columns = ['Value', 'Description'] # Example rename

            major_holders_raw['ticker'] = ticker
            return major_holders_raw

        except Exception as e:
            logger.error(f"Error fetching major holders for {ticker}: {e}", exc_info=True)
            return pd.DataFrame({'ticker': [ticker]}) # Minimal for merge

    def run(self, ticker: str) -> pd.DataFrame:
        logger.info(f"--- Running FinancialDataRetriever for ticker: {ticker} ---")
        
        balance_sheet_df = self.get_balance_sheet(ticker)
        major_holders_df = self.get_major_holders(ticker) # This might be multi-row, need care in merge
        ticker_info_df = self.get_ticker_info_data(ticker)

        # Start with ticker_info_df as it's always one row per ticker
        merged_df = ticker_info_df

        # Merge balance sheet (should also be one row for the latest after processing)
        if not balance_sheet_df.empty:
            # Ensure 'ticker' is a column in balance_sheet_df for merging
            if 'ticker' not in balance_sheet_df.columns: # Should be there now
                 balance_sheet_df['ticker'] = ticker
            merged_df = pd.merge(merged_df, balance_sheet_df, on="ticker", how="left", suffixes=('', '_bs'))
        else:
            logger.warning(f"Balance sheet data for {ticker} was empty, skipping merge.")
            # Add empty columns if needed for schema consistency downstream
            for col in self.EXPECTED_BALANCE_SHEET_COLS + ['ReportDate_bs']: # Add suffix if used
                 if col not in merged_df.columns: merged_df[col] = None


        # Major holders can be tricky as it might be multiple rows.
        # For now, if we want to add it, we'd need to decide how.
        # E.g., pivot it, or only take specific rows like '% of Shares Held by Institutions'.
        # A simple left merge might create duplicate rows in merged_df if major_holders_df has multiple rows for the ticker.
        # For this iteration, I will log it and not merge it directly into the single-row-per-ticker structure
        # unless a clear single-row representation is derived.
        if not major_holders_df.empty and 'ticker' in major_holders_df.columns:
            logger.info(f"Major holders data for {ticker} (shape: {major_holders_df.shape}):\n{major_holders_df.head()}")
            # Example: To merge, one might pivot major_holders_df first or select specific fields.
            # For now, not merging it to keep merged_df as one row per ticker.
            # If specific single values are needed, they should be extracted and added.
            # Example of extracting institutional ownership if available and structured:
            try:
                if 'Description' in major_holders_df.columns and 'Value' in major_holders_df.columns:
                    inst_own = major_holders_df[major_holders_df['Description'].str.contains("Institutions", na=False)]['Value'].iloc[0]
                    merged_df['institutional_ownership_percent'] = inst_own
            except Exception:
                logger.debug(f"Could not extract institutional ownership for {ticker}")
                merged_df['institutional_ownership_percent'] = None
        else:
            logger.warning(f"Major holders data for {ticker} was empty or unmergeable, skipping.")
            merged_df['institutional_ownership_percent'] = None


        # If 'ReportDate_bs' was created from balance_sheet merge, rename it for clarity
        if 'ReportDate_bs' in merged_df.columns and 'ReportDate' not in merged_df.columns:
            merged_df.rename(columns={'ReportDate_bs': 'BalanceSheet_ReportDate'}, inplace=True)
        elif 'ReportDate' in merged_df.columns and balance_sheet_df.empty : # If no BS data, no BS report date
            pass # ReportDate from ticker_info might be different context

        logger.info(f"Final merged financial data for {ticker} (shape: {merged_df.shape}):\n{merged_df.head().to_string()}")
        return merged_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Ensure logger is configured for direct script run
    logger.info("Starting FinancialDataRetriever test...")
    retriever = FinancialDataRetriever()
    
    test_tickers = ["AAPL", "MSFT", "GOOG", "NONEXISTENTTICKER", "WRLG.V", "AG", "USAU"] 
    # WRLG.V is a very small cap, likely sparse data
    # AG (First Majestic) and USAU (US Gold Corp) are mining tickers
    
    all_results = []
    for ticker_symbol in test_tickers:
        print(f"\n--- Testing ticker: {ticker_symbol} ---")
        try:
            data = retriever.run(ticker_symbol)
            if data is not None and not data.empty:
                print(f"Successfully retrieved data for {ticker_symbol}:")
                # print(data.to_string())
                all_results.append(data)
            else:
                print(f"No data or empty data returned for {ticker_symbol}")
        except Exception as e:
            print(f"An error occurred while processing {ticker_symbol}: {e}")
            logger.error(f"Error in main test loop for {ticker_symbol}: {e}", exc_info=True)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        print("\n--- Combined Results ---")
        print(final_df.to_string())
        # final_df.to_csv("financial_data_test_output.csv", index=False)
        # logger.info("Saved combined results to financial_data_test_output.csv")
    else:
        print("\n--- No results to combine ---")

    logger.info("FinancialDataRetriever test finished.")