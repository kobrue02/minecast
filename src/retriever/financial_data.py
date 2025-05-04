import yfinance as yf
import pandas as pd
from pandas._libs.tslibs.parsing import DateParseError

class FinancialDataRetriever:

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

    def get_financial_data(self, ticker):
        data = yf.Ticker(ticker) 
        info = data.info
        current_price = info.get("currentPrice")
        market_cap = info.get("marketCap")
        shares_outstanding = info.get("sharesOutstanding")
        currency = info.get("currency")
        df = pd.DataFrame({
            "current_price": [current_price],
            "currency": [currency],
            "market_cap": [market_cap],
            "shares_outstanding": [shares_outstanding],
            "ticker": [ticker]
        })
        print(df)
        return df

    def get_balance_sheet(self, ticker: str):
        """
        Fetches the balance sheet for a given stock ticker.
        
        Args:
            ticker (str): The stock ticker symbol.
        
        Returns:
            DataFrame: A DataFrame containing the balance sheet data.
        """
        ticker_data = yf.Ticker(ticker)
        balance_sheet = ticker_data.balance_sheet
        balance_sheet = self._prep_df(balance_sheet)
        balance_sheet['ticker'] = [ticker]
        columns = [
            "ticker",
            "Cash And Cash Equivalents",
            "Total Debt",
            "Current Debt",
            "Tangible Book Value",
            "Working Capital",
            "Net Tangible Assets"
        ]
        # keep only the columns that are in the list
        return balance_sheet[columns]
    
    def get_major_holders(self, ticker: str):
        """
        Fetches major holders for a given stock ticker.
        
        Args:
            ticker (str): The stock ticker symbol.
        
        Returns:
            DataFrame: A DataFrame containing the major holders data.
        """
        ticker_data = yf.Ticker(ticker)
        major_holders = ticker_data.major_holders
        major_holders = self._prep_df(major_holders)
        major_holders['ticker'] = [ticker]
        return major_holders
    
    def run(self, ticker):
        """
        Fetches and prints the balance sheet, insider transactions, and major holders for a given stock ticker.
        
        Args:
            ticker (str): The stock ticker symbol.
        """
        balance_sheet = self.get_balance_sheet(ticker)
        major_holders = self.get_major_holders(ticker)
        financial_data = self.get_financial_data(ticker)
        # merge on "ticker" column
        balance_sheet = balance_sheet.reset_index()
        balance_sheet = balance_sheet.merge(major_holders, on="ticker", how="left")
        balance_sheet = balance_sheet.merge(financial_data, on="ticker", how="left")
        return balance_sheet

if __name__ == "__main__":
    retriever = FinancialDataRetriever()
    ticker = "WRLG.V"
    balance_sheet = retriever.run(ticker)
    print(balance_sheet)