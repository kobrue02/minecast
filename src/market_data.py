import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
from yfinance.exceptions import YFRateLimitError # YFDownloadError might not be directly available
# import requests # Import requests if you want to catch requests.exceptions.HTTPError specifically

app = Flask(__name__)
CORS(app)

@app.route('/market_data/<ticker>')
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get('regularMarketPrice') is None: # Using a common field like regularMarketPrice
            # This condition handles tickers that exist but have no market data (e.g., very old or delisted)
            # or if yfinance returns an empty info dict for an invalid ticker.
            return jsonify({"error": f"No market data found for ticker {ticker}. It might be invalid, delisted, or lack recent data."}), 404

        data = {
            'market_cap': info.get('marketCap'),
            'total_cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'shares_outstanding': info.get('sharesOutstanding')
        }
        return jsonify(data)
    except YFRateLimitError:
        return jsonify({"error": "Rate limit exceeded with yfinance API. Please try again in a few minutes."}), 429
    # except requests.exceptions.HTTPError as e: # Optional: if you want to specifically catch HTTP errors from underlying requests
    #     if e.response.status_code == 404:
    #         return jsonify({"error": f"Data not found for ticker {ticker} (404). It may be invalid or delisted."}), 404
    #     return jsonify({"error": f"HTTP error fetching data for {ticker}: {e.response.status_code}"}), e.response.status_code
    except Exception as e:
        print(f"An unexpected error occurred while fetching market data for {ticker}: {str(e)}")
        # Check if the error message string itself indicates a 404 from yfinance internals
        if "404 Client Error" in str(e) or "No data found for ticker" in str(e):
             return jsonify({"error": f"No data found for ticker {ticker}. It might be invalid or delisted."}), 404
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/historical_data/<ticker>')
def get_historical_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist_period = request.args.get('period', '1y')
        hist_data = stock.history(period=hist_period)

        if hist_data.empty:
            # This is a reliable way yfinance indicates no historical data for the period/ticker.
            return jsonify({"error": f"No historical data found for ticker {ticker} for period '{hist_period}'. It might be an invalid ticker or has no data for this range."}), 404

        hist_data_json = hist_data.reset_index().to_dict(orient='records')
        
        for record in hist_data_json:
            if 'Date' in record and hasattr(record['Date'], 'strftime'):
                record['Date'] = record['Date'].strftime('%Y-%m-%d')
            
        return jsonify(hist_data_json)
    except YFRateLimitError:
        return jsonify({"error": "Rate limit exceeded with yfinance API. Please try again in a few minutes."}), 429
    except Exception as e:
        print(f"An unexpected error occurred while fetching historical data for {ticker}: {str(e)}")
        if "404 Client Error" in str(e) or "No data found for ticker" in str(e):
            return jsonify({"error": f"No data found for ticker {ticker}. It might be invalid or delisted."}), 404
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 