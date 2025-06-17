from typing import Any, Dict, List, Optional, Union
from src.etl.extractor.types import NewsType
from src.etl.extractor.schemas import (
    DrillingNewsRelease,
    Corp10K,
    CorporateUpdateRelease,
    ProjectUpdateRelease,
    ResourceUpdateRelease,
    TechnicalReportData
)
from src.etl.extractor.open_ai_api_client import OpenAIClient
from src.etl.extractor.news_type_classifier import FastEmbedNewsTypeClassifier
from src.etl.extractor.pdf_loader import PDFLoader
from src.etl.extractor.htm_processor import HtmProcessor
from src.retriever.financial_data import FinancialDataRetriever

import logging
import json
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)

# Mapping from Edgar.py string doc_types to NewsType Enum
EDGAR_DOC_TYPE_TO_NEWSTYPE_MAP = {
    "FS": NewsType.FS,
    "PFS": NewsType.PFS,
    "PEA": NewsType.PEA,
    "SK1300": NewsType.SK1300,
    "TRS": NewsType.TRS
    # Add more mappings if Edgar.py produces other string types
}

class Pipeline:
    """
    Main pipeline class for extracting structured data from mining company news releases and technical reports.
    """

    def __init__(self):
        """
        Initialize the pipeline with necessary components.
        """
        self.logger = logging.getLogger(__name__)
        self.extractor = OpenAIClient(logger=self.logger, vebosity=1)
        self.news_type_classifier = FastEmbedNewsTypeClassifier.load_model("models/fastEmbedModel.pkl")
        self.db = None
        self.financial_data_retriever = FinancialDataRetriever()
        self.logger.info("Pipeline initialized.")

    def _clean_text_content(self, text_content: str) -> str:
        """
        Clean the extracted text content by removing unwanted characters and normalizing whitespace.
        This can be a generic cleaner for text from PDF or HTM.
        Args:
            text_content (str): Extracted text
        Returns:
            str: Cleaned text
        """
        cleaned_text = text_content.replace("\n", " ").replace("\r", "")
        # remove extra spaces around punctuation
        for punct in [",", ".", ";", ":", "!", "?"]:
            cleaned_text = cleaned_text.replace(f" {punct}", punct)
        cleaned_text = cleaned_text.replace("  ", " ") # Collapse multiple spaces
        cleaned_text = cleaned_text.replace(" ”", "”")
        cleaned_text = " ".join(cleaned_text.split()) # Normalize all whitespace to single spaces
        return cleaned_text

    def _structured_output_to_df(self, structured_data, ticker: Optional[str]) -> Optional[pd.DataFrame]:
        if not structured_data:
            self.logger.warning("No structured data to convert to DataFrame.")
            return None
        try:
            json_data = structured_data.model_dump_json()
            py_dict = json.loads(json_data)
            if ticker:
                py_dict["ticker"] = ticker # Add ticker to the root of the dict
            df = pd.json_normalize(py_dict)
            return df
        except Exception as e:
            self.logger.error(f"Error converting structured data to DataFrame: {e}", exc_info=True)
            return None

    def run(self, file_path: str, news_type_override: Union[NewsType, str] = None, output_format: str = None, include_financial_data: bool = True, ticker: Optional[str] = None) -> Union[Dict[str, Any], pd.DataFrame, None]:
        """
        Run the pipeline to extract structured data from a given file (PDF or HTM).

        Args:
            file_path (str): Path to the input file (PDF or HTM).
            news_type_override (Union[NewsType, str], optional): Manually override news type. 
                                                             Can be a NewsType enum or a string from Edgar.py.
                                                             Defaults to None.
            output_format (str, optional): "df" for DataFrame, "json" for JSON dict. Defaults to None (returns Pydantic model).
            include_financial_data (bool): Whether to add financial data if output is DataFrame and ticker is provided. Defaults to True.
            ticker (str, optional): Ticker symbol associated with the file. Defaults to None.
        """
        self.logger.info(f"Starting pipeline run for file: {file_path}, Ticker: {ticker}, NewsType Override: {news_type_override}")
        if ticker:
            self._ticker = ticker 
        else:
            self._ticker = None # Ensure it's reset if not provided for this run

        file_extension = os.path.splitext(file_path)[1].lower()
        text_content_for_llm = None
        article_title_for_classification = file_path # Fallback for title
        final_news_type: Optional[NewsType] = None

        # First, determine the final news type from the override
        if isinstance(news_type_override, NewsType):
            final_news_type = news_type_override
        elif isinstance(news_type_override, str):
            final_news_type = EDGAR_DOC_TYPE_TO_NEWSTYPE_MAP.get(news_type_override)
            if not final_news_type:
                self.logger.warning(f"Unknown string news_type_override \'{news_type_override}\'. Type will be classified if possible.")

        if file_extension == '.pdf':
            self.logger.info("Processing PDF file.")
            pdf_loader = PDFLoader(file_path, read_only=True)
            article_title_for_classification = pdf_loader.get_title() or os.path.basename(file_path)
            
            # Only classify if no news type was provided
            if not final_news_type:
                try:
                    final_news_type = self.news_type_classifier.inference(article_title_for_classification)
                    self.logger.info(f"Classified PDF news type: {final_news_type} for title: '{article_title_for_classification}'")
                except Exception as e:
                    self.logger.error(f"Error classifying PDF news type: {e}. Will attempt generic extraction.")
                    return None  # Return None if classification fails and no type was provided

            # Load PDF content based on the news_type
            if final_news_type in [NewsType.PEA, NewsType.PFS, NewsType.FS, NewsType.NI, NewsType.SK1300]:
                raw_text = pdf_loader.load_pfs()
            else:
                raw_text = pdf_loader.load_corporate_announcement()
            
            if not raw_text:
                self.logger.error(f"Failed to extract text from PDF: {file_path}")
                return None
            text_content_for_llm = self._clean_text_content(raw_text)

        elif file_extension in ['.htm', '.html']:
            self.logger.info("Processing HTM file.")
            try:
                htm_processor = HtmProcessor(file_path)
                if not final_news_type:
                    self.logger.error(f"News type for HTM file {file_path} not provided and not overridden. Cannot proceed with LLM extraction without a type.")
                    return None
                
                self.logger.info(f"Using NewsType for HTM: {final_news_type}")
                text_content_for_llm = htm_processor.get_processed_content_for_llm()
                article_title_for_classification = htm_processor.soup.title.string if htm_processor.soup and htm_processor.soup.title else os.path.basename(file_path)
            except Exception as e:
                self.logger.error(f"Error processing HTM file {file_path}: {e}", exc_info=True)
                return None
        else:
            self.logger.error(f"Unsupported file type: {file_extension}. Please provide a PDF or HTM file.")
            return None

        if not text_content_for_llm:
            self.logger.error(f"Failed to extract text content from: {file_path}")
            return None
        
        if not final_news_type:
            self.logger.error(f"News type could not be determined for {file_path}. Extraction cannot proceed without a type.")
            return None 

        self.logger.info(f"Cleaned text length for LLM: {len(text_content_for_llm)}. Title for classification: '{article_title_for_classification}'. Final NewsType: {final_news_type}")

        try:
            structured_data_model = self.extractor.run(text_content_for_llm, news_type=final_news_type)
        except ValueError as ve:
            self.logger.error(f"OpenAI client run failed, possibly due to unmapped NewsType or schema issue for {final_news_type}: {ve}")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during OpenAI extraction for {final_news_type}: {e}", exc_info=True)
            return None

        if not structured_data_model:
            self.logger.error("OpenAI client did not return structured data.")
            return None
        
        self.logger.info(f"Successfully extracted structured data using schema: {type(structured_data_model).__name__}")

        if output_format:
            df_output = self._structured_output_to_df(structured_data_model, ticker)
            if output_format == "df":
                if include_financial_data and ticker and df_output is not None:
                    df_output = self.add_financial_data_to_df(df_output, ticker)
                    self.logger.info(f"Added financial data to DataFrame for ticker: {ticker}")
                return df_output
            elif output_format == "json":
                if df_output is not None:
                    return df_output.to_dict(orient='records') 
                else:
                    return json.loads(structured_data_model.model_dump_json()) 
            else:
                self.logger.warning(f"Unsupported output format: {output_format}. Returning Pydantic model.")
                return structured_data_model
        
        return structured_data_model
    
    def add_financial_data_to_df(self, df: pd.DataFrame, ticker) -> pd.DataFrame:
        """
        Add financial data to the DataFrame.
        """
        if df is None or df.empty:
            self.logger.warning("Input DataFrame is empty, cannot add financial data.")
            return df

        ticker_data = self.financial_data_retriever.run(ticker)
        if ticker_data is not None and not ticker_data.empty:
            self.logger.info(f"Financial data retrieved for ticker: {ticker}")
            # Ensure ticker column exists in df for merging if not already present from _structured_output_to_df
            if 'ticker' not in df.columns and hasattr(self, '_ticker') and self._ticker == ticker:
                 df['ticker'] = self._ticker
            elif 'ticker' not in df.columns:
                self.logger.warning(f"'ticker' column missing in main DataFrame, cannot merge financial data reliably.")
                return df

            ticker_data = ticker_data.reset_index() # Assuming ticker is in index or a column
            # If 'ticker' is an index in ticker_data, reset_index makes it a column.
            # If 'ticker' is already a column, reset_index might add an 'index' column we don't need for merge.
            # Ensure 'ticker' column exists in ticker_data before merge
            if 'ticker' not in ticker_data.columns:
                self.logger.error(f"Financial data for {ticker} does not contain a 'ticker' column for merging.")
                return df # Return original df
            
            # Perform merge, ensuring no duplicate columns from ticker_data other than 'ticker' key
            # Suffix duplicate columns from financial data to avoid clashes if they accidentally exist in main df
            merged_df = pd.merge(df, ticker_data, on="ticker", how="left", suffixes=('', '_financial'))
            return merged_df
        else:
            self.logger.warning(f"No financial data found or data is empty for ticker: {ticker}. Returning original DataFrame.")
            return df # Return original DataFrame

    def run_all(self, files_info: List[Dict[str, str]], news_type_override: NewsType = None, output_format: str = None, include_financial_data: bool = True) -> list:
        """
        Run the pipeline for a list of files, each associated with a ticker and potentially a doc_type.

        Args:
            files_info (List[Dict[str, str]]): List of dicts, each like 
                                                {'file_path': 'path/to/file', 'ticker': 'TICKER', 'doc_type': 'FS' (optional)}.
            news_type_override, output_format, include_financial_data: Same as in run(). 
                                                                     Note: news_type_override here is a general override for all files if set.
                                                                     The 'doc_type' in files_info is per-file.
        """
        results = []
        for item in files_info:
            file_path = item.get('filepath')
            ticker = item.get('ticker')
            # Use doc_type from item if available, otherwise fallback to general news_type_override
            # The `run` method's news_type_override can handle the string from item['doc_type']
            per_file_news_type_override = item.get('doc_type', news_type_override) 

            if not file_path or not ticker:
                self.logger.warning(f"Skipping item due to missing file_path or ticker: {item}")
                continue
            
            self.logger.info(f"--- Running pipeline for file: {file_path}, ticker: {ticker}, doc_type from item: {item.get('doc_type')} ---")
            try:
                result = self.run(
                    file_path=file_path, 
                    news_type_override=per_file_news_type_override, # Pass the specific doc_type here
                    output_format=output_format, 
                    include_financial_data=include_financial_data, 
                    ticker=ticker
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Pipeline run failed for {file_path} (ticker: {ticker}): {e}", exc_info=True)
                results.append(None) # Add None or an error marker for this run
        return results

# Example of how to run the pipeline (for testing purposes)
if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    # Ensure logger is set to DEBUG for detailed output when running this script directly
    if not logger.handlers or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    pipeline = Pipeline()

    # Create a dummy HTM file (as created by HtmProcessor test, or use a real one from Edgar)
    dummy_htm_file = "_test_technical_report.htm"
    # Ensure this file exists or is created by your Edgar script, e.g. from ticker USAU / Accession: 0001437749-24-006300, exhibit ex_99-1.htm
    # For this example, let's assume a simple one exists or gets created by previous Edgar script run.
    # If not, create a simple one for pipeline testing:
    if not os.path.exists(dummy_htm_file):
        with open(dummy_htm_file, "w", encoding="utf-8") as f_dummy:
            f_dummy.write("""
            <html><head><title>Test S-K 1300 Report for USAU</title></head>
            <body>
                <h1>Test S-K 1300 Technical Report Summary</h1>
                <p>This is an S-K 1300 technical report summary for the <b>USAU Gold Project</b>, effective Jan 1, 2024.</p>
                <p>The project shows promising results with an estimated NPV of $150M and IRR of 22%.</p>
                <p>Key qualified person is John Doe (P.Geo).</p>
                <h2>Resource Summary</h2>
                <table><thead><tr><th>Category</th><th>Tonnes (Mt)</th><th>Grade (g/t Au)</th><th>Contained Au (koz)</th></tr></thead>
                    <tbody><tr><td>Indicated</td><td>25</td><td>1.3</td><td>1045</td></tr></tbody></table>
            </body></html>
            """)

    # Test with an HTM file from Edgar (assuming ticker is known)
    test_files_info = [
        {'file_path': dummy_htm_file, 'ticker': 'USAU'} 
        # Add more files, including PDFs, if you have them for testing
        # {'file_path': 'path/to/some.pdf', 'ticker': 'OTHERTICKER'}
    ]

    results_from_run_all = pipeline.run_all(test_files_info, output_format="df") # Get DataFrames

    for i, result_df in enumerate(results_from_run_all):
        if result_df is not None:
            print(f"\n--- Result for {test_files_info[i]['file_path']} (Ticker: {test_files_info[i]['ticker']}) ---")
            print(result_df.to_string())
        else:
            print(f"\n--- No result or error for {test_files_info[i]['file_path']} (Ticker: {test_files_info[i]['ticker']}) ---")

    # Example for a single run (more direct)
    # single_result = pipeline.run(file_path=dummy_htm_file, ticker='USAU', output_format='json')
    # if single_result:
    #     print("\n--- Single Run Result (JSON) ---")
    #     print(json.dumps(single_result, indent=2))