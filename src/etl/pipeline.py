from src.etl.extractor.types import NewsType
from src.etl.extractor.schemas import (
    DrillingNewsRelease,
    Corp10K,
    CorporateUpdateRelease,
    ProjectUpdateRelease,
    ResourceUpdateRelease
)
from src.etl.extractor.open_ai_api_client import OpenAIClient
from src.etl.extractor.news_type_classifier import FastEmbedNewsTypeClassifier
from src.etl.extractor.pdf_loader import PDFLoader
from src.retriever.financial_data import FinancialDataRetriever

import logging
import json
import pandas as pd

logging.basicConfig(level=logging.INFO)


class Pipeline:
    """
    Main pipeline class for extracting structured data from mining company news releases.
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

    def _clean_pdf_text(self, pdf_text: str) -> tuple[str]:
        """
        Clean the extracted PDF text by removing unwanted characters.

        Args:
            pdf_text (str): Extracted text from the PDF

        Returns:
            str: Cleaned text
        """
        # Remove unwanted characters and whitespace
        cleaned_text = pdf_text.replace("\n", " ").replace("\r", "")
        
        # remove extra spaces around commas or periods
        cleaned_text = cleaned_text.replace(" ,", ",")
        cleaned_text = cleaned_text.replace(" .", ".")
        cleaned_text = cleaned_text.replace(" ;", ";")
        cleaned_text = cleaned_text.replace(" :", ":")
        cleaned_text = cleaned_text.replace(" !", "!")
        cleaned_text = cleaned_text.replace(" ?", "?")
        cleaned_text = cleaned_text.replace("  ", " ")
        cleaned_text = cleaned_text.replace(" ”", "”")

        # Remove extra spaces
        cleaned_text = " ".join(cleaned_text.split())
        return cleaned_text

    def _structured_output_to_df(self, structured_data: Corp10K | ProjectUpdateRelease | DrillingNewsRelease) -> pd.DataFrame:
        json_data = structured_data.model_dump_json()
        json_data = json.loads(json_data)
        # add ticker field to json_data
        json_data["ticker"] = self._ticker
        df = pd.json_normalize(json_data)
        return df

    def run(self, file_path: str, news_type: NewsType = None, output_format: str = None, include_financial_data = True, ticker = None) -> dict | pd.DataFrame:
        """
        Run the pipeline to extract structured data from a PDF URL.

        Args:
            pdf_url (str): URL of the PDF file
        """
        if ticker is not None:
            self._ticker = ticker
        pdf_loader = PDFLoader(file_path, read_only=True)
        article_title = pdf_loader.get_title()
        if news_type is None:
            try:
                news_type = self.news_type_classifier.inference(article_title)
                self.logger.info(f"Classified news type: {news_type}")
            except Exception as e:
                self.logger.error(f"Error classifying news type: {e}")
        
        self.logger.info(f"Extracting text from PDF: {article_title}")
        match news_type:
            case NewsType.PEA | NewsType.PFS | NewsType.FS:
                text_content = pdf_loader.load_pfs()
            case _:
                text_content = pdf_loader.load_corporate_announcement()
        if not text_content:
            self.logger.error(f"Failed to extract text from PDF: {file_path}")
            return None
        
        news_text = self._clean_pdf_text(text_content)
        self.logger.info(f"Extracted text: \n {news_text}")
        structured_data = self.extractor.run(news_text, news_type=news_type)
        if output_format:
            match output_format:
                case "df":
                    df = self._structured_output_to_df(structured_data)
                    if include_financial_data and ticker:
                        df = self.add_financial_data_to_df(df, ticker)
                        self.logger.info(f"Added financial data to DataFrame for ticker: {ticker}")
                    return df
                case "json":
                    json_data = structured_data.model_dump_json()
                    return json.loads(json_data)
                case _:
                    pass
        
        return structured_data
    
    def add_financial_data_to_df(self, df: pd.DataFrame, ticker) -> pd.DataFrame:
        """
        Add financial data to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing structured data
            ticker (str): Ticker symbol of the company

        Returns:
            pd.DataFrame: DataFrame with financial data added
        """
        ticker_data = self.financial_data_retriever.run(ticker)
        self.logger.info(f"Financial data retrieved for ticker: {ticker}")
        if ticker_data is not None:
            # add alongside the existing data, discarding the index
            # use `ticker` column to join both dataframes
            ticker_data = ticker_data.reset_index()
            merged_df = pd.merge(df, ticker_data, on="ticker", how="left")
            return merged_df
        else:
            self.logger.warning(f"No financial data found for ticker: {ticker}")

    def run_all(self, text_contents: list) -> list:
        """
        Run the pipeline to extract structured data from a list of PDF URLs.

        Args:
            pdf_urls (list): List of PDF URLs
        """
        structured_data = []
        for text_content in text_contents:
            data = self.run(text_content)
            structured_data.append(data)
        return structured_data