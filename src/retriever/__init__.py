from ..etl.extractor.pdf_loader import PDFLoader
from .sedarplus import SedarPlusLoader

import logging


class Retriever:
    """
    A class for retrieving and processing PDF documents from URLs.
    """

    def __init__(self, logger=None):
        """
        Initialize the Retriever class.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.pdf_loader = PDFLoader(logger=self.logger)
        self.sedarplus_loader = SedarPlusLoader(self.logger)

    def load_url(self, url):
        """
        Load a PDF from a URL and extract its text content.

        Args:
            url (str): URL of the PDF file

        Returns:
            str: Extracted text from the PDF
        """
        return self.pdf_loader.load(url)
    
    def run(self):
        output = []
        urls = self.sedarplus_loader.load()
        for url in urls:
            text_content = self.load_url(url)
            output.append(text_content)
        return output
