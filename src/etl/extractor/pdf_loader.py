from io import BytesIO
from langchain.docstore.document import Document
from langchain_community.document_loaders import PDFPlumberLoader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import base64
import logging
import string
import json
import time
import tiktoken
import logging
import os
import pandas as pd
import tabula


def preprocess_text(text: str) -> str:
    """ Apply basic cleaning to the text. """
    text = text.replace("\n\n", "\n") # remove double newlines
    text = ' '.join(text.split()) # remove extra spaces
    text = text.strip() # remove leading and trailing spaces
    text = ''.join(filter(lambda x: x in string.printable, text)) # remove non-ASCII characters
    text = text.replace("\x0c", "") # remove form feed characters
    text = text.replace("\x0b", "") # remove vertical tab characters
    text = text.replace("\x0a", "") # remove line feed characters
    text = text.replace("\x0d", "") # remove carriage return characters
    text = text.replace("\x1c", "") # remove file separator characters
    return text


def remove_repeating_lines(text: str) -> str:
    lines = text.split("\n")
    counts = {}
    for line in lines:
        line = line.strip()
        if line: counts[line] = counts.get(line, 0) + 1
    common = {line for line, count in counts.items() if count > 3}
    return "\n".join([line for line in lines if line.strip() not in common])


class PDFLoader:
    """
    PDFLoader class for loading and processing PDF files.
    """
    with open("src/etl/extractor/relevant_vocab.json") as f:
        vocab = json.load(f)
    TERMS_10K_R = vocab["10-K"]
    TERMS_PEA = vocab["PEA"]
    TERMS_PFS = vocab["PEA"]
    TERMS_FS = vocab["PEA"]
    TERMS_IGNORE = [
        "table of contents",
        "cautionary note",
        "forward-looking statements",
        "indicate by check mark",
        "report of independent registered public accounting firm",
        "involves a high degree of risk",
        "accept professional responsibility for those sections",
        "do hereby certify that",
        "we have audited the accompanying",
        "does not accept responsibility for any errors",
    ]
    def __init__(self, file_path: str = None, read_only=True, headless=True, logger: logging.Logger = None):
        """
        Initialize PDFLoader with optional headless mode.
        
        Args:
            headless (bool): Whether to run Chrome in headless mode. Default is True.
        """
        self._logger = logger or logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.info("Initializing PDFLoader")

        if not read_only:
            chrome_options = Options()
            if headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--kiosk-printing')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            self._driver = webdriver.Chrome(options=chrome_options)
            self._driver.set_window_size(1920, 1080)
            self._driver.set_page_load_timeout(60)

        self._file_path = file_path
        self._loader = PDFPlumberLoader(file_path) if file_path else None

        os.makedirs("temp", exist_ok=True)
        os.makedirs("temp/data", exist_ok=True)
        # Suppress pdfminer warnings
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        
    def _get_pdf_bytes(self, url):
        """
        Fetch a URL and convert it to PDF bytes using Chrome's PDF capabilities.
        
        Args:
            url (str): The URL to fetch and convert to PDF
            
        Returns:
            BytesIO: PDF content as BytesIO object
        """
        self.driver.get(url)
        # Wait for page to fully load
        time.sleep(30)
        
        print_options = {
            'landscape': False,
            'displayHeaderFooter': False,
            'printBackground': True,
            'preferCSSPageSize': True,
        }
        
        # Execute CDP command to generate PDF
        result = self.driver.execute_cdp_cmd('Page.printToPDF', print_options)
        
        # Convert base64 string to BytesIO object
        pdf_data = base64.b64decode(result['data'])
        return BytesIO(pdf_data)
    
    def _remove_unimportant_pages(self, docs: list[Document]):
        """ Remove empty pages and those that contain banned words."""
        out = []
        for doc in docs:
            for term in self.TERMS_IGNORE:
                if term.lower() in doc.page_content.lower():
                    break
            else:
                # check if the page is empty
                if len(doc.page_content.strip()) < 100:
                    continue
                out.append(doc)
        return out
    
    def _keep_relevant_pages(self, docs: list[Document], terms: list[str]):
        filtered_documents = []
        for document in docs:
            for term in terms:
                if term.lower() in document.page_content.lower():
                    filtered_documents.append(document)
                    break
        return filtered_documents
    
    def _merge_documents(self, docs: list[Document]) -> str:
        """
        Merge a list of Document objects into a single string.
        
        Args:
            docs (list[Document]): List of Document objects
            
        Returns:
            str: Merged text content
        """
        merged_text = ""
        for doc in docs:
            merged_text += doc.page_content + "\n"
        processed = preprocess_text(merged_text)
        return remove_repeating_lines(processed)
    
    def _load_tables(self, file_path: str = None) -> list[Document]:
        """
        Load tables from a PDF file.
        
        Args:
            file_path (str): The path to the PDF file
            
        Returns:
            list[Document]: List of extracted tables
        """
        dfs = tabula.read_pdf(file_path, pages="all", multiple_tables=True, output_format="dataframe", stream=True, pandas_options={"on_bad_lines": "skip"})
        documents = []
        for df in dfs:
            doc = Document(page_content=df.to_string(index=False))
            doc.metadata = {"source": file_path}
            documents.append(doc)
        return documents

    def _load_docs(self, file_path: str = None) -> list[Document]:
        """
        Load documents from a PDF file.
        
        Args:
            file_path (str): The path to the PDF file
            
        Returns:
            list[Document]: List of extracted documents
        """
        if file_path:
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
        else:
            docs = self._loader.load()
        return docs

    def get_title(self) -> str:
        """
        Get the title of the PDF document.
        
        Returns:
            str: Title of the document
        """
        if self._loader:
            title_page = next(self._loader.lazy_load())
            title = title_page.page_content.split("\n\n")[0]
            title = preprocess_text(title)
            return title
        else:
            raise ValueError("File path not provided.")
        
    def load_10k_report_documents(self, file_path = None) -> list[Document]:
        """
        Load a 10-K report from a file path and extract elements.
        
        Args:
            file_path (str): The path to the PDF file
            
        Returns:
            list[Document]: List of extracted elements
        """
        if file_path:
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
        else:
            docs = self._loader.load()
        
        docs = self._remove_unimportant_pages(docs)
        terms = self.TERMS_10K_R
        filtered_documents = self._keep_relevant_pages(docs, terms)
        return filtered_documents
    
    def load_10k_report(self, file_path = None) -> str:
        docs = self.load_10k_report_documents(file_path)
        return self._merge_documents(docs)
    
    def load_technical_report(self, file_path: str = None) -> str:
        pass

    def load_corporate_announcement(self, file_path: str = None) -> str:
        pass

    def load_pea(self, file_path: str = None) -> str:
        pass

    def _load_pfs_documents(self, file_path: str = None, tables_only = True) -> list[Document]:
        """
        Load PFS documents from a file path and extract elements.

        Args:
            file_path (str): The path to the PDF file
            tables_only (bool): Whether to load only tables
        Returns:
            list[Document]: List of extracted elements
        """
        if tables_only:
            docs = self._load_tables(file_path)
        else:
            docs = self._load_docs(file_path)
            docs = self._remove_unimportant_pages(docs)
            terms = self.TERMS_PFS
            docs = self._keep_relevant_pages(docs, terms)
        return docs

    def load_pfs(self, file_path: str = None, tables_only = False) -> str:
        if not file_path:
            file_path = self._file_path
        docs = self._load_pfs_documents(file_path, tables_only)
        return self._merge_documents(docs)

    def load_fs(self, file_path: str = None) -> str:
        pass

    def get_token_count(self, text: str) -> int:
        """
        Get the number of tokens in a string.
        
        Args:
            text (str): The input string
            
        Returns:
            int: Number of tokens
        """
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return len(encoding.encode(text))
    

if __name__ == "__main__":
    # Example usage
    pdf_loader = PDFLoader(file_path="data/MadsenPFS-NI43-101-Final-20250218.pdf", read_only=True)
    title = pdf_loader.get_title()
    print(f"Title: {title}")
    time.sleep(5)
    text = pdf_loader.load_pfs()
    print(f"Text: {text[:1000]}")  # Print first 1000 characters of the text
        
