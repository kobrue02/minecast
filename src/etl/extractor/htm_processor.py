import logging
import re
from bs4 import BeautifulSoup, NavigableString, Tag
import pandas as pd
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class HtmProcessor:
    """
    Processes an HTM file to extract meaningful text content and tables.
    """
    def __init__(self, htm_filepath: str):
        """
        Initializes the HtmProcessor with the path to the HTM file.

        Args:
            htm_filepath (str): Path to the HTM file.
        """
        self.filepath = htm_filepath
        self.soup = None
        try:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                self.soup = BeautifulSoup(f, 'html.parser')
            logger.info(f"Successfully read and parsed HTM file: {self.filepath}")
        except FileNotFoundError:
            logger.error(f"HTM file not found: {self.filepath}")
            raise
        except Exception as e:
            logger.error(f"Error reading or parsing HTM file {self.filepath}: {e}")
            raise

    def _table_to_markdown(self, table_tag: Tag) -> str:
        """
        Converts a BeautifulSoup table Tag into a Markdown formatted string.
        Handles basic tables with <thead>, <tbody>, <tr>, <th>, <td>.
        """
        markdown_table = ""
        headers = []
        header_row_processed = False

        # Try to find headers in <thead> or the first row
        thead = table_tag.find('thead')
        if thead:
            header_tags = thead.find_all(['th', 'td']) # Some tables use <td> in header
            if header_tags:
                headers = [th.get_text(separator=' ', strip=True) for th in header_tags]
                markdown_table += "| " + " | ".join(headers) + " |\n"
                markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                header_row_processed = True
        
        # Process body rows
        tbody = table_tag.find('tbody')
        if not tbody: # If no tbody, assume all rows are part of the body or table is simple
            tbody = table_tag

        for row_tag in tbody.find_all('tr', recursive=False): # recursive=False for direct children of tbody or table
            # If headers weren't found in <thead> and this is the first row, try to use it as header
            if not header_row_processed:
                potential_headers = row_tag.find_all(['th', 'td'])
                if all(h.name == 'th' for h in potential_headers) or not headers: # Good chance this is a header row
                    headers = [th.get_text(separator=' ', strip=True) for th in potential_headers]
                    if headers: # Ensure we actually found some header text
                        markdown_table += "| " + " | ".join(headers) + " |\n"
                        markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                        header_row_processed = True
                        continue # Skip this row as it's processed as a header
            
            columns = row_tag.find_all(['td', 'th'], recursive=False) # Some data rows might use <th> for row headers
            row_data = [col.get_text(separator=' ', strip=True).replace('|', r'\|') for col in columns] # Escape pipe characters
            markdown_table += "| " + " | ".join(row_data) + " |\n"
        
        return markdown_table.strip()

    def extract_text_and_tables(self) -> Tuple[str, List[str]]:
        """
        Extracts the main textual content and converts tables to Markdown.

        Returns:
            Tuple[str, List[str]]: A tuple containing the extracted text (str) 
                                     and a list of Markdown-formatted tables (List[str]).
        """
        if not self.soup:
            return "", []

        extracted_texts = []
        markdown_tables = []

        # Attempt to find a main content area if common tags exist
        main_content = self.soup.find('main') or self.soup.find('article')
        parse_area = main_content if main_content else self.soup.body
        if not parse_area: # Fallback to the whole soup if no body
            parse_area = self.soup

        for element in parse_area.find_all(True, recursive=True): # Iterate over all elements
            if element.name == 'table':
                try:
                    md_table = self._table_to_markdown(element)
                    if md_table: # Only add if table conversion was successful and produced content
                        markdown_tables.append(md_table)
                        # Add a placeholder in text to indicate table position
                        extracted_texts.append(f"\n[TABLE PLACEHOLDER {len(markdown_tables)}]\n")
                    element.decompose() # Remove table from further text processing
                except Exception as e:
                    logger.warning(f"Could not process a table: {e}")
            elif element.name in ['script', 'style', 'nav', 'footer', 'header', 'aside']: # Tags to ignore for text extraction
                element.decompose()
            elif element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div', 'td', 'th']:
                # Extract text from common text-containing tags
                text = element.get_text(separator=' ', strip=True)
                if text:
                    extracted_texts.append(text)
            # Other tags are implicitly ignored for text extraction unless they contain text captured by parents

        # Consolidate text. Join with newlines to preserve some structure.
        full_text = ' \n\n '.join(filter(None, extracted_texts))
        
        # Basic cleaning: remove excessive newlines and leading/trailing whitespace
        full_text = re.sub(r'\n\s*\n', '\n\n', full_text).strip()
        
        logger.info(f"Extracted text (length: {len(full_text)}) and {len(markdown_tables)} tables.")
        return full_text, markdown_tables

    def get_processed_content_for_llm(self) -> str:
        """
        Extracts text and tables, then combines them into a single string 
        suitable for an LLM prompt, with tables in Markdown format.
        """
        text, tables = self.extract_text_and_tables()
        
        # Integrate tables into the text where placeholders are, or append at the end
        # For simplicity, appending all tables at the end with clear delimiters.
        # A more sophisticated approach might interleave them based on placeholders.
        
        llm_input_text = text
        if tables:
            llm_input_text += "\n\n--- TABLES ---\n\n"
            for i, table_md in enumerate(tables):
                llm_input_text += f"Table {i+1}:\n{table_md}\n\n"
        
        # Further cleaning for LLM, ensuring it's not overly long or full of redundant spaces
        llm_input_text = re.sub(r'\s{3,}', '  ', llm_input_text) # Reduce 3+ spaces to 2
        llm_input_text = re.sub(r'(\n\s*){3,}', '\n\n', llm_input_text) # Reduce 3+ newlines (with optional space) to 2
        return llm_input_text.strip()

if __name__ == '__main__':
    # Example Usage (assuming you have an HTM file to test with)
    # Replace 'path/to/your/test_file.htm' with an actual file path
    # This part is for testing the HtmProcessor independently.
    logging.basicConfig(level=logging.DEBUG)
    try:
        # Create a dummy htm file for testing if one doesn't exist
        dummy_htm_path = "_test_dummy_report.htm"
        with open(dummy_htm_path, "w", encoding="utf-8") as f_dummy:
            f_dummy.write("""
            <html><head><title>Test Report</title></head>
            <body>
                <h1>Sample Technical Report</h1>
                <p>This is an S-K 1300 technical report summary for the <b>Gold Mine Project</b>.</p>
                <p>The project shows promising results with an estimated NPV of $100M.</p>
                <h2>Resource Summary</h2>
                <table>
                    <thead><tr><th>Category</th><th>Tonnes (Mt)</th><th>Grade (g/t Au)</th><th>Contained Au (koz)</th></tr></thead>
                    <tbody>
                        <tr><td>Measured</td><td>10</td><td>1.5</td><td>482</td></tr>
                        <tr><td>Indicated</td><td>20</td><td>1.2</td><td>772</td></tr>
                        <tr><td>Inferred</td><td>30</td><td>1.0</td><td>965</td></tr>
                    </tbody>
                </table>
                <p>Further details are in the main sections.</p>
                <h2>Feasibility Study Key Metrics</h2>
                <p>A feasibility study was also completed with the following highlights:</p>
                <ul><li>IRR: 25%</li><li>CAPEX: $50M</li></ul>
                <h3>Another Table</h3>
                <table><tr><td>Data A1</td><td>Data B1</td></tr><tr><td>Data A2</td><td>Data B2</td></tr></table>
            </body></html>
            """)
        
        logger.info(f"Testing HtmProcessor with: {dummy_htm_path}")
        processor = HtmProcessor(dummy_htm_path)
        
        # Test text and table extraction separately
        # text, tables_md = processor.extract_text_and_tables()
        # logger.debug("--- Extracted Text ---")
        # logger.debug(text)
        # logger.debug("--- Extracted Tables (Markdown) ---")
        # for i, table in enumerate(tables_md):
        #     logger.debug(f"Table {i+1}:\n{table}")

        # Test combined content for LLM
        llm_content = processor.get_processed_content_for_llm()
        logger.debug("--- Content for LLM ---")
        logger.debug(llm_content)
        
        # Clean up dummy file
        # os.remove(dummy_htm_path)

    except Exception as e:
        logger.error(f"Error during HtmProcessor test: {e}", exc_info=True) 