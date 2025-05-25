import requests
from datetime import datetime
import time
import re
from typing import Optional, List, Dict, Tuple
import pandas as pd
import os
import logging
import json
# from pyhtml2pdf import converter # Keep commented if not actively used for this core task

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define document type constants to be used by the pipeline and for filenames
DOC_TYPE_FS = "FS"
DOC_TYPE_PFS = "PFS"
DOC_TYPE_PEA = "PEA"
DOC_TYPE_SK1300 = "SK1300"
DOC_TYPE_TRS = "TRS" # Technical Report Summary (generic)

class Edgar:
    BASE_URL = "https://www.sec.gov"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"

    # Keywords for different report types
    FS_KEYWORDS = ['feasibility study', 'fs', 'definitive feasibility study', 'detailed feasibility study']
    PFS_KEYWORDS = ['pre-feasibility study', 'prefeasibility study', 'pfs']
    PEA_KEYWORDS = ['preliminary economic assessment', 'pea']
    SK1300_KEYWORDS = ['s-k 1300', 'sk 1300', 'item 601(b)(96)'] # Added item for SK1300 identification
    GENERIC_TECH_REPORT_KEYWORDS = ['technical report', 'technical report summary', 'mineral resource', 'resource estimate', 'reserve estimate']

    # S-K 1300 reports are typically under EX-96.x or similar variants
    SK1300_EXHIBIT_PATTERNS = [
        'ex-96', 'exhibit 96', 'ex.96',
        '96.1', '96.2', '96.3', '96.4', '96.5'
    ]
    # General exhibit patterns that might house PEA/PFS/FS if not EX-96
    OTHER_RELEVANT_EXHIBIT_PATTERNS = ['ex-99', 'exhibit 99', 'ex.99']

    # Order of preference for report types if multiple keywords match
    DOC_TYPE_PREFERENCE_ORDER = [DOC_TYPE_FS, DOC_TYPE_PFS, DOC_TYPE_PEA, DOC_TYPE_SK1300, DOC_TYPE_TRS]


    def __init__(self, user_agent_email: str = "your-email@domain.com"):
        self.headers = {
            'User-Agent': f'MineCast Mining Research Tool ({user_agent_email})',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        if user_agent_email == "your-email@domain.com":
            logger.warning("Using default User-Agent email. Please provide a real email for SEC compliance.")
        logger.info(f"Edgar instance initialized with User-Agent: {self.headers['User-Agent']}")

    def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        logger.info(f"Looking up CIK for ticker: {ticker}")
        try:
            time.sleep(0.1) 
            headers_for_tickers = self.headers.copy()
            response = requests.get(self.COMPANY_TICKERS_URL, headers=headers_for_tickers)
            response.raise_for_status()
            data = response.json()
            ticker_upper = ticker.upper()
            for _, company_data in data.items():
                if isinstance(company_data, dict) and company_data.get('ticker') == ticker_upper:
                    cik = str(company_data['cik_str']).zfill(10)
                    logger.info(f"Found CIK {cik} for ticker {ticker_upper}")
                    return cik
            logger.warning(f"No CIK found for ticker {ticker_upper}")
            return None
        except requests.RequestException as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for CIK lookup: {e}")
            return None

    def _get_document_details(self, description: str, href: str) -> Tuple[Optional[str], str]:
        """
        Determines the document type and a clean title from its description and link.
        Returns (doc_type, cleaned_title)
        """
        desc_lower = description.lower() if description else ""
        href_lower = href.lower() if href else ""
        text_content_for_matching = desc_lower + " " + href_lower # Combine for keyword search

        doc_type = None

        if any(k in text_content_for_matching for k in self.FS_KEYWORDS):
            doc_type = DOC_TYPE_FS
        elif any(k in text_content_for_matching for k in self.PFS_KEYWORDS):
            doc_type = DOC_TYPE_PFS
        elif any(k in text_content_for_matching for k in self.PEA_KEYWORDS):
            doc_type = DOC_TYPE_PEA
        elif any(k in text_content_for_matching for k in self.SK1300_KEYWORDS) or \
             any(patt in text_content_for_matching for patt in self.SK1300_EXHIBIT_PATTERNS):
            doc_type = DOC_TYPE_SK1300
        elif any(k in text_content_for_matching for k in self.GENERIC_TECH_REPORT_KEYWORDS):
            doc_type = DOC_TYPE_TRS
        
        # If it's an exhibit type often used for these reports, but no keywords matched yet,
        # and it's an EX-99, it *might* be a generic technical report if description hints at it.
        # This part is tricky to not over-classify.
        # For now, explicit keywords are primary. The SK1300_EXHIBIT_PATTERNS are quite specific.

        cleaned_title = re.sub(r'<[^>]+>', ' ', description).strip() # Remove HTML tags from description for title
        cleaned_title = re.sub(r'\\s+', ' ', cleaned_title) # Consolidate multiple spaces
        if not cleaned_title and href: # Fallback to href if description is empty
             cleaned_title = os.path.basename(href)
        
        if doc_type:
            logger.debug(f"Determined doc_type: {doc_type} for Desc: '{description}', Href: '{href}'")
        else:
            logger.debug(f"No specific doc_type determined for Desc: '{description}', Href: '{href}'")
            
        return doc_type, cleaned_title


    def _extract_pdf_url_from_exhibit_page(self, exhibit_html_url: str, filing_archive_base_url: str) -> Optional[str]:
        logger.debug(f"Fetching HTML exhibit page to find PDF: {exhibit_html_url}")
        try:
            time.sleep(0.1)
            response = requests.get(exhibit_html_url, headers=self.headers)
            response.raise_for_status()
            html_content = response.text
            
            pdf_link_regex = r"""<a\s+[^>]*?href=(['"])([^'"]*?\.pdf(?:#[\w-]+)?)\1"""
            pdf_link_matches = re.findall(pdf_link_regex, html_content, re.IGNORECASE)
            
            if pdf_link_matches:
                pdf_file_href = pdf_link_matches[0][1].strip()
            else:
                fallback_pdf_regex = r"""href=(['"])([^'"]*?\.pdf(?:#[\w-]+)?)\1"""
                fallback_pdf_matches = re.findall(fallback_pdf_regex, html_content, re.IGNORECASE)
                if fallback_pdf_matches:
                    pdf_file_href = fallback_pdf_matches[0][1].strip()
                else:
                    logger.debug(f"No PDF link found within HTML exhibit: {exhibit_html_url}") # Changed from warning to debug
                    return None
            
            final_pdf_url = pdf_file_href if pdf_file_href.startswith('http') else f"{filing_archive_base_url}/{pdf_file_href.lstrip('/')}"
            logger.info(f"PDF link extracted from HTML exhibit: {final_pdf_url}")
            return final_pdf_url
        except requests.RequestException as e:
            logger.error(f"Failed to fetch or process HTML exhibit page {exhibit_html_url}: {e}")
            return None

    def _get_latest_8k_filings_metadata(self, cik: str, num_filings: int = 100) -> List[Dict]: # Increased default slightly
        logger.info(f"Fetching recent filings for CIK: {cik} (checking up to {num_filings} filings)")
        submissions_api_url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
        api_headers = self.headers.copy()
        api_headers['Host'] = 'data.sec.gov'
        
        try:
            time.sleep(0.1)
            response = requests.get(submissions_api_url, headers=api_headers)
            response.raise_for_status()
            data = response.json()

            recent_filings_data = data.get('filings', {}).get('recent', {})
            if not recent_filings_data or not recent_filings_data.get('accessionNumber'):
                logger.warning(f"No recent filings data found for CIK: {cik}")
                return []

            # Include primaryDocDescription as it's useful
            df = pd.DataFrame({
                'accessionNumber': recent_filings_data.get('accessionNumber', []),
                'filingDate': recent_filings_data.get('filingDate', []),
                'reportDate': recent_filings_data.get('reportDate', []), 
                'form': recent_filings_data.get('form', []),
                'primaryDocument': recent_filings_data.get('primaryDocument', []),
                'primaryDocDescription': recent_filings_data.get('primaryDocDescription', [])
            })
            
            # Focus on 8-K, but also consider 8-K/A (amendments)
            df_8k = df[df['form'].str.contains('8-K', na=False)].copy()
            if df_8k.empty:
                logger.warning(f"No 8-K (or 8-K/A) filings found for CIK: {cik}")
                return []

            df_8k.loc[:, 'filingDate'] = pd.to_datetime(df_8k['filingDate'])
            df_8k = df_8k.sort_values('filingDate', ascending=False)
            
            latest_filings = df_8k.head(num_filings).to_dict('records')
            logger.info(f"Found {len(latest_filings)} latest 8-K type filings (up to {num_filings} requested) for CIK {cik}.")
            return latest_filings
        except requests.RequestException as e:
            logger.error(f"Error fetching submissions for CIK {cik}: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching submissions for CIK {cik}: {e}", exc_info=True)
            return []

    def get_latest_economic_reports(self, ticker: str, output_dir: str = "downloaded_reports", num_filings_to_check: int = 5) -> List[Dict]:
        logger.info(f"Starting search for latest economic reports for ticker: {ticker} (checking up to {num_filings_to_check} 8-K filings).")
        cik = self._get_cik_from_ticker(ticker)
        if not cik:
            return []

        latest_filings_metadata = self._get_latest_8k_filings_metadata(cik, num_filings_to_check)
        if not latest_filings_metadata:
            logger.warning(f"No 8-K filings found to process for ticker {ticker} (CIK {cik}).")
            return []

        downloaded_reports_info = []
        processed_urls = set() # To avoid re-downloading the same document if linked multiple times or from different filings (if processing more than one)

        for filing_metadata in latest_filings_metadata:
            accession_number = filing_metadata['accessionNumber']
            formatted_accession = accession_number.replace('-', '')
            filing_date_str = filing_metadata['filingDate'].strftime('%Y-%m-%d')
            primary_document_name = filing_metadata['primaryDocument']
            primary_doc_desc = filing_metadata.get('primaryDocDescription', '')
            
            logger.info(f"Processing filing from {filing_date_str} (Accession: {accession_number}, Primary Doc: {primary_document_name})")
            filing_archive_base_url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{formatted_accession}"

            # Option 1: Check if the primary document itself is the report
            # (e.g. an 8-K that IS the Feasibility Study announcement)
            # This is less common for full S-K 1300s but can happen for PEA/FS news releases
            if primary_document_name.lower().endswith(('.htm', '.html', '.pdf')):
                # Use primary_doc_desc for classification if available, else the filename
                doc_type, report_title = self._get_document_details(primary_doc_desc or primary_document_name, primary_document_name)
                if doc_type:
                    doc_url = f"{filing_archive_base_url}/{primary_document_name}"
                    if doc_url not in processed_urls:
                        file_ext = ".pdf" if primary_document_name.lower().endswith(".pdf") else ".htm"
                        # Add to list to be downloaded later, or download directly
                        # For now, let's assume exhibits are more common for full reports
                        logger.info(f"Primary document '{primary_document_name}' identified as potential report type: {doc_type}. URL: {doc_url}")
                        # This path could be further developed to download it if it's the sole source.
                        # For now, prioritizing exhibits for full reports.

            # Option 2: Process exhibits linked FROM the primary document
            main_filing_page_url = f"{filing_archive_base_url}/{primary_document_name}"
            if not primary_document_name.lower().endswith((".htm", ".html")):
                logger.debug(f"Primary document {primary_document_name} is not HTML, skipping exhibit parsing for it.")
            else:
                try:
                    logger.debug(f"Fetching main filing page for exhibits: {main_filing_page_url}")
                    time.sleep(0.1)
                    response = requests.get(main_filing_page_url, headers=self.headers)
                    response.raise_for_status()
                    filing_page_content = response.text

                    exhibit_matches = re.findall(
                        r"""<a\s+[^>]*href=(['"])([^'"]*?\.(?:pdf|htm[l]?)(?:#[\w-]+)?)\1[^>]*>([^<]+(?:<[^>]+>)*[^<]*)</a>""",
                        filing_page_content, re.IGNORECASE | re.DOTALL 
                    )
                    logger.debug(f"Found {len(exhibit_matches)} potential exhibit links in {primary_document_name}")

                    for quote_char, href, description_html in exhibit_matches:
                        href_stripped = href.strip()
                        
                        doc_type, report_title = self._get_document_details(description_html, href_stripped)

                        if doc_type:
                            logger.info(f"Relevant exhibit identified: Type: {doc_type}, Desc: '{report_title}', Href: '{href_stripped}'")
                            
                            document_url_to_download = href_stripped
                            if not href_stripped.startswith('http'):
                                document_url_to_download = f"{filing_archive_base_url}/{href_stripped.lstrip('/')}"

                            if document_url_to_download in processed_urls:
                                logger.debug(f"Skipping already processed URL: {document_url_to_download}")
                                continue

                            file_extension_to_save = ".htm" # Default

                            if href_stripped.lower().endswith('.pdf'):
                                file_extension_to_save = ".pdf"
                                logger.info(f"Direct PDF exhibit identified: {document_url_to_download}")
                            elif href_stripped.lower().endswith(('.htm', '.html')):
                                file_extension_to_save = ".htm"
                                logger.info(f"HTML exhibit identified: {document_url_to_download}. Checking for internal PDF.")
                                potential_pdf_url = self._extract_pdf_url_from_exhibit_page(document_url_to_download, filing_archive_base_url)
                                if potential_pdf_url:
                                    logger.info(f"Found PDF link '{potential_pdf_url}' inside HTML exhibit. Prioritizing PDF.")
                                    document_url_to_download = potential_pdf_url
                                    file_extension_to_save = ".pdf"
                            else: # Should not be hit due to regex, but good to be safe
                                logger.warning(f"Exhibit '{href_stripped}' has an unexpected extension. Defaulting to .htm.")
                                file_extension_to_save = ".htm"


                            if document_url_to_download:
                                logger.info(f"Attempting to download: {document_url_to_download}")
                                os.makedirs(output_dir, exist_ok=True)
                                
                                clean_title_for_file = re.sub(r'[^a-zA-Z0-9_.-]', '_', report_title)[:50]
                                document_filename = f"{filing_date_str}_{ticker}_{doc_type}_{clean_title_for_file}_{accession_number}{file_extension_to_save}"
                                document_filepath = os.path.join(output_dir, document_filename)

                                try:
                                    time.sleep(0.1)
                                    doc_response = requests.get(document_url_to_download, headers=self.headers, stream=True, timeout=30)
                                    doc_response.raise_for_status()

                                    with open(document_filepath, 'wb') as f:
                                        for chunk in doc_response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                    
                                    logger.info(f"Successfully downloaded: {document_filepath}")
                                    downloaded_reports_info.append({
                                        "filepath": document_filepath,
                                        "ticker": ticker,
                                        "doc_type": doc_type,
                                        "filing_date": filing_date_str,
                                        "accession_number": accession_number,
                                        "report_title": report_title, # Use the cleaned title from _get_document_details
                                        "source_url": document_url_to_download
                                    })
                                    processed_urls.add(document_url_to_download)
                                    # If only one report is desired per run, could return here.
                                    # For now, collecting all within the num_filings_to_check scope.

                                except requests.RequestException as e_dl:
                                    logger.error(f"Failed to download document {document_url_to_download}: {e_dl}")
                                except Exception as e_general_dl:
                                    logger.error(f"Unexpected error downloading {document_url_to_download}: {e_general_dl}")
                except requests.RequestException as e_fetch:
                    logger.error(f"Error fetching or parsing main filing page {main_filing_page_url}: {e_fetch}")
                except Exception as e_general_main:
                    logger.error(f"Unexpected error processing {main_filing_page_url}: {e_general_main}")
        
        if not downloaded_reports_info:
            logger.warning(f"No relevant economic reports were downloaded for ticker {ticker} after checking {num_filings_to_check} filings.")
        else:
             # Sort by filing date (desc) then by preferred type (more specific like FS first)
            # Primary sort: filing_date descending (newest first)
            # Secondary sort: doc_type by preference_order ascending (FS before PEA before SK1300 etc.)
            downloaded_reports_info.sort(key=lambda r: (
                datetime.strptime(r['filing_date'], '%Y-%m-%d'), # Date object for correct chronological sorting
                self.DOC_TYPE_PREFERENCE_ORDER.index(r['doc_type']) if r['doc_type'] in self.DOC_TYPE_PREFERENCE_ORDER else float('inf') # Lower index = higher preference
            ), reverse=False) # Sort ascending by preference, then reverse for date
            # Since we want date descending (newest first) and preference ascending (FS first), we sort by preference first, then by date descending (stable sort)
            
            # Step 1: Sort by preference (ascending - FS, PFS, PEA, SK1300, TRS)
            downloaded_reports_info.sort(key=lambda r: self.DOC_TYPE_PREFERENCE_ORDER.index(r['doc_type']) if r['doc_type'] in self.DOC_TYPE_PREFERENCE_ORDER else float('inf'))
            # Step 2: Sort by date (descending - newest first), this is a stable sort and will maintain the preference order for same dates
            downloaded_reports_info.sort(key=lambda r: datetime.strptime(r['filing_date'], '%Y-%m-%d'), reverse=True)

        return downloaded_reports_info

if __name__ == "__main__":
    if not logger.handlers or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
        
    edgar_client = Edgar(user_agent_email="your.email@example.com") # PLEASE REPLACE
    
    test_ticker = "USAU" 
    # test_ticker = "CTGO"
    # test_ticker = "GOLD"

    logger.info(f"--- Starting test for ticker: {test_ticker} ---")
    
    # The method name is now get_latest_economic_reports
    reports = edgar_client.get_latest_economic_reports(
        ticker=test_ticker,
        output_dir="downloaded_reports_test", # Use a specific test output dir
        num_filings_to_check=100 
    )

    if reports:
        logger.info(f"--- Test for {test_ticker} successful. {len(reports)} report(s) found: ---")
        for report_info in reports:
            logger.info(f"  Doc Type: {report_info['doc_type']}, Title: {report_info['report_title']}, Path: {report_info['filepath']}")
    else:
        logger.error(f"--- Test for {test_ticker} failed. No reports downloaded. ---")