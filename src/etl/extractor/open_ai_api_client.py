from openai import OpenAI
from typing import Dict, Any, Optional

import logging
import json
import tiktoken
import nltk

from .schemas import (
    DrillingNewsRelease,
    ResourceUpdateRelease,
    CorporateUpdateRelease,
    ProjectUpdateRelease,
    Corp10K,
    PFSRelease,
    TechnicalReportData
)
from .types import NewsType

SYSTEM_PROMPT = """
INSTRUCTION:
You are an information extraction engine. 
You will be provided with a mining company news release, which may include information about drilling results,
resource updates, project updates, or corporate updates as well as technical reports such as feasibility studies.
Your task is to extract the relevant information from the news release and return it in the specified JSON format.

TIPS:
Some useful acronyms to know:
- AISC: all-in sustaining costs (typically between $800 and $1,800 per ounce)
- NPV: net present value (in USD, after tax)
- IRR: internal rate of return (IRR in %)
- LOM: life of mine (in years)
More useful knowledge:
- in some reports, companies may assume a long term price of gold, silver, or other metals.
- They might also provide different scenarios, such as base case, upside case, and downside case.
- Gold prices are always specified in USD per ounce.
- The same applies to silver and other metals.
- AISC occurs in the context of the NPV calculation. 
- Average annual production is typically specified in ounces per year.

TASK:
You will be provided with a JSON schema that describes the information to extract.
Return the extracted information in the specified JSON format.
"""


class OpenAIClient:
    """
    A class for extracting structured data from mining company news releases
    using OpenAI's API with JSON schema validation.
    """
    MAX_INPUT_LENGTH = 128000
    
    def __init__(
            self,
            api_key: str = None,
            logger = None,
            vebosity: int = 0
            ):
        """
        Initialize the extractor with OpenAI API credentials.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4-turbo)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "o3-mini-2025-01-31"
        self.logger = logger or logging.getLogger(__name__)
        self._verbosity = vebosity or 0
    
    def _determine_news_type(self, news_text: str) -> NewsType:
        """
        Determine the type of mining news release.
        
        Args:
            news_text: The full text of the news release
            
        Returns:
            NewsType enum value representing the type of news release
        """
        type_schema = {
            "type": "object",
            "properties": {
                "news_type": {
                    "type": "string",
                    "enum": [t.value for t in NewsType]
                }
            },
            "required": ["news_type"],
            "additionalProperties": False
        }
        schema = {
                "format": {
                    "type": "json_schema",
                    "name": "news_type",
                    "schema": type_schema,
                    "strict": True
                }
            }
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "Determine the type of this mining company news release."},
                {"role": "user", "content": f"News release text:\n{news_text}"}
            ],
            text=schema,
        )
        result = json.loads(response.output_text)
        if self._verbosity > 0:
            self.logger.info("OpenAI response: %s", result)
        return NewsType(result["news_type"])
    
    def _verify_query_length(self, news_text: str) -> bool:
        """
        Verify if the length of the news text exceeds the maximum allowed length.
        
        Args:
            news_text: The full text of the news release
            
        Returns:
            bool: True if within limits, False otherwise
        """
        enc = tiktoken.encoding_for_model("gpt-4o")
        tokens = enc.encode(news_text)
        if len(tokens) > self.MAX_INPUT_LENGTH:
            self.logger.warning("Query length exceeds the maximum allowed length.")
            return False
        return True

    def _condense_text(self, news_text: str) -> str:
        """
        Condense the news text to fit within the token limit.
        
        Args:
            news_text: The full text of the news release
            
        Returns:
            Condensed text
        """
        if self._verbosity > 0:
            self.logger.info("Condensing text to fit within token limit.")
        
        # Condense the text to fit within the token limit
        news_text = news_text.replace("\n", " ")
        news_text_tokens = nltk.word_tokenize(news_text)
        condensed_text = []
        # skip stop words
        try:
            stop_words = set(nltk.corpus.stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            stop_words = set(nltk.corpus.stopwords.words("english"))
        for word in news_text_tokens:
            if word.lower() not in stop_words:
                condensed_text.append(word)
        condensed_text = " ".join(condensed_text)
        condensed_text = condensed_text[:self.MAX_INPUT_LENGTH]
        if self._verbosity > 0:
            self.logger.info("Condensed text to length: %d", len(condensed_text))
        return condensed_text
    
    def _extract_structured_data(self, news_text: str, news_type: NewsType) -> Dict[str, Any]:
        """
        Extract structured data from a mining news release based on its type.
        
        Args:
            news_text: The full text of the news release
            news_type: Optional pre-determined news type. If None, type will be determined automatically.
            
        Returns:
            Dictionary containing structured data extracted from the news release
        """

        schema = self._get_schema_for_type(news_type)
        if not self._verify_query_length(news_text):
            news_text = self._condense_text(news_text)
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"News release text:\n{news_text}"}
            ],
            response_format=schema,
        )
        return response.choices[0].message.parsed
    
    def _get_schema_for_type(self, news_type: NewsType) -> Dict[str, Any]:
        """
        Get the appropriate JSON schema for the news release type.
        
        Args:
            news_type: The type of news release
            
        Returns:
            JSON schema dictionary for the specified news type
        """
        schema_map = {
            NewsType.DRILLING: DrillingNewsRelease,
            NewsType.RESOURCE: ResourceUpdateRelease,
            NewsType.PROJECT: ProjectUpdateRelease,
            NewsType.CORPORATE: CorporateUpdateRelease,
            NewsType.R_10K: Corp10K,
            NewsType.FINANCIAL: PFSRelease, # Keep as is for now, or decide if TechnicalReportData is better
            NewsType.PFS: TechnicalReportData,    # Updated
            NewsType.PEA: TechnicalReportData,    # Updated
            NewsType.FS: TechnicalReportData,     # Updated
            NewsType.NI: TechnicalReportData,     # Updated
            NewsType.SK1300: TechnicalReportData, # Newly added
            NewsType.TRS: TechnicalReportData     # Added for Technical Report Summary
        }
        schema_model = schema_map.get(news_type)
        if schema_model is None:
            self.logger.error(f"No schema defined for NewsType: {news_type}. Defaulting to a generic approach or failing.")
            # Fallback or error handling: For now, let's raise an error or return a very basic schema.
            # This part depends on desired behavior for unmapped types.
            # Forcing a failure if no schema is found ensures we explicitly map types.
            raise ValueError(f"Schema not found for news type: {news_type}")

        self.logger.info(f"Mapped {news_type} to schema: {schema_model.__name__}")
        # The OpenAI API expects the Pydantic model itself for the `response_format` with `parse`
        return schema_model

    def run(self, news_text: str, news_type: NewsType) -> Dict[str, Any]:
        """
        Run the extraction process on the provided news text.
        
        Args:
            news_text: The full text of the news release
            
        Returns:
            Dictionary containing structured data extracted from the news release
        """
        output = self._extract_structured_data(news_text, news_type)
        return output