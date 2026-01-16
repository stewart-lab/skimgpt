import os
import logging
import google.genai as genai
from google.genai import types

class FullTextChunker:
    """Agent for chunking enhanced full text into hypothesis-relevant evidence."""
    
    def __init__(self, secrets=None, model_name="gemini-3-flash-preview", logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if secrets and "GEMINI_API_KEY" in secrets:
            api_key = secrets["GEMINI_API_KEY"]
            
        if not api_key:
            self.logger.warning("GEMINI_API_KEY not found. Chunker will fail.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
                self.logger.info(f"FullTextChunker initialized with model {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini Client: {e}")
                self.client = None

    def chunk_document(self, text: str, hypothesis: str) -> str:
        """
        Extracts evidence from the full text that supports or refutes the hypothesis.
        
        Args:
            text: The "Universal Document" (Enhanced Full Text).
            hypothesis: The scientific hypothesis to evaluate.
            
        Returns:
            A string containing structured evidence chunks.
        """
        if not self.client:
            return "Error: Gemini client not initialized."
            
        prompt = f"""You are an expert scientific analyst.
Your task is to extract "Evidence Chunks" from the provided scientific article that are relevant to the following Hypothesis.

HYPOTHESIS: "{hypothesis}"

INSTRUCTIONS:
1. Scan the full text, including figure transcriptions and tables.
2. Identify specific experimental results, quantitative data (p-values, fold changes), and methodological details that DIRECTLY support or refute the hypothesis.
3. Ignore general background information, introductions, or experiments unrelated to the hypothesis.
4. Extract the evidence as concise, self-contained chunks.
5. For each chunk, cite the section or figure it comes from (e.g., [Results], [Figure 1]).
6. If the article contains NO relevant info, return "No relevant evidence found."

FORMAT:
Return the output as a Markdown list of evidence blocks:

- **Evidence [Source]**: <Concise description of finding and methodology>
- **Evidence [Figure X]**: <Key data point from figure analysis>

ARTICLE CONTENT:
{text}
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="high")
                )
            )
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"Chunking failed: {e}")
            return f"Error extracting evidence: {e}"
