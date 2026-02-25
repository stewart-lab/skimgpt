import os
import logging
from google import genai
from google.genai import types
import base64
import time
import concurrent.futures
from pathlib import Path
from src.prompt_library import get_transcription_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Handles image analysis and description generation using Google Gemini 3 Flash."""

    def __init__(
        self,
        secrets=None,
        max_workers=5,
        logger=None,
        model_name="gemini-3-flash-preview",
    ):
        """Initialize with optional credentials and concurrency settings."""
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug("Initializing ImageAnalyzer with Gemini")
        api_key = os.environ.get("GEMINI_API_KEY")
        if secrets and "GEMINI_API_KEY" in secrets:
            api_key = secrets["GEMINI_API_KEY"]

        if not api_key:
            self.logger.warning("GEMINI_API_KEY not found in environment or secrets.")

        try:
            self.client = genai.Client(
                api_key=api_key, http_options={"api_version": "v1alpha"}
            )
            self.logger.info("Gemini Client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini Client: {e}")
            self.client = None

        self.max_workers = max_workers
        self.model_name = model_name
        self.logger.info(
            f"ImageAnalyzer initialized with {max_workers} workers using model {self.model_name}"
        )

    def _encode_image(self, image_path):
        """Read and encode local image file."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {str(e)}")
            return ""

    def _process_single_image(self, figure, full_text, hypothesis):
        """Process a single image with error handling and retries using Gemini."""
        if not self.client:
            self.logger.error("Gemini client not initialized. Skipping.")
            return figure

        img_path = figure.get("local_path")
        if not img_path or not os.path.exists(img_path):
            self.logger.warning(f"Image path missing or invalid: {img_path}")
            figure["enhanced_content"] = figure.get("caption", "")
            return figure

        self.logger.info(f"Processing figure: {img_path}")
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Prepare image data
                with open(img_path, "rb") as f:
                    image_bytes = f.read()

                # Determine MIME type
                ext = Path(img_path).suffix.lower()
                mime_type = "image/png" if ext == ".png" else "image/jpeg"

                start_time = time.time()

                # --- Step 1: Transcription ---
                # TODO: Get hypothesis from user-inputted hypothesis
                # hypothesis = "Does RBM20 antisense oligo treatment improve Hfpef mouse disease severity?"
                # hypothesis = ""
                self.logger.info(f"Hypothesis: {hypothesis}")
                transcription_prompt = get_transcription_prompt(hypothesis)
                self.logger.info(f"Transcription prompt: {transcription_prompt}")
                caption = figure.get("caption", "")

                transcription_contents = [
                    types.Content(
                        parts=[
                            types.Part(text=transcription_prompt),
                            types.Part(text=f"Caption: {caption}"),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=mime_type,
                                    data=base64.b64encode(image_bytes).decode("utf-8"),
                                ),
                                # Using high resolution for better OCR/details
                                media_resolution={"level": "media_resolution_high"},
                            ),
                        ]
                    )
                ]

                self.logger.info(
                    f"Sending transcription request to {self.model_name}..."
                )
                transcription_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=transcription_contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="high"
                        )  # Leverage reasoning
                    ),
                )

                transcription = transcription_response.text.strip()
                self.logger.debug(f"Transcription received: {transcription[:100]}...")

                processing_time = time.time() - start_time
                self.logger.info(
                    f"Total processing time for {img_path}: {processing_time:.2f} seconds"
                )

                # Store results
                figure["transcription"] = transcription
                figure["enhanced_content"] = (
                    transcription  # Transcription is the enhanced content now
                )

                self.logger.info(f"Successfully processed figure: {img_path}")
                return figure

            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt+1}/{max_retries} failed: {str(e)}"
                )
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2**attempt)
                    time.sleep(sleep_time)
                else:
                    self.logger.error(
                        f"Failed to process figure after {max_retries} attempts: {e}"
                    )
                    figure["enhanced_content"] = figure.get("caption", "")
                    return figure

    def enhance_figure_descriptions(self, figures, full_text, hypothesis):
        """Generate enhanced descriptions for figures using parallel processing."""
        self.logger.info(
            f"Starting to process {len(figures)} figures in parallel with Gemini"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_figure = {
                executor.submit(
                    self._process_single_image, figure, full_text, hypothesis
                ): figure
                for figure in figures
            }

            enhanced_figures = []
            for future in concurrent.futures.as_completed(future_to_figure):
                figure = future_to_figure[future]
                try:
                    result = future.result()
                    enhanced_figures.append(result)
                except Exception as e:
                    self.logger.error(
                        f"Error processing figure {figure.get('local_path')}: {str(e)}"
                    )
                    figure["enhanced_content"] = figure.get("caption", "")
                    enhanced_figures.append(figure)

        self.logger.info(f"Completed processing all {len(figures)} figures")
        return enhanced_figures
