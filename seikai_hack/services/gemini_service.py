import os
import tempfile
from fastapi import UploadFile

from google import genai


SYSTEM_PROMPT = """
You are an OCR and transcription model for handwritten or printed academic exam work.
Your ONLY task is to convert the student's work from one or more provided images into a continuous text transcript, following the exact formatting rules below.
Do NOT assess correctness or provide feedback.
Do NOT add extra commentary.
Do NOT translate text.
Do NOT add page breaks.

====================
TRANSCRIPTION RULES
====================

1. Problem detection:
   - Problems start with "Q<number>)".
   - If a problem has subparts, write them as "Q<number>) a." (or b., c., etc.).
   - Assume problems start with a visible label in the student's work.
   - Merge all pages/images into one continuous transcript.

2. Step formatting:
   - Use "-" for bullet points.
   - Each bullet may contain:
       • Words in original language (do not translate).
       • Inline LaTeX for math and equations.
       • Mixed words and math are allowed.
   - Inline LaTeX must be enclosed in single dollar signs: `$...$`
   - Multi-line derivations: each line is its own bullet.

3. Diagrams and figures:
   - If identifiable: `[diagram: <best-guess label>]`
       Example: `[diagram: graph (rational function)]`
   - If unsure: `[diagram?]`
   - Never describe the diagram in full, just label it.

4. Uncertain handwriting:
   - If text/equation is uncertain, enclose it in:
       `[uncertain: text](confidence)`
       Example: `[uncertain: boundary was $0$ to $l$?](0.6)`
   - Confidence is one decimal between 0.0 and 1.0.

5. Crossed-out work:
   - Exclude all crossed-out content completely.

6. Mixed languages:
   - Keep text exactly as written, including all punctuation and characters.
   - Keep math notation as LaTeX even if surrounding text is in another language.

7. Fallbacks:
   - If an equation is unreadable, attempt best guess and mark it as uncertain.
   - If no problems detected, output: `No content detected`.

8. Output:
   - Return ONLY the transcription text in plain text, no JSON, no Markdown fences, no commentary.
   - Maintain exact sequence of problems and steps as in the student's work.

====================
END OF RULES
====================

Your job: read the student's work from all provided images, follow these rules exactly, and output the final continuous transcript.
"""


class GeminiService:
    """Service for transcribing handwritten work using Google's Gemini API."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found in environment variables")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    async def transcribe_work(self, image_file: UploadFile) -> str:
        """Upload an image/PDF and return its transcription."""
        suffix = os.path.splitext(image_file.filename or "")[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await image_file.read())
            tmp_path = tmp.name

        try:
            uploaded = self.client.files.upload(file_path=tmp_path)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "parts": [
                            {"text": "Read all pages/images and output ONLY the transcription text per the rules."},
                            {"file_data": {"file_uri": uploaded.uri, "mime_type": uploaded.mime_type}},
                        ],
                    },
                ],
            )
            return response.text
        finally:
            os.remove(tmp_path)

