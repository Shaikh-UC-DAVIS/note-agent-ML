import json
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from ml.config import config

# ── Structured output models (unchanged) ──────────────────────────────────────

class ExtractedObject(BaseModel):
    id: str = Field(description="Unique identifier for the object")
    type: Literal['Idea', 'Claim', 'Assumption', 'Question', 'Task', 'Evidence', 'Definition']
    canonical_text: str = Field(description="The concise, canonical text representation of the object")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class Link(BaseModel):
    source_id: str
    target_id: str
    type: Literal['Supports', 'Contradicts', 'Refines', 'DependsOn', 'SameAs', 'Causes']
    confidence: float

class ExtractionResult(BaseModel):
    objects: List[ExtractedObject]
    links: List[Link]

# ── LLM Extractor ─────────────────────────────────────────────────────────────

# Using Zero-Shot Extraction to reply on model's high-level reasoning
SYSTEM_PROMPT = """You are a knowledge-extraction engine. Given a passage of text, you must identify every discrete knowledge object and every relationship between them.

### Object types
- **Idea**: A concept, hypothesis, or creative thought.
- **Claim**: A factual assertion that can be true or false.
- **Assumption**: An unstated premise taken for granted.
- **Question**: An open question or inquiry.
- **Task**: An action item or to-do.
- **Evidence**: Data, observations, or citations supporting a claim.
- **Definition**: A formal definition of a term or concept.

### Link types
- **Supports**: Source provides evidence or reasoning for target.
- **Contradicts**: Source conflicts with or opposes target.
- **Refines**: Source is a more specific version of target.
- **DependsOn**: Source requires target to hold true.
- **SameAs**: Source and target express the same idea.
- **Causes**: Source causally leads to target.

### Rules
1. Every object needs a unique, short, kebab-case `id` (e.g. `claim-earth-round`).
2. `confidence` is your estimate (0.0-1.0) of how clearly the text states this object or link.
3. Extract ALL objects you can find; do not omit minor ones.
4. Create links wherever a relationship exists between two extracted objects.

### Output format
Return **only** valid JSON matching this schema (no markdown, no commentary):
{
  "objects": [
    {"id": "...", "type": "...", "canonical_text": "...", "confidence": 0.0}
  ],
  "links": [
    {"source_id": "...", "target_id": "...", "type": "...", "confidence": 0.0}
  ]
}"""


class LLMExtractor:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLMExtractor.

        Args:
            api_key: Optional Groq API key override. Defaults to config['groq_api_key'].
            model: Optional model override. Defaults to config['model'].
        """
        self.model = model or config['model']
        resolved_key = api_key or config['groq_api_key']
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass api_key= or set the GROQ_API_KEY env var."
            )
        self.client = OpenAI(
            api_key=resolved_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def extract(self, text: str) -> ExtractionResult:
        """
        Extracts structured objects and links from the given text using an LLM.

        Args:
            text: The unstructured text to process.

        Returns:
            An ExtractionResult containing the extracted objects and links.
        """
        user_prompt = self._construct_prompt(text)

        try:
            # Here is where we send a request to Groq servers
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                # Use ChatML format for security
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                timeout=config['timeout'],
            )

            raw_json = response.choices[0].message.content # Return object with the metadata
            parsed = json.loads(raw_json)
            result = ExtractionResult(**parsed) # Pour dictionary into model and get extraction

            print(f"[Extraction] ✓ Extracted {len(result.objects)} objects, {len(result.links)} links  (model={self.model})")
            return result

        except json.JSONDecodeError as e:
            print(f"[Extraction] ✗ Failed to parse LLM JSON response: {e}")
            return ExtractionResult(objects=[], links=[])
        except Exception as e:
            print(f"[Extraction] ✗ LLM call failed: {e}")
            return ExtractionResult(objects=[], links=[])

    def _construct_prompt(self, text: str) -> str:
        """Constructs the user prompt for the LLM."""
        return f"""Analyze the following text and extract all knowledge objects and their relationships.

Text:
\"\"\"
{text}
\"\"\"

Return the JSON now."""
