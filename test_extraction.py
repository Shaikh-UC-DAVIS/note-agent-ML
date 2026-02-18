"""
Smoke test for the real LLM extraction pipeline.
Run: python test_extraction.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

from ml.extraction import LLMExtractor

def main():
    print("=" * 50)
    print("  EXTRACTION SMOKE TEST")
    print("=" * 50)

    # Verify API key is set
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("✗ GROQ_API_KEY not set. Create a .env file.")
        return

    print(f"✓ API key loaded (ends in ...{api_key[-4:]})")

    # Create extractor
    extractor = LLMExtractor()
    print(f"✓ LLMExtractor initialized (model={extractor.model})")

    # Test passage
    test_text = """
    Photosynthesis is the process by which plants convert sunlight into energy.
    This process requires water and carbon dioxide.
    Some researchers question whether artificial photosynthesis could replace solar panels.
    Task: Review the latest papers on artificial photosynthesis efficiency.
    """

    print(f"\n── Input Text ──")
    print(test_text.strip())
    print(f"── End Input ──\n")

    # Extract
    result = extractor.extract(test_text)

    # Display results
    print(f"\n── Results ──")
    print(f"Objects extracted: {len(result.objects)}")
    for obj in result.objects:
        print(f"  [{obj.type}] {obj.id}: \"{obj.canonical_text}\" (conf={obj.confidence})")

    print(f"\nLinks extracted: {len(result.links)}")
    for link in result.links:
        print(f"  {link.source_id} --[{link.type}]--> {link.target_id} (conf={link.confidence})")

    # Assertions
    assert len(result.objects) > 0, "Expected at least one extracted object!"
    print(f"\n✓ PASSED — {len(result.objects)} objects, {len(result.links)} links extracted.")

if __name__ == "__main__":
    main()
