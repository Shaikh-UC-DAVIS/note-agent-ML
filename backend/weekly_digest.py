import os
import json
from collections import defaultdict
from openai import OpenAI
from ml.config import config


def generate_llm_summary(objects):
    if not objects:
        return "No activity this week."

    openai_key = os.environ.get("OPENAI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    resolved_key = openai_key or groq_key
    if resolved_key and (resolved_key.startswith("sk-") or "openai" in str(resolved_key).lower()):
        base_url = "https://api.openai.com/v1"
        model = config.get("model", "gpt-4o")
    else:
        base_url = "https://api.groq.com/openai/v1"
        model = config.get("model", "llama-3.3-70b-versatile")
    client = OpenAI(api_key=resolved_key, base_url=base_url)

    # group by type
    groups = defaultdict(list)
    for o in objects[:50]:
        groups[o["type"]].append(o["text"])

    lines = []
    for obj_type, texts in groups.items():
        lines.append(f"{obj_type} ({len(texts)} objects):")
        for t in texts[:5]:
            lines.append(f'  - "{t}"')
    content = "\n".join(lines)

    prompt = f"""
You are generating a weekly executive summary from extracted knowledge objects.

Objects grouped by type:
{content}

For each type identify the dominant theme from the sample texts.
Return JSON:
{{
  "summary": "<2-3 sentence high-level summary>",
  "type_summaries": [
    {{"type": "<type>", "count": <int>, "theme": "<dominant theme>", "detail": "<one sentence>"}}
  ],
  "themes": ["<theme1>", "<theme2>"],
  "insights": ["<insight1>", "<insight2>"]
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert at summarizing structured knowledge."},
                {"role": "user", "content": prompt},
            ],
            temperature=config['temperature'],
            max_tokens=500,
            timeout=config['timeout'],
        )
        raw = response.choices[0].message.content
        try:
            parsed = json.loads(raw)
        except:
            return raw  # fallback if JSON breaks

        output = parsed.get("summary", "") + "\n\n"

        if "type_summaries" in parsed:
            output += "This week:\n"
            for ts in parsed["type_summaries"]:
                line = f"- {ts.get('count')} new {ts.get('type')}s"
                if ts.get("theme"):
                    line += f" centred around '{ts.get('theme')}'"
                if ts.get("detail"):
                    line += f". {ts.get('detail')}"
                output += line + "\n"
            output += "\n"

        if "themes" in parsed:
            output += "Themes:\n"
            for t in parsed["themes"]:
                output += f"- {t}\n"

        if "insights" in parsed:
            output += "\nInsights:\n"
            for i in parsed["insights"]:
                output += f"- {i}\n"

        return output.strip()

    except Exception as e:
        return f"[LLM Error] {e}"