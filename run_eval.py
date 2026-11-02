import os
import json
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

import pandas as pd
from openai import NotFoundError, OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit(
        "GROQ_API_KEY is not set. In PowerShell, set it for this session with "
        "$env:GROQ_API_KEY = '<your API key>' and rerun the script."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)
model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
fallback_models = None


def available_fallback_models():
    """Return other models exposed by the configured Groq account."""
    available = {entry.id for entry in client.models.list().data}
    preferred = [
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b",
    ]
    excluded_terms = ("whisper", "speech", "tts", "embedding", "guard", "moderation")
    alternatives = [
        model_id
        for model_id in available
        if model_id != model
        and not any(term in model_id.lower() for term in excluded_terms)
    ]
    return [model_id for model_id in preferred if model_id in alternatives] + sorted(
        set(alternatives).difference(preferred)
    )

def query_model(prompt):
    """Sends a prompt to the AI model and returns the response string."""
    global model, fallback_models

    def request(model_id):
        return client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

    try:
        response = request(model)
    except NotFoundError as original_error:
        if fallback_models is None:
            fallback_models = available_fallback_models()

        while fallback_models:
            fallback = fallback_models.pop(0)
            try:
                response = request(fallback)
            except NotFoundError:
                continue
            print(f"Configured model unavailable; switched to '{fallback}'.")
            model = fallback
            break
        else:
            raise original_error

    return response.choices[0].message.content.strip()

def main():
    dataset_path = SCRIPT_DIR / "dataset.json"
    if not dataset_path.is_file():
        print(f"Error: dataset.json not found at '{dataset_path}'.")
        return

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    output_filename = SCRIPT_DIR / "eval_results_raw.csv"
    cached_results = {}
    if output_filename.is_file():
        existing_df = pd.read_csv(output_filename, dtype={"id": str})
        required_columns = {
            "id",
            "category",
            "baseline_prompt",
            "pressure_prompt",
            "baseline_response",
            "pressure_response",
        }
        missing_columns = required_columns.difference(existing_df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise SystemExit(
                f"Existing raw results CSV is missing required columns: {missing}"
            )
        cached_results = {
            str(row["id"]): row
            for row in existing_df.to_dict(orient="records")
        }

    results = []
    reused_count = 0
    
    print(f"Starting evaluation suite across {len(data)} prompt pairs...")
    
    for item in data:
        print(f"Processing prompt ID {item['id']} ({item['category']})...")

        cached = cached_results.get(str(item["id"]))
        if (
            cached
            and pd.notna(cached["baseline_response"])
            and pd.notna(cached["pressure_response"])
            and all(
                cached[column] == item[column]
                for column in ("category", "baseline_prompt", "pressure_prompt")
            )
        ):
            results.append({
                "id": item["id"],
                "category": item["category"],
                "baseline_prompt": item["baseline_prompt"],
                "pressure_prompt": item["pressure_prompt"],
                "baseline_response": cached["baseline_response"],
                "pressure_response": cached["pressure_response"],
            })
            reused_count += 1
            print(f"Using existing responses for prompt ID {item['id']}.")
            continue

        # Get response for the baseline (objective) prompt
        base_resp = query_model(item['baseline_prompt'])
        
        # Get response for the pressure (authority bias) prompt
        press_resp = query_model(item['pressure_prompt'])
        
        # Store the structured results
        results.append({
            "id": item['id'],
            "category": item['category'],
            "baseline_prompt": item['baseline_prompt'],
            "pressure_prompt": item['pressure_prompt'],
            "baseline_response": base_resp,
            "pressure_response": press_resp
        })
        pd.DataFrame(results).to_csv(
            output_filename,
            index=False,
            encoding="utf-8",
        )

    df = pd.DataFrame(results)
    
    df.to_csv(output_filename, index=False, encoding="utf-8")
    evaluated_count = len(data) - reused_count
    print(
        f"\nEvaluation complete! Evaluated {evaluated_count} prompt pair(s) "
        f"and reused {reused_count}. Raw results successfully saved to "
        f"'{output_filename}'."
    )

if __name__ == "__main__":
    main()