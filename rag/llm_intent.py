import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"   # or mistral, phi3, etc.

PROMPT_TEMPLATE = """
You are a healthcare data privacy classifier.

Classify the user query into ONE of the following categories:

AGGREGATE:
- Counts
- Totals
- Statistics
- Disease-wise numbers
- Non-identifiable summaries

IDENTIFIABLE:
- Patient names
- Patient IDs
- Insurance IDs
- Lists of patients
- Any personal or sensitive data

ONLY respond with:
AGGREGATE
or
IDENTIFIABLE

Query:
"{query}"
"""

def classify_intent(query: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": PROMPT_TEMPLATE.format(query=query),
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        # Fail-safe: block on error
        return "IDENTIFIABLE"

    result = response.json().get("response", "").strip().upper()

    if "AGGREGATE" in result:
        return "AGGREGATE"

    return "IDENTIFIABLE"
