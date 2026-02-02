import requests
import os

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


def ask_perplexity(query, context):

    url = "https://api.perplexity.ai/chat/completions"

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = """
You are an assistant answering questions strictly from the given context.
If the answer is not present in the context, say:
"I could not find this information in the document."
"""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}

Answer:
"""
            }
        ]
    }

    r = requests.post(url, json=payload, headers=headers)
    return r.json()["choices"][0]["message"]["content"]