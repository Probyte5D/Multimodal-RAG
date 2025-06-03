import requests
import json
import os

def generate_response_stream(
    context: list[str],
    user_input: str,
    model="llama3.2:1b",
    lang="it",
    max_tokens=300
):
    ollama_host = os.getenv("OLLAMA_HOST")
    if not ollama_host:
        raise RuntimeError("OLLAMA_HOST non è definito. Verifica le variabili d'ambiente.")

    url = f"http://{ollama_host}:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    lang_prompts = {
        "it": (
            "Rispondi in italiano.\n"
            "Sei un assistente esperto nell'analisi delle immagini. "
            "Il contesto contiene descrizioni visive di una o più immagini. "
            "Analizza attentamente il contenuto, elenca tutti gli oggetti identificabili, "
            "indica quantità, colori, posizioni e trascrivi qualsiasi testo visibile. "
            "Evita interpretazioni non supportate visivamente. "
            "Utilizza un linguaggio chiaro, preciso e conciso."
        ),
        "en": (
            "Reply in English.\n"
            "You are an expert assistant in image analysis. "
            "The context contains visual descriptions of one or more images. "
            "Analyze carefully, list all identifiable objects, indicate quantities, colors, positions, and transcribe any visible text. "
            "Avoid unsupported interpretations. "
            "Use clear, precise, and concise language."
        )
    }

    if lang not in lang_prompts:
        lang = "it"  # fallback sicuro

    system_prompt = lang_prompts[lang]
    context_str = "\n---\n".join(context)

    print("=== CONTEXT PASSED TO GPT ===")
    print(context)
    print("=== USER INPUT ===")
    print(user_input)
    print("============================")

    prompt = f"{system_prompt}\n\nContesto:\n{context_str}\n\nRichiesta:\n{user_input}"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        for line in response.iter_lines():
            if line:
                line_data = line.decode('utf-8').strip()
                if line_data == "[DONE]":
                    break
                try:
                    data = json.loads(line_data)
                    if "response" in data:
                        yield data["response"]
                        if data.get("done", False):
                            break
                    else:
                        yield f"[⚠️ Chunk senza 'response': {data}]"
                except Exception as err:
                    yield f"[⚠️ Errore parsing JSON: {err}]"
    except Exception as e:
        yield f"[❌ Errore di rete: {e}]"
