import requests
import json
import os

def generate_response_stream(context: list[str], question: str, model="llama3.2:1b", lang="it", max_tokens=300):
    ollama_host = os.getenv("OLLAMA_HOST")
    if not ollama_host:
        raise RuntimeError("OLLAMA_HOST non è definito. Verifica le variabili d'ambiente.")

    url = f"http://{ollama_host}:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    lang_prompts = {
        "it": (
            "Sei un assistente esperto nell'analisi delle immagini. "
            "Il primo contesto è una descrizione dettagliata e strutturata dell'immagine caricata. "
            "Se l'immagine contiene molti oggetti, elenca tutti quelli identificabili, "
            "specificando la quantità di ciascuno. "
            "Se l'immagine è composta da una griglia o collage di sotto-immagini, "
            "descrivi gli oggetti in ogni sezione della griglia, indicando la posizione quando possibile. "
            "Se c'è un solo oggetto, descrivilo dettagliatamente. "
            "Indica chiaramente se qualche oggetto è parzialmente visibile o poco definito. "
            "Utilizza un linguaggio chiaro, preciso e conciso."
        ),
        "en": (
            "You are an expert assistant in image analysis. "
            "The first context is a detailed and structured description of the uploaded image. "
            "If the image contains many objects, list all identifiable ones specifying their quantity. "
            "If the image is composed of a grid or collage of sub-images, describe the objects in each section of the grid, indicating their position if possible. "
            "If there is only one object, describe it in detail. "
            "Clearly state if any object is partially visible or unclear. "
            "Use clear, precise, and concise language."
        ),
    }

    system_prompt = lang_prompts.get(lang, lang_prompts["it"])
    context_str = "\n---\n".join(context)
    print("=== CONTEXT PASSED TO GPT ===")
    print(context)
    print("============================")

    prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {question}"

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
                        yield f"[Warning: chunk senza 'response' -> {data}]"
                except Exception as err:
                    yield f"[Parsing error: {err}]"
    except Exception as e:
        yield f"[Error: {e}]"
