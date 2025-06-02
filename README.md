##  🖼️RAG Multimodale

RAG Images è un'applicazione che utilizza tecniche di Retrieval-Augmented Generation (RAG) per analizzare immagini, generare descrizioni automatiche e rispondere a domande in linguaggio naturale. Combina il modello BLIP per la generazione di caption, Milvus o FAISS per la ricerca vettoriale e un LLM locale (es. LLaMA 2) per la generazione delle risposte.


---

## Anteprima

![Project Demo GIF](images/gif.gif)

---

## 🚀Funzionalità
📷 Caricamento immagini con descrizione automatica tramite BLIP

🔎 Indicizzazione embedding in Milvus

💬 Domande in linguaggio naturale con risposte generate da un LLM via Ollama

🗂️ Precaricamento automatico di immagini da una cartella (images_folder/images)

🌍 Supporto multilingua (prompt personalizzabile).

⚡ Interfaccia web semplice e interattiva (Streamlit).

🧪 Test automatici su componenti chiave.

---

## 📦 Requisiti
Python ≥ 3.8

streamlit, pillow, torch, transformers, sentence-transformers

faiss-cpu oppure pymilvus se usi Milvus

requests, pytest



---

## Installazione

1. Clona il repository:

```bash
1. Clona il repository

git clone https://github.com/tuo-username/rag-images.git
cd rag-images
2. Crea un ambiente virtuale
Windows:

python -m venv venv
.\venv\Scripts\activate
macOS/Linux:

python -m venv venv
source venv/bin/activate
3. Installa le dipendenze

pip install -r requirements.txt
(Opzionale) Configura eventuali variabili d'ambiente (es. API key, endpoint personalizzati)
```


```bash
⚙️ Avvio dei Servizi
🧠 Avvia Milvus con Docker
Metodo 1 - Docker Compose:

docker-compose -f docker/milvus.yaml up -d
Metodo 2 - Docker singolo comando:

docker run -d --name milvus-standalone -p 19530:19530 -p 19121:19121 milvusdb/milvus:latest


Controlla che Milvus sia attivo:

docker ps
Dovresti vedere milvus-standalone in esecuzione.

Per fermare Milvus:

docker stop milvus-standalone
Per rimuoverlo completamente:

docker rm milvus-standalone

🧠 Avvia Ollama
Vai su https://ollama.com/ e installalo per il tuo sistema operativo.

Apri il terminale ed esegui:

ollama serve
In un altro terminale, scarica il modello desiderato (es. LLaMA 2):

ollama pull llama2:7b
Oppure usa una versione specifica, adatta alla tua GPU.

Avvia il modello:

ollama run llama2

ps: io ho utilizzato il model="llama3.2:1b perchè più adatto alla GPU
OLLAMA: here the documentation https://github.com/ollama/ollama)

Vai sul sito di https://ollama.com/
fai download per utilizzarlo localmente

Ollama ha una REST API per far partire e testare i modelli:
http://localhost:11434/api/generate


Avvia l'app:
streamlit run app.py

oppure Se usi Windows, a volte conviene lanciare Streamlit così:

python -m streamlit run app.py


```

## 🔄 Workflow
Precaricamento immagini

Un pulsante nella UI consente di caricare tutte le immagini dalla cartella images_folder/images

Ogni immagine viene descritta e indicizzata in Milvus automaticamente

Caricamento dinamico

Puoi anche caricare immagini singole tramite il form nella UI

La descrizione viene generata in automatico

Gli embeddings vengono calcolati e salvati in Milvus

Domande e risposte

Poni una domanda sull'immagine

Il sistema recupera i segmenti più rilevanti

Il modello LLM genera una risposta coerente e basata sul contesto visivo

## 🧩 Struttura del Progetto

rag-images/
├── app.py                        # UI Streamlit e flusso principale
├── models/
│   ├── blip_model.py             # Estrazione descrizione da immagine
│   ├── vector_store.py           # Gestione Milvus (embedding e retrieval)
│   ├── gpt_model.py              # Streaming risposte da LLM via Ollama
│   ├── utils.py                  # Funzioni di supporto generiche
│   └── image_embedder.py         # Estrazione degli embeddings dalle immagini
├── images_folder/
│   └── images/                   # Cartella per precaricamento immagini
├── docker/
│   └── milvus.yaml               # (opzionale) Configurazione Docker Compose
├── requirements.txt
└── README.md

## 🧪 Test automatici 
Quando saranno attivi:

Windows:


set PYTHONPATH=. && pytest
macOS/Linux:
PYTHONPATH=. pytest

per uno specifico ad esempio:
set PYTHONPATH=. && pytest tests/test_gpt_model.py

MIT License

Copyright (c) 2025 [Primula Robustelli]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
