#!/bin/sh

MODEL_NAME="llama3.2:1b"

# Avvia il server Ollama in background
ollama serve &
SERVER_PID=$!

# Aspetta che il server risponda (max 30s)
for i in $(seq 1 30); do
  if curl -s http://localhost:11434/api/v1/healthcheck > /dev/null; then
    echo "Server Ollama pronto!"
    break
  else
    echo "Aspetto Ollama... $i"
    sleep 1
  fi
done

# Scarica modello se non esiste
if ! ollama list | grep -q "$MODEL_NAME"; then
  echo "Scarico modello $MODEL_NAME"
  ollama pull $MODEL_NAME
else
  echo "Modello $MODEL_NAME già presente"
fi

# Mantieni attivo il processo Ollama server
wait $SERVER_PID
