# Guided Track Scaffolds

The following APIs are exposed:

```text
GET /health
POST /index
POST /chat
```

An OpenAI API key is required before running the server:

```bash
export OPENAI_API_KEY="sk-..."
```

Vector RAG uses the key for final answer generation and embeddings.

The service could be started using the following command:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
