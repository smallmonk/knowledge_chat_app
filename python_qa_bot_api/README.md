# Guided Track Scaffolds

The following APIs are exposed:

```text
GET /health
POST /index
POST /chat
```

Gemini API key is required before running the server:

```bash
export GEMINI_API_KEY="GEMINI_API_KEY"
```

The key is used for final answer generation.

The service could be started using the following command:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
