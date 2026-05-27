# Knowledge Base Q&A Bot

This project consists of a Knowledge Base Q&A Bot with a Python FastAPI backend and a React frontend.

## Project Structure

The repository is organized into two main components:

- **`python_qa_bot_api/`**: The backend API built with FastAPI. It handles document indexing, retrieval, and integration with the Gemini API for generating answers.
- **`qa_bot_frontend/`**: The frontend application built with React, TypeScript, and Vite. It provides a user interface for asking questions, indexing documents, and checking the system's health.

## Getting Started

To run the complete system, you need to start both the backend and the frontend.

### Prerequisites

- Python 3.11
- Node.js (v20 recommended)
- npm or yarn
- A Gemini API Key

### Running the Backend

1. Navigate to the `python_qa_bot_api` directory:
   ```bash
   cd python_qa_bot_api
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
4. Start the backend server:
   ```bash
   uvicorn app.main:app --reload
   ```
   The backend will be available at `http://localhost:8000`.

### Running the Frontend

1. Navigate to the `qa_bot_frontend` directory:
   ```bash
   cd qa_bot_frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will be available at the URL provided in the terminal (usually `http://localhost:5173`).

## How They Work Together

1. **Indexing**: The frontend can trigger the indexing process by sending a POST request to the backend's `/index` endpoint. The backend processes documents and prepares them for retrieval.
2. **Q&A**: When a user asks a question in the frontend, it sends a POST request to the backend's `/chat` endpoint. The backend retrieves relevant information from the indexed knowledge base and uses the Gemini API to generate a concise answer.
3. **Health Check**: The frontend can verify the backend's status by calling the `/health` endpoint.
