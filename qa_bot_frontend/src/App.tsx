import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

interface SourceInfo {
  source: string;
  heading: string;
  score: number;
  content: string;
}

interface ChatResponse {
  answer: string;
  sources: SourceInfo[];
}

function App() {
  const [query, setQuery] = useState('')
  const [response, setResponse] = useState<ChatResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data: ChatResponse = await res.json();
      setResponse(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An error occurred while fetching the response.');
      }
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '2rem' }}>
      <h1>Knowledge Base Q&A Bot</h1>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '2rem' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question..."
          style={{ padding: '0.5rem', fontSize: '1rem' }}
          disabled={isLoading}
        />
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button type="submit" disabled={isLoading || !query.trim()} style={{ padding: '0.5rem 1rem', fontSize: '1rem' }}>
            {isLoading ? 'Sending...' : 'Send'}
          </button>
          <button type="button" style={{ padding: '0.5rem 1rem', fontSize: '1rem' }}>
            Heath
          </button>
          <button type="button" style={{ padding: '0.5rem 1rem', fontSize: '1rem' }}>
            Index
          </button>
        </div>
      </form>

      {error && (
        <div style={{ color: 'red', marginBottom: '1rem' }}>
          {error}
        </div>
      )}

      {response && (
        <div style={{ border: '1px solid #ccc', padding: '1rem', borderRadius: '4px' }}>
          <h2>Response</h2>
          <p>{response.answer}</p>

          {response.sources && response.sources.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <h3>Sources</h3>
              <ul style={{ listStyleType: 'none', padding: 0 }}>
                {response.sources.map((source, index) => (
                  <li key={index} style={{ marginBottom: '1rem', backgroundColor: '#f9f9f9', padding: '0.5rem', borderRadius: '4px' }}>
                    <strong>{source.source}#{source.heading}</strong> (Score: {source.score.toFixed(2)})
                    <pre style={{ whiteSpace: 'pre-wrap', backgroundColor: '#eee', padding: '0.5rem', borderRadius: '4px', fontSize: '0.85em' }}>
                      {source.content}
                    </pre>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
