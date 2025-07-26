# Customer Support AI Agent

An intelligent customer support system that uses AI agents to handle technical, billing, and general inquiries.

## Project Structure

```
customer_support_ai_agent/
├── backend/                 # FastAPI backend
│   ├── agents/             # AI agent implementations
│   ├── data/               # Knowledge base data
│   ├── tools/              # Agent tools
│   └── main.py            # Main FastAPI application
└── frontend/              # Frontend implementation
    └── index.html         # Main HTML interface
```

## Prerequisites

- Python 3.8+
- Node.js (for running frontend locally)
- Google AI API key

## Setup Instructions

### Backend Setup

1. Create and activate a virtual environment:

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the `backend` directory:

   ```
   GOOGLE_API_KEY=your_google_api_key_here
   CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
   ```

4. Start the backend server:

   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. You can serve the frontend using any static file server. For example:

   ```bash
   python -m http.server 3000
   ```

   Or use Node.js's `http-server`:

   ```bash
   npx http-server -p 3000
   ```

## Features

- Multi-agent system with specialized agents for:
  - Triage (Initial query assessment)
  - Technical Support
  - Billing Support
- Automated FAQ responses
- Smart routing based on query type
- Human escalation system
- Real-time chat interface

## API Endpoints

- `POST /chat`: Send a customer query
  - Request body: `{"message": "string", "chat_history": []}`
  - Response: `{"response": "string"}`

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| GOOGLE_API_KEY | Google AI API key for LangChain | Yes |
| CORS_ORIGINS | Allowed CORS origins (comma-separated) | Yes |

## Development Notes

### Adding New Knowledge Base Items

1. Add new FAQ entries to `backend/data/faq_knowledge_base.json`
2. Add technical solutions to `backend/data/tech_kb.json`
3. Add billing information to `backend/data/billing_db.json`

### Modifying Agent Behavior

Agent configurations can be modified in their respective files under `backend/agents/`:

- `triage_agent.py`
- `tech_agent.py`
- `billing_agent.py`

## Testing

Run the test suite:

```bash
cd backend
pytest
```

## Security Considerations

- Keep your `.env` file secure and never commit it to version control
- Always validate and sanitize user inputs
- Implement rate limiting for production use
- Use HTTPS in production
- Regularly update dependencies

## Troubleshooting

Common issues and solutions:

1. CORS errors:
   - Ensure CORS_ORIGINS in .env matches your frontend URL
   - Check if frontend is using the correct backend URL

2. Agent not responding:
   - Verify Google API key is valid
   - Check knowledge base files exist and are properly formatted

3. Connection errors:
   - Ensure backend server is running
   - Verify ports are not blocked by firewall
