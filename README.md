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

3. Create a `.env` file in the `backend` directory with the following variables:

   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your_email@example.com
   SMTP_PASSWORD=your_email_password
   SMTP_FROM_EMAIL=your_email@example.com
   ```

4. Start the backend server:

   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Frontend Setup

1. Navigate to the `frontend` directory and open `index.html` directly in the browser.

## Features

- Multi-agent system with specialized agents for:
  - Triage (Initial query assessment)
  - Technical Support
  - Billing Support
- Automated FAQ responses
- Smart routing based on query type
- Escalation system that assigns a ticket ID and sends it to user via email for future reference.
- Real-time chat interface

## API Endpoints

- `POST /chat`: Send a customer query
  - Request body: `{"message": "string", "chat_history": []}`
  - Response: `{"response": "string"}`

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| GOOGLE_API_KEY | Google AI API key for LangChain | Yes |

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

## Security Considerations

- Keep your `.env` file secure and never commit it to version control
- Always validate and sanitize user inputs
- Implement rate limiting for production use
- Use HTTPS in production
- Regularly update dependencies

## Troubleshooting

Common issues and solutions:

1. CORS errors:
   - Ensure your main.py file contain the frontend URL as a origin.
   - Check if frontend is using the correct backend URL

2. Agent not responding:
   - Verify Google API key is valid
   - Check knowledge base files exist and are properly formatted

3. Connection errors:
   - Ensure backend server is running
   - Verify ports are not blocked by firewall
