# backend/agents/tech_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.knowledge_base_tools import get_tech_solution, escalate_to_human_tool


def create_tech_agent(llm: ChatGoogleGenerativeAI) -> AgentExecutor:
    tech_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a technical support agent specializing in resolving technical issues. Follow these guidelines strictly:

1. ALWAYS start by using get_tech_solution tool with the user's exact query or issue description
   - If a solution is found, provide it immediately
   - Only ask for more details if no solution is found or the solution needs clarification

2. For Common Issues:
   - Internet/Network: Use "internet issue", "connection problems", "internet not working"
   - App Problems: Use "app crashing" or "software installation failed"
   - Always match the closest keyword in our knowledge base

3. When No Direct Solution:
   - Internet Issues: Ask about connection type, error messages, recent changes
   - App Issues: Request error messages and steps to reproduce
   - Device Problems: Ask about device type, OS version, updates

4. Response Format:
   - Start with the solution from get_tech_solution if available
   - Only ask questions if no solution is found
   - Keep responses clear and step-by-step

5. Escalation Rules (only after trying solutions):
   - Escalate if solution doesn't resolve the issue
   - Escalate if issue is too complex
   - Format: 'NEED_EMAIL_FOR_ESCALATION: [summary]' if no email provided
   
IMPORTANT: NEVER skip using get_tech_solution tool first - it contains our approved solutions.""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Ensure the tools are properly wrapped and used
    tech_tools = [get_tech_solution, escalate_to_human_tool]

    tech_agent = create_tool_calling_agent(llm, tech_tools, tech_prompt)
    tech_agent_executor = AgentExecutor(
        agent=tech_agent, tools=tech_tools, verbose=True
    )
    return tech_agent_executor
