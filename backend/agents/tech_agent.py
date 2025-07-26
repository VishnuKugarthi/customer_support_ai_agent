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
                "You are a technical support agent. Your goal is to provide solutions to technical issues "
                "using the `get_tech_solution` tool. "
                "If you cannot find a relevant solution in your knowledge base, you MUST escalate. "
                "When escalating, first check if the user's email is present in the current query or chat history. "
                "If an email is found (e.g., 'my email is example@domain.com'), use the `escalate_to_human_tool` tool with the extracted email. "
                "If NO email is found, your FINAL response MUST be exactly 'NEED_EMAIL_FOR_ESCALATION: [concise summary of issue]'. "
                "Do NOT call `escalate_to_human_tool` if you don't have an email. "
                "If you need more information to provide a solution (before escalating), ask a clarifying question.",
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
