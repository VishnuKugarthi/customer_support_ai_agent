# backend/agents/triage_agent.py
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.knowledge_base_tools import get_faq_answer


def create_triage_agent(llm: ChatGoogleGenerativeAI) -> AgentExecutor:
    tools = [get_faq_answer]

    triage_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a customer support triage agent. Your primary goal is to understand the user's problem 
            and either answer it directly using the provided FAQ tool, or determine if it's a technical 
            or billing-related issue.
            
            Instructions for routing:
            1. ALWAYS check the FAQ tool first to see if you can answer the question directly
            2. If it's a technical issue, respond with 'ROUTE_TECH:' followed by a brief description of the issue
            3. If it's a billing issue, respond with 'ROUTE_BILLING:' followed by a brief description of the issue
            4. NEVER show routing messages to the user - they should be handled internally
            
            Example responses:
            - FAQ answer: "Here's what I found in our FAQ..."
            - Technical: "ROUTE_TECH: User having issues with app installation"
            - Billing: "ROUTE_BILLING: Question about subscription charges""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    try:
        agent = create_tool_calling_agent(llm, tools, triage_prompt)

        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, max_iterations=3
        )

        return agent_executor

    except Exception as e:
        print(f"Error in create_triage_agent: {str(e)}")
        raise
