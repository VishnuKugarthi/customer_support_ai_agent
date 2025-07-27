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
                """You are a customer support triage agent. Your primary goal is to help customers by routing them to the right team or providing FAQ answers.

STRICT ROUTING RULES:

1. TECHNICAL ISSUES (Route to TECH) - If query contains ANY of these indicators:
   - Internet/network/wifi/connection problems or issues
   - App or website issues
   - Login/access problems
   - Error messages
   - System performance
   - Device problems
   - Software/hardware concerns
   - Loading/speed issues
   YOU MUST respond with EXACTLY: "ROUTE_TECH: [brief description]"

2. BILLING ISSUES (Route to BILLING) - If query contains ANY of these:
   - Customer IDs (e.g., "customer_101", "customer 102")
   - Words like: balance, payment, bill, charge, plan, account
   - Financial symbols ($)
   YOU MUST respond with EXACTLY: "ROUTE_BILLING: [brief description]"

3. FAQ (Use get_faq_answer tool) - ONLY if:
   - Query is about general policies/information
   - NO technical issues mentioned
   - NO billing/account issues
   - NO customer IDs
   
CRITICAL RULES:
- For ANY internet/connection/network issue → ALWAYS reply "ROUTE_TECH: [exact issue from user]"
- For ANY technical problem → MUST use ROUTE_TECH format
- NEVER use FAQ tool for technical or billing issues
- When in doubt → Route to TECH
- Keep descriptions brief and use user's own words""",
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
