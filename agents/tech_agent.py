# agents/tech_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools.knowledge_base_tools import get_tech_solution, escalate_to_human


def create_tech_agent(llm: ChatGoogleGenerativeAI) -> AgentExecutor:
    tech_prompt_template = """
    You are a technical support agent. Your goal is to provide solutions to technical issues
    using the `get_tech_solution` tool. If you cannot find a relevant solution in your knowledge base
    or if the user requires further assistance that you cannot provide, you MUST use the
    `escalate_to_human` tool with a clear summary of the issue and what has been attempted.
    If the user provides their email in the query (e.g., "my email is example@domain.com"), extract it and pass it to the `escalate_to_human` tool.

    Tools available to you:
    {tools}

    Use the following format for your thinking process:
    Question: the input technical question
    Thought: you should always think about what to do to solve the technical problem using the available tools.
    Action: the action to take, which must be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer or if escalation is needed.
    Final Answer: the final solution to the technical problem, or the output of escalate_to_human.

    Current conversation:
    {chat_history}
    User query: {input}
    {agent_scratchpad}
    """

    tech_prompt = PromptTemplate.from_template(tech_prompt_template)
    tech_tools = [get_tech_solution, escalate_to_human]

    tech_agent = create_react_agent(llm, tech_tools, tech_prompt)
    tech_agent_executor = AgentExecutor(
        agent=tech_agent, tools=tech_tools, verbose=True, handle_parsing_errors=True
    )
    return tech_agent_executor
