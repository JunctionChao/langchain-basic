# agent: model, tools, system_prompt, checkpointer, middleware
# 多agent架构
# 1. supervisor agent 监督模式, 负责协调其他agent的工作，根据任务分配和结果反馈，确定下一步的操作。
#    user -> supervisor agent -> worker agents
# 2. handoff agent 移交/轮换模式, 负责将任务从一个agent传递给另一个agent，确保任务的流畅过渡。
#    user -> agent1, user -> agent2, ...


# supervisor agent
# 1. 创建多个 worker agent, 有各自的 tools
# 2. 将多个worker agent 封装成新的tools
# 3. 创建supervisor agent, 使用第二步封装的agents作为自己的tools


from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from calendar_agent import calendar_agent
from email_agent import email_agent
import os


@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text



SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
    "请用中文回复"
)

supervisor_agent = create_agent(
    model=ChatOpenAI(
        model_name=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.3, timeout=30, max_tokens=96e3, max_retries=2
    ),
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)


def  test_supervisor_agent():

    query = "hello"
    for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": query}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()

    query = (
        "Schedule a named Design Team meeting on 2025-12-10 at 9am 1 hour, "
        "then send email to notify the team members include zhangsan@example.com, lisi@example.com"
        "with subject 'Design Team Meeting'"
    ) 
    
    for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": query}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    test_supervisor_agent()

