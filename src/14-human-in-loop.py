import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command


load_dotenv()

# 系统提示词
SYSTEM_PROMPT = """
你是一个诙谐的天气预测助手，你会用幽默的方式回答用户的问题。
你有两个可以使用的工具:
- get_weather_for_location: 通过城市名称获取天气信息
- get_user_location: 获取用户当前位置
"""

# 工具定义
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    import random
    weather_options = ["sunny", "cloudy", "rainy", "windy"]
    return f"It's {random.choice(weather_options)} in {city}"

# 上下文对象，提供一些静态信息，用户 ID、数据库连接或其他代理调用的依赖关系
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "上海" if user_id == "1" else "北京"


# 定义结构化输出格式
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


# 记忆存储/检查点
checkpointer = InMemorySaver()

agent = create_agent(
    model=ChatOpenAI(
        model_name=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.3, timeout=30, max_tokens=96e3, max_retries=5
    ),
    checkpointer=checkpointer,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context, # 提供一个上下文schema
    response_format=ToolStrategy(ResponseFormat),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_user_location": True, # 所有决策都允许，approve, reject, edit
                "get_weather_for_location": {
                    "allowed_decisions": ["approve", "reject"]
                }
            },
            description_prefix="工具执行待审批" # 设置中断消息提示描述
        )
    ]
)


config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "现在室外天气如何?"}]},
    config=config,
    context=Context(user_id="1") # 上下文数据传入
)
messages = response['messages']
print(f"历史消息一共 {len(messages)} 条")
for message in messages:
    message.pretty_print()

if "__interrupt__" in response:
    print("---有中断消息:---")
    print(response["__interrupt__"])
    interrupt = response["__interrupt__"][0]
    for request in interrupt.value["action_requests"]:
        print(request["description"])

# 通过指令，继续执行
response = agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}
    ),
    config=config,
    context=Context(user_id="1")
)
messages = response['messages']
print(f"历史消息一共 {len(messages)} 条")
for message in messages:
    message.pretty_print()
if "__interrupt__" in response:
    print("---有中断消息:---")
    print(response["__interrupt__"])
    interrupt = response["__interrupt__"][0]
    for request in interrupt.value["action_requests"]:
        print(request["description"])

# 第二个通过指令
response = agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}
    ),
    config=config,
    # context=Context(user_id="1")
)

messages = response['messages']
print(f"历史消息一共 {len(messages)} 条")
for message in messages:
    message.pretty_print()

