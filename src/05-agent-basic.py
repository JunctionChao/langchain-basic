import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


model = ChatOpenAI(
    model_name=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.3, timeout=30, max_tokens=96e3, max_retries=2
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个使用工具的助手",
)

print(agent) # langgraph.graph.state.CompiledStateGraph
# agent 本质上是一个Graph对象，用于管理状态转换和执行
print(agent.nodes) # nodes 和 edges 组成
# { '__start__': <langgraph.pregel._read.PregelNode>, 
#   'model': <langgraph.pregel._read.PregelNode>, 
#   'tools': <langgraph.pregel._read.PregelNode> }



# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "查询下今天北京的天气怎么样？"}]}
)
# agent将所有的message保存
# print(response) # {'messages': [HumanMessage(content='今天北京天气怎么样？', id='3d0f928e-e768-475a-a270-0ec23b0a2ff5'), AIMessage(content='好的，请告诉我您要查询的城市名称。', id='0d0f928e-e768-475a-a270-0ec23b0a2ff5')]}

messages = response["messages"]
for message in messages:
    message.pretty_print()
'''
================================ Human Message =================================

查询下今天北京的天气怎么样？
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_-8101194421465350163)
 Call ID: call_-8101194421465350163
  Args:
    city: 北京
================================= Tool Message =================================
Name: get_weather

It's always sunny in 北京!
================================== Ai Message ==================================

根据您的查询，我查询了北京的天气情况。根据API调用结果，今天北京的天气是晴朗的！
'''