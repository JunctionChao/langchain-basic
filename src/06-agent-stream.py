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


# for event in agent.stream(
#     {"messages": [{"role": "user", "content": "查询下今天北京的天气怎么样？"}]},
#     stream_mode="values"  # values模式  返回所有状态完整信息
# ):
#     messages = event["messages"]
#     print(f"历史消息：{len(messages)} 条")
#     messages[-1].pretty_print() # 只取最后一条消息，即最新消息



for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "查询下今天北京的天气怎么样？"}]},
    stream_mode="messages"  # token 方式, 每次返回一个token
):
    print(chunk[0].content, end='', flush=True)
    # print(chunk)
    # (
    #   AIMessageChunk(content='我', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--02107c84-92f3-4945-ade8-7a21f3425417'), 
    #   {'langgraph_step': 3, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:8bdd8382-7e61-57a5-93c9-e4d7b9a86a12', 'checkpoint_ns': 'model:8bdd8382-7e61-57a5-93c9-e4d7b9a86a12', 'ls_provider': 'openai', 'ls_model_name': 'glm-4-flash', 'ls_model_type': 'chat', 'ls_temperature': 0.3, 'ls_max_tokens': 96000}
    # )
