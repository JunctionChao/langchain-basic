from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatZhipuAI
from langchain.chat_models import init_chat_model

import os


load_dotenv()
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")


# 使用openai兼容模式
model = ChatOpenAI(
    model_name=LLM_MODEL_ID,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=0.3,
    timeout=30,
    max_tokens=96e3,
    max_retries=2
)

# model = ChatZhipuAI(
#     model_name=LLM_MODEL_ID,
#     api_key=LLM_API_KEY,
#     base_url=LLM_BASE_URL,
#     temperature=0.7,
#     timeout=30,
#     max_tokens=96e3,
#     max_retries=2
# )


# 暂时不支持zhipuai
# model = init_chat_model(
#     model=LLM_MODEL_ID,
#     model_provider="zhipuai",
#     api_key=LLM_API_KEY,
#     base_url=LLM_BASE_URL,
#     temperature=0.7,
#     timeout=30,
#     max_tokens=96e3,
#     max_retries=2
# )

for chunk in model.stream("写一首五言绝句，不要多余解释"):
    print(chunk.content, end="", flush=True)