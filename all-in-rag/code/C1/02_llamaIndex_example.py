# llama-index框架更偏向数据层的RAG应用，更靠近存储/索引
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
# from llama_index.llms.deepseek import DeepSeek
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_openai import ChatOpenAI
from llama_index.llms.langchain import LangChainLLM


load_dotenv()

# Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
# Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
Settings.llm = LangChainLLM(
    llm=ChatOpenAI(
        model_name=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.3,
        timeout=30,
        max_tokens=4096,
        max_retries=5
    )
)

# Settings.llm = ChatOpenAI(
#     model_name=os.getenv("LLM_MODEL_ID"),
#     api_key=os.getenv("LLM_API_KEY"),
#     base_url=os.getenv("LLM_BASE_URL"),
#     temperature=0.3,
#     timeout=30,
#     max_tokens=4096,
#     max_retries=5
# )
Settings.embed_model = OllamaEmbedding(
    base_url="http://localhost:11434",
    # model_name="bge-large:latest",
    model_name='quentinz/bge-large-zh-v1.5:latest',
    # model_name='nomic-embed-text:latest',
    embed_batch_size=4,
)

docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))