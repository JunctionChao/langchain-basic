from llama_index.core import VectorStoreIndex, Document, Settings, load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
Settings.llm = None

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")


# 4. 加载索引并执行查询
loaded_index = load_index_from_storage(
    storage_context=StorageContext.from_defaults(persist_dir=persist_path)
)

query = "LlamaIndex 是做什么的？"
results = loaded_index.as_query_engine().query(query)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for node in results.source_nodes:
    print(f"- {node.text}")
