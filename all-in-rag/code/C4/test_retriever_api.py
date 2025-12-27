from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 创建简单的向量存储
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.from_texts(['test document 1', 'test document 2'], embeddings)
retriever = vectorstore.as_retriever()

# 查看可用方法
print('检索器类型:', type(retriever))
print('可用方法:', [m for m in dir(retriever) if not m.startswith('_')])

# 测试invoke方法
try:
    result = retriever.invoke('test query')
    print('invoke方法成功:', result)
except Exception as e:
    print('invoke方法失败:', e)

# 测试直接检索
result = vectorstore.similarity_search('test query')
print('vectorstore.similarity_search结果:', result)
