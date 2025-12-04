import chromadb


def list_collections(db_path):
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    print(f'数据库{db_path}中共有 {len(collections)} 个集合')
    for i, collection in enumerate(collections):
        print(f'collection {i}: {collection.name}, 向量数: {collection.count()}')

def del_collection(db_path, collection_name):
    try:
        client = chromadb.PersistentClient(path=db_path)
        client.delete_collection(collection_name)
    except Exception as e:
        print(f'删除集合{collection_name}出错: {e}')


if __name__ == '__main__':
    db_path = './chroma_db'
    list_collections(db_path)
 