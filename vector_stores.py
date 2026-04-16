from langchain_chroma import Chroma
import config_data as config
from langchain_core.documents import Document

class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding:嵌入模型的传入        
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name, 
            embedding_function=self.embedding, 
            persist_directory=config.persist_directory
        )
    def get_retriever(self):
        """
        返回向量检索器，方便加入链
        """
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_filenum})

    def similarity_search_with_relevance_scores(self, query: str, k: int) -> list[tuple[Document, float]]:
        """返回带相关度分数的检索结果。"""
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k)
    
if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我的体重180斤，身高1米8，给出尺码推荐")
    print(res)