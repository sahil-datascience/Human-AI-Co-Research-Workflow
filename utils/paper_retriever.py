

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class PaperRetriever:

    def __init__(self, collection_name, persist_directory, embedding_model):
        self._vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(model=embedding_model),
            persist_directory=persist_directory
        )

    def for_paper(self, paper_id: str, k: int = 6):
        return self._vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"paper_id": paper_id}
            }
        )
