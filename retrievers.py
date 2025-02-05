
from imports import *

class Retriever:
    @abstractmethod
    def search(self, q):
        pass



class BM25Retriever(Retriever):
    def __init__(self, index_ref, num_results=100):
        import pyterrier as pt
        self.retriever = pt.BatchRetrieve(index_ref, wmodel="BM25", num_results=num_results)
    
    def search(self, query):
        return self.retriever.search(query)