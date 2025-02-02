from preprocessing import *
from abc import ABC, abstractmethod

class Expander(ABC):
    @abstractmethod
    def expand(self, query, **kwargs):
        pass


    class RM3(Expander):
        def __init__()
            

    class BM25(Expander):
        def __init__(self, index, fb_terms=10, fb_docs=100):
            self.index = index
            self.fb_terms = fb_terms
            self.fb_docs = fb_docs


    class glove(Expander):
        def __init__():

      

    class word2vec(Expander):
        def __init__():
            

    class bert(Expander):
        def __init__():
      