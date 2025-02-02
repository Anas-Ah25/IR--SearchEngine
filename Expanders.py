from preprocessing import *  # Make sure your Preprocessing class is available



class Expander(ABC):
    @abstractmethod
    def expand(self, query, **kwargs): # abstract method for tools 
        pass

# ============================== Expanding Methods ==============================

''' ---------------------------- Rm3 Expander ---------------------------- '''
class RM3(Expander):
    def __init__(self, bm25, index, preprocessor, fb_terms=10, fb_docs=100):
        """
        Rm3 used as a part of pipeline which take the retreived data and expand, 
         where it uses the prf on the best retreived documents, to get the words with highest probability, expanding the main query with them 
        """
        self.bm25 = bm25 # retriever
        self.index = index # index reference (pyterrier)
        self.preprocessor = preprocessor # from the preprocessing class
        self.fb_terms = fb_terms # number of feedback terms
        self.fb_docs = fb_docs # number of feedback documents
        self.pt_index = pt.IndexFactory.of(self.index)  # create an index object from the reference
        self.expander = pt.rewrite.RM3(self.pt_index, fb_terms=self.fb_terms, fb_docs=self.fb_docs)  # expander 
        
    def expand(self, query, **kwargs):
        processed_query = self.preprocessor.preprocessing(query)
        # retreival using bm25 then expand (using the data in expanding)
        ret_expanded = self.bm25 >> self.expander
        result_df = ret_expanded.search(processed_query)
        expanded_query = result_df.iloc[0]["query"] # get the expanded query 
        print('not formatted query: ', expanded_query)
        # remove the first token, shape now: ' '
        expanded_query_formatted = ' '.join(expanded_query.split()[1:]) # formatting to be normal scentence like input query again
        return expanded_query_formatted
# -----------------------------------------------------------------------------------------------------
# test the class 


bm25 = pt.BatchRetrieve(index, wmodel="BM25")
rm3 = RM3(bm25, index, preprocessor)
query = "What is the capital of France?"
expanded_query = rm3.expand(query)

