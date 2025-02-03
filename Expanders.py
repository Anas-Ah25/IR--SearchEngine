from preprocessing import *  # Make sure your Preprocessing class is available
import pandas as pd
import os

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
        expanded_query = result_df.iloc[0]["query"] # get the expanded query, weighted query
        '''
        # print('not formatted query: ', expanded_query)
        expansion gives weighted query not just string, shape now: ' auror^0.044444438 aurora^0.044444438 occur^0.044444438 '
        # formattedQuery = ' '.join(expanded_query.split()[1:])
        '''
        return expanded_query 
# -----------------------------------------------------------------------------------------------------

# ''' ---------------------------- Glove Expander ---------------------------- '''

# class Glove(Expander):
#     def __init__(self, glove_model):
#         """
#         Args:
#             glove_model: A preloaded GloVe embedding model or dictionary.
#         """
#         self.glove_model = glove_model
        
#     def expand(self, query, **kwargs):
#         """
#         Placeholder for a GloVe-based expansion method.
#         You might, for example, retrieve similar words from your glove_model.
#         """
#         # TODO: Implement expansion logic using self.glove_model.
#         return query
    


# ###############################################################################
# # Word2Vec Expander (Placeholder)
# ###############################################################################

# class Word2Vec(Expander):
#     def __init__(self, word2vec_model):
#         """
#         Args:
#             word2vec_model: A preloaded Word2Vec model.
#         """
#         self.word2vec_model = word2vec_model
        
#     def expand(self, query, **kwargs):
#         """
#         Placeholder for a Word2Vec-based expansion method.
#         """
#         # TODO: Implement expansion logic using self.word2vec_model.
#         return query

# ###############################################################################
# # BERT Expander
# ###############################################################################

# class Bert(Expander):
#     def __init__(self, device=None, model_name: str = "bert-base-uncased"):

#         self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
#     def _encode(self, text, max_length=32):
#         return self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             truncation=True,
#             max_length=max_length,
#             padding="max_length",
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
        
#     def _get_embedding(self, text):
#         tokens = self._encode(text)
#         input_ids = tokens["input_ids"].to(self.device)
#         attention_mask = tokens["attention_mask"].to(self.device)
#         with torch.no_grad():
#             output = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         return output.last_hidden_state
        
#     def _compute_cosine_similarity(self, a, b):
#         a_np = a.cpu().detach().numpy().reshape(1, -1)
#         b_np = b.cpu().detach().numpy().reshape(1, -1)
#         return cosine_similarity(a_np, b_np)[0, 0]
        
#     def expand(self, query, **kwargs):
#         """
#         Expands the query using a BERT-based similarity approach.
        
#         Expected kwargs:
#             documents_df: A pandas DataFrame with a 'preprocessed_text' column.
#             topRank (int): Number of expansion terms to select (default 4).
#             similarity_threshold (float): Cosine similarity threshold (default 0.6).
#         """
#         documents_df = kwargs.get("documents_df")
#         if documents_df is None:
#             raise ValueError("Bert expander requires 'documents_df' in kwargs")
#         topRank = kwargs.get("topRank", 4)
#         similarity_threshold = kwargs.get("similarity_threshold", 0.6)
        
#         # Obtain the [CLS] embedding for the query.
#         query_embedding = self._get_embedding(query)[0, 0]
        
#         # Build vocabulary from preprocessed documents.
#         vocabulary = set()
#         for doc in documents_df["preprocessed_text"]:
#             vocabulary.update(doc.split())
        
#         candidate_terms = []
#         for term in vocabulary:
#             term_embedding = self._get_embedding(term)[0, 0]
#             sim = self._compute_cosine_similarity(query_embedding, term_embedding)
#             if sim > similarity_threshold:
#                 candidate_terms.append((term, sim))
        
#         # Sort candidates by similarity in descending order.
#         candidate_terms.sort(key=lambda x: x[1], reverse=True)
#         top_terms = [term for term, sim in candidate_terms[:topRank]]
        
#         # Return the expanded query (original plus expansion terms).
#         return query + " " + " ".join(top_terms)
