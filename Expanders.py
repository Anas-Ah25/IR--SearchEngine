from preprocessing import * 

class Expander(ABC):
    @abstractmethod
    def expand(self, query, **kwargs): # abstract method for tools 
        pass

    @staticmethod
    def cosine_similarity_M(a,b):
        ''' manual cosine similarity function, using arrays and torch tensors'''
        try:
            a_np = a.cpu().detach().numpy().reshape(1, -1)
            b_np = b.cpu().detach().numpy().reshape(1, -1)
        except AttributeError:
            a_np = np.array(a).reshape(1, -1)
            b_np = np.array(b).reshape(1, -1)
        return cosine_similarity(a_np, b_np)[0, 0] # sklearn cosine similarity function
# ============================== Expanding Methods ==============================

''' =================---------------------------- Rm3 Expander ----------------------------================= '''
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

''' =================-------------------------- Glove Expander ----------------------------================='''

class Glove(Expander):
    ''' 
    The main approach i usedhere is importing a glove trained model results from "https://nlp.stanford.edu/projects/glove/", and used the 6B.100d version,
    then used it as a base to search for the words from query, compare cosine similarity between the words and the words in the glove model to have an 
    estimation for the semantic meaning of the word and other words, then add this words to the query
    '''
    def __init__(self, glove_file, topK=5, similarity_threshold=0.5):

        self.topK = topK # words will be added as expansion
        self.similarity_threshold = similarity_threshold #  minimum similarity to be added as expansion
        self.embeddings = {}  # Dictionary to map word -> vector
        self.dim = None
        self._load_glove(glove_file) # glove emebedding file (used the 100b version)
    
    def _load_glove(self, glove_file):
        # print("loading from glove file")
        with open(glove_file,'r',encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                if self.dim is None:
                    self.dim = vector.shape[0]
                self.embeddings[word] = vector
        # print(f"loaded {len(self.embeddings)} word vectors")
    
    def expand(self, query, **kwargs):
        # query in normal should be passed preprocessed but in normal string scentence, so we will tokeinze it
        tokens = re.findall(r'\w+', query) # tokenize the query
        expansion_terms = [] # terms will be added
        for token in tokens:
            if token not in self.embeddings: # used small version from glove, so some words may not be found
                continue
            token_vec = self.embeddings[token] # get the embedding vector of the token
            similarities = []
            for word, vec in self.embeddings.items():
                if word == token:
                    continue
                sim = Expander.cosine_similarity_M(token_vec,vec)
                if sim >= self.similarity_threshold:
                    similarities.append((word,sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_terms = [w for w, sim in similarities[:self.topK]] # most similar words
            expansion_terms.extend(top_terms) 
        expansion_terms = list(set(expansion_terms))
        expanded_query = query + " " + " ".join(expansion_terms)
        return expanded_query
# -----------------------------------------------------------------------------------------------------



''' =================-------------------------- Bert Expander ----------------------------================='''


class Bert(Expander):
    def __init__(self, device=None, model_name: str = "prajjwal1/bert-tiny"): 
        '''tried the amazon/bort, distilbert-base-uncased, prajjwal1/bert-tiny'''
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.similarity_threshold = 0.6
        self.topRank = 5

    def _encode(self, text, max_length=32):
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt'
        )

    def _get_embedding(self, text):
        tokens = self._encode(text)
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state

    def expand(self, query, documents_df):
        query_embedding = self._get_embedding(query)[0, 0]  # cls token, representing the whole query
        vocabulary = set()
        for doc in documents_df["preprocessed_text"]:
            vocabulary.update(doc.split())

        candidate_terms = []
        for term in vocabulary:
            term_embedding = self._get_embedding(term)[0, 0]
            sim = self.cosine_similarity_M(query_embedding, term_embedding)
            if sim > self.similarity_threshold:
                candidate_terms.append((term, sim))

        # Sort candidates by similarity in descending order
        candidate_terms.sort(key=lambda x: x[1], reverse=True)
        top_terms = [term for term, sim in candidate_terms[:self.topRank]]

        # Return the expanded query (original plus expansion terms).
        return query + " " + " ".join(top_terms)
