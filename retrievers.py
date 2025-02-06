
from imports import *

class Retriever:
    @abstractmethod
    def search(self, q):
        pass



class BM25Retriever(Retriever):
    def __init__(self, index_ref, num_results=100):
        self.retriever = pt.BatchRetrieve(index_ref, wmodel="BM25", num_results=num_results)
    def search(self, query):
        return self.retriever.search(query)
    




class DPRRetriever(Retriever):
    def __init__(self,
                 model_name="facebook/dpr-question_encoder-single-nq-base",
                 device=None,
                 num_results=100,
                 index_file="dpr/index.idx",
                 doc_ids_file="dpr/doc_ids.csv"):
    
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRQuestionEncoder.from_pretrained(model_name).to(self.device)
        self.num_results = num_results
        
        # ----------- loading the ann index and doc ids ----------------
        self.index = faiss.read_index(index_file)
        doc_df = pd.read_csv(doc_ids_file)
        self.doc_ids = doc_df["docno"].tolist()
    
    def search(self, query):
        """
        Encodes the query, searches the Faiss index for nearest neighbors, and returns a list of docs and similarity score

        """
      
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            query_embedding = self.model(**inputs).pooler_output  # (1,hidden_dim)

        query_embArray = query_embedding.cpu().numpy() # has to be array for faiss
        faiss.normalize_L2(query_embArray)
        
        # get the nearest neighbors by index
        distances, indices = self.index.search(query_embArray, self.num_results)
        
        # map indices to document IDs and prepare the results list
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            results.append((self.doc_ids[idx], dist))
        return results
    





class ColBERTRetriever(Retriever):
    def __init__(self,
                 model_name="colbert-ir/colbertv2.0",
                 index_file="colbert/index.faiss",
                 embeddings_file="colbert/embeddings.npy",  # (optional, for reference)
                 doc_ids_file="colbert/doc_ids.csv",
                 num_results=10,
                 device=None):
        """
        ColBERT retriever using precomputed document embeddings and a Faiss ANN index.
        
        Args:
            model_name (str): Identifier of the ColBERT model checkpoint.
            index_file (str): Path to the Faiss index file (precomputed).
            embeddings_file (str): Path to the precomputed embeddings file (for reference).
            doc_ids_file (str): CSV file mapping each embedding to a document ID.
            num_results (int): Number of retrieval results to return.
            device (torch.device): Device to run inference (GPU if available, otherwise CPU).
        """
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.num_results = num_results
        
        # Load the precomputed Faiss index.
        self.index = faiss.read_index(index_file)
        
        # Load document IDs from CSV.
        self.doc_ids = pd.read_csv(doc_ids_file)["docno"].tolist()
    
    def _encode(self, text, max_length=32):
        """Tokenize the text and return the encoded inputs."""
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
    
    def _get_embedding(self, text):
        """
        Compute token-level embeddings for the input text and apply max-pooling
        to get a single aggregated embedding.
        """
        inputs = self._encode(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            # Get token-level embeddings: shape (1, seq_length, hidden_dim)
            outputs = self.model(**inputs).last_hidden_state
        # Use max-pooling over tokens (dim=1) to get one vector per document.
        pooled = outputs.max(dim=1)[0]  # shape: (1, hidden_dim)
        return pooled  # Returns a torch tensor
    
    def search(self, query):
        """
        Compute the query embedding, normalize it, and search the Faiss index.
        
        Returns:
            A list of tuples (docno, score) for the top retrieval results.
        """
        # Compute the query's aggregated embedding.
        query_embedding_tensor = self._get_embedding(query)  # shape: (1, hidden_dim)
        query_embedding = query_embedding_tensor.cpu().numpy()  # Convert to NumPy array.
        # Normalize the query embedding so that inner product approximates cosine similarity.
        faiss.normalize_L2(query_embedding)
        
        # Search the precomputed Faiss index.
        distances, indices = self.index.search(query_embedding, self.num_results)
        
        # Map the retrieved indices to document IDs and create a list of (docno, score) tuples.
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            results.append((self.doc_ids[idx], score))
        return results