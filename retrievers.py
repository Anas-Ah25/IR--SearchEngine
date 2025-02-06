
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
                 index_file="dpr_faiss_index.idx",
                 doc_ids_file="doc_ids.csv"):
    
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