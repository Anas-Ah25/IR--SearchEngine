
from imports import *




class Reranker(ABC):
    @abstractmethod
    def rerank(self, query, doc_df):
        pass


class SpladeReranker(Reranker):
    def __init__(self, model_name="naver/splade-cocondenser-ensembledistil", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rerank(self, query, doc_df):
        scores = []
        for _, row in doc_df.iterrows():
            text = row["preprocessed_text"] 
            # "Cross-encode" query+doc
            inputs = self.tokenizer.encode_plus(
                query, text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding="max_length"
            )
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
                score = outputs.logits[0, 1].item()  # if 2-dim, picking the "positive" class
            scores.append(score)
        doc_df = doc_df.copy()
        doc_df["score"] = scores
        # Sort descending
        doc_df = doc_df.sort_values("score", ascending=False)
        return doc_df
    


class BertCrossEncoderReranker(Reranker):
    def __init__(self, model_name="nboost/pt-bert-base-uncased-msmarco", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rerank(self, query, doc_df):
        scores = []
        for _, row in doc_df.iterrows():
            doc_text = row["preprocessed_text"]  # or row["text"]
            inputs = self.tokenizer.encode_plus(
                query, doc_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding="max_length"
            )
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
                # If it's a single score: [batch,1], just do:
                score = outputs.logits[0].item()
            scores.append(score)
        doc_df = doc_df.copy()
        doc_df["score"] = scores
        doc_df = doc_df.sort_values("score", ascending=False)
        return doc_df