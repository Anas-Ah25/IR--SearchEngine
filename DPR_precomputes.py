from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from preprocessing import *
from retrievers import *

""" In this file i will precompute the values of embeddings for the documents in the dataset using the DPR model manually
ave them using an ANN index (Faiss) to be used directly in search process to speed up the retrieval process, this was made on 500 document from
vaswani dataset, 3 files are exported, embeddings.npy, index.faiss, doc_ids.csv where the embeddings is the matrix of the embeddings of the documents
index.faiss is the index file for the embeddings, and doc_ids.csv is the file containing the document ids.
"""
# --------------------------- data ------------------------------------
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

dataset = pt.datasets.get_dataset('vaswani')
docs = []
count = 0
for item in dataset.get_corpus_iter(verbose=True):
    doc_id = item.get("docno")
    doc_text = item.get("text")
    docs.append((doc_id, doc_text))
    count += 1
    if count == 500:
        break

documentsDf = pd.DataFrame(docs, columns=["docno", "text"])
print("Loaded", len(documentsDf))
preprocessor = Preprocessing()
documentsDf["preprocessed_text"] = documentsDf["text"].apply(lambda x: preprocessor.preprocessing(x))

# ---------------------------------------------------------------------

# Dpr model and it's tokenizer
modell = "facebook/dpr-ctx_encoder-single-nq-base"
tokenizer = DPRContextEncoderTokenizer.from_pretrained(modell)
model = DPRContextEncoder.from_pretrained(modell)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

texts = documentsDf["preprocessed_text"].tolist()
doc_ids = documentsDf["docno"].tolist()
num_texts = len(texts)
batch_size = 32 
embeddings_list = []

print("size", num_texts,"batches:", batch_size,)
for i in range(0, num_texts, batch_size): # this loop is to process the embeddings in batches
    batch_texts = texts[i:i+batch_size] # texts 
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True) # tokeinze
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    with torch.no_grad():
        outputs = model(**inputs).pooler_output  # shape: (batch_size, hidden_dim)
    embeddings_list.append(outputs.cpu().numpy())
    print("Processed", i + len(batch_texts))

embeddings_matrix = np.vstack(embeddings_list) 
print("matrix",embeddings_matrix.shape)

faiss.normalize_L2(embeddings_matrix)
dim = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_matrix)
print("faiss index has", index.ntotal, "vectors")


if not os.path.exists("dpr"):
    os.makedirs("dpr")

embeddings_file = "dpr/embeddings.npy"
index_file = "dpr/index.faiss"
doc_ids_file = "dpr/doc_ids.csv"

np.save(embeddings_file,embeddings_matrix)
faiss.write_index(index, index_file)
pd.DataFrame({"docno": doc_ids}).to_csv(doc_ids_file, index=False)
