from imports import *
from preprocessing import Preprocessing

"""
In this file we precompute document embeddings for the Vaswani dataset using a ColBERT-style model,  from https://huggingface.co/colbert-ir/colbertv2.0
but the real ColBERT implementation uses token-level embeddings with late interaction, 
here i compute token-level embeddings for each document and then use max-pooling
across tokens to form a single vector  document, all of this stored in a Faiss index to be used in the retrieval process with no computation each time
"""

# ---------------   data  -------------------
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
print("Loaded", len(documentsDf), "documents.")

# ---------------   preprocessing  -------------------
# colbert in normal is working with tsv files, but here i will use the preprocessed text directly
preprocessor = Preprocessing()
documentsDf["preprocessed_text"] = documentsDf["text"].apply(lambda x: preprocessor.preprocessing(x))


# -----------  ColBERT -----------

model_name = "colbert-ir/colbertv2.0"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# compute ColBERT embeddings for each document in batches
texts = documentsDf["preprocessed_text"].tolist()
doc_ids = documentsDf["docno"].tolist()
num_texts = len(texts)
batch_size = 32
embeddings_list = []

for i in range(0, num_texts, batch_size):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        #  token level embeddings, shape (batch_size, seq_length, hidden_dim)
        outputs = model(**inputs).last_hidden_state
    # max-pooling across the token dimension (dim=1) to get one vector per document
    pooled = outputs.max(dim=1)[0] # shape: (batch_size, hidden_dim)
    embeddings_list.append(pooled.cpu().numpy())
    # print("processed")

embeddings_matrix = np.vstack(embeddings_list)
print("Embeddings matrix shape:", embeddings_matrix.shape)

# ---------------------------
# ANN, Faiss index which has the embeddings, where the embedding is stored with 
# ---------------------------
# normalize the embeddings
faiss.normalize_L2(embeddings_matrix)
dim = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_matrix)
# print("index built with", index.ntotal,"vectors")




# --------- outputs ---------
output_dir = "colbert"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

embeddings_file = os.path.join(output_dir, "embeddings.npy") # embeddings matrix
index_file = os.path.join(output_dir, "index.faiss") # faiss index
doc_ids_file = os.path.join(output_dir, "doc_ids.csv") # docnos

np.save(embeddings_file, embeddings_matrix)
faiss.write_index(index, index_file)
pd.DataFrame({"docno": doc_ids}).to_csv(doc_ids_file, index=False)

print("Precomputed ColBERT embeddings saved to", embeddings_file)
print("Faiss index saved to", index_file)
print("Document IDs saved to", doc_ids_file)
