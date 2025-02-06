from Expanders import *
from retrievers import BM25Retriever, DPRRetriever, ColBERTRetriever
from preprocessing import *
from imports import *

if __name__ == "__main__":
    if not pt.started():
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

    dataset = pt.datasets.get_dataset('vaswani')
    docs = []
    count = 0
    for item in dataset.get_corpus_iter(verbose=True):
        doc_id = item.get("docno")
        doc_text = item.get("text")
        docs.append((doc_id,doc_text))
        count += 1
        if count == 500:
            break
    documentsDf = pd.DataFrame(docs, columns=["docno", "text"])
    preprocessor = Preprocessing()
    documentsDf["preprocessed_text"] = documentsDf["text"].apply(lambda x: preprocessor.preprocessing(x))
    queriesDf = dataset.get_topics()
    qrelsDf = dataset.get_qrels()
    
    #  --------------- BM25 indexing ----------------
    index_dir_python = r"C:\AnasProjects\DatasetIndex"
    if not os.path.exists(index_dir_python):
        os.makedirs(index_dir_python)
    indexer = pt.DFIndexer(index_dir_python, overwrite=True)
    index_ref = indexer.index(documentsDf["preprocessed_text"], documentsDf["docno"])

    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", num_results=100) # internal retreiver object corrcupt, so pass directly 
    
    
    # ----------- Expander objects ---------------
    rm3 = RM3(bm25, index_ref, preprocessor)
    glove_file = 'glove.6B.100d.txt' 
    glove_expander = Glove(glove_file, topK=3, similarity_threshold=0.6)
    bert_expander = Bert()
    
    # ---------- dpr ----------------
    dpr = DPRRetriever(model_name="facebook/dpr-question_encoder-single-nq-base",num_results=100,
                                  index_file="dpr_faiss_index.idx",doc_ids_file="doc_ids.csv")
    
    colbert_retriever = ColBERTRetriever(
        model_name="colbert-ir/colbertv2.0",
        index_file="colbert/index.faiss",
        doc_ids_file="colbert/doc_ids.csv",
        num_results=10
    )
    


    while True:
        query = input("Enter your query: ")
        retrieval_method = input("Enter the retrieval method you want to use (BM25, DPR, ColBERT): ")
        expander_method = input("Enter the expander method you want to use (RM3, Glove, Bert, or none): ")
        
        final_query = query  
        final_query = preprocessor.preprocessing(final_query)
        
        if expander_method.lower() == "rm3":
            final_query = rm3.expand(query)
        elif expander_method.lower() == "glove":
            final_query = glove_expander.expand(query)
        elif expander_method.lower() == "bert":
            final_query = bert_expander.expand(query, documents_df=documentsDf)
        
        if retrieval_method.lower() == "bm25":
            results = bm25.search(final_query)
            print("BM25 results:")
            print(results.head()) 
        elif retrieval_method.lower() == "dpr":
            results = dpr.search(final_query)
            print("DPR results (docno, score):")
            print(results)
        elif retrieval_method.lower() == "colbert":
            results = colbert_retriever.search(final_query)
            print("ColBERT results (docno, score):")
            print(results)
        else:
            print("Invalid retrieval method. Please choose BM25, DPR, or ColBERT.")
            continue
        
        print("Final query used:", final_query)
        print("=====================================")
        if input("Do you want to continue? (y/n): ").lower() == "n":
            break
