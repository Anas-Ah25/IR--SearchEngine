from Expanders import * 


if __name__ == "__main__":
    import pyterrier as pt
    if not pt.started():
       pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    # Load a dataset from PyTerrier (for testing)
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
    
    # Preprocess documents and add a 'preprocessed_text' column
    preprocessor = Preprocessing()
    documentsDf["preprocessed_text"] = documentsDf["text"].apply(lambda x: preprocessor.preprocessing(x))
    
    # (Optional) Get topics and qrels for completeness
    queriesDf = dataset.get_topics()
    qrelsDf = dataset.get_qrels()
    
    # Use an index directory that avoids special characters
    index_dir_python = r"C:\AnasProjects\DatasetIndex"  # Adjust this path as needed.
    if not os.path.exists(index_dir_python):
        os.makedirs(index_dir_python)
    
    # Create an index using the preprocessed texts (pass a Python string, not a Java String)
    indexer = pt.DFIndexer(index_dir_python, overwrite=True)
    index_ref = indexer.index(documentsDf["preprocessed_text"], documentsDf["docno"])
    
    # Create a BM25 retriever using the index
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    
    # Instantiate the RM3 expander
    rm3 = RM3(bm25, index_ref, preprocessor)
    
    # Test query expansion
    query = "What is the capital of France?"
    expanded_query = rm3.expand(query)
    print("Expanded query:", expanded_query)
   