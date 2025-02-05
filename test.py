from Expanders import *
from preprocessing import Preprocessing
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
        docs.append((doc_id, doc_text))
        count += 1
        if count == 500:
            break
    documentsDf = pd.DataFrame(docs, columns=["docno", "text"])
    
    preprocessor = Preprocessing()
    documentsDf["preprocessed_text"] = documentsDf["text"].apply(lambda x: preprocessor.preprocessing(x))
    queriesDf = dataset.get_topics()
    qrelsDf = dataset.get_qrels()
    index_dir_python = r"C:\AnasProjects\DatasetIndex"
    if not os.path.exists(index_dir_python):
        os.makedirs(index_dir_python)
    
    indexer = pt.DFIndexer(index_dir_python, overwrite=True)
    index_ref = indexer.index(documentsDf["preprocessed_text"], documentsDf["docno"])
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    rm3 = RM3(bm25, index_ref, preprocessor)
    # glove_file = 'https://nlp.stanford.edu/data/glove.6B.zip'
    glove_file = 'glove.6B.100d.txt'
    glove_expander = Glove(glove_file, topK=3, similarity_threshold=0.6)
    bert_expander = Bert()



    while True:
        query = input("Enter your query: ")
        method = input("Enter the method you want to use (RM3, Glove, or Bert): ")
        
        if method == "rm3":
            expanded_query = rm3.expand(query)
        elif method == "glove":
            expanded_query = glove_expander.expand(query)
        elif method == "bert":
            expanded_query = bert_expander.expand(query, documents_df=documentsDf)
        else:
            print("Invalid method. Please choose RM3, Glove, or Bert.")
            continue
        
        print("Original query:", query)
        print("Expanded query:", expanded_query)
        print("=====================================")
        if input("Do you want to continue? (y/n): ").lower() == "n":
            break
