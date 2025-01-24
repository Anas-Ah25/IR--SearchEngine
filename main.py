from flask import Flask, render_template,request
import pandas as pd
import pyterrier as pt
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

app = Flask(__name__)


def preprocessing(row): # manual function to preprocess the data
  stemmer = PorterStemmer()
  stopWords = set(stopwords.words('english'))
  tokens = word_tokenize(row)
  preprocessed = []
  for token in tokens:
    # Ntoken = re.sub(r'[^a-zA-Z0-9\s\?]',' ',token)
    Ntoken = re.sub(r"[^a-zA-Z0-9 ]", "", token)
    Ntoken = re.sub(r"\?\s", " ", token)
    preprocessed.append((stemmer.stem(Ntoken)).lower())
    preprocessed = [token for token in preprocessed if token != '?' and token not in stopWords and len(token) != 1 and not (token.isdigit())]
  return ' '.join(preprocessed)

# ---------------- load the data ----------------
'''  This data is a built in dataset from pyterrier library,
 and the loading of data has main objective of just define documents, qrels and queries as dataframes, 
 before this i run the code on manual acquired data from xml file, 
 so it's just the same process but with different data source.'''


dataset = pt.datasets.get_dataset('vaswani') # using vaswani dataset
docs = []
count = 0 
for item in dataset.get_corpus_iter(verbose=True):
    doc_id = item.get("docno")
    doc_text = item.get("text")
    docs.append((doc_id, doc_text))
    count += 1
    if count == 500: # here i used only 500 documents to make the process faster
        break
documentsDf = pd.DataFrame(docs, columns=["docno","text"])
queriesDf = dataset.get_topics()
qrelsDf = dataset.get_qrels()
# ----------------------------------------------------------

 # ---------------- preprocess current data ----------------
documentsDf['preprocessed_text'] = documentsDf['text'].apply(preprocessing) 
queriesDf['Query'] = queriesDf['query'].apply(preprocessing)
# ----------------------------------------------------------

#---------------- indexing the data  ----------------
indexer = pt.DFIndexer('C:/Users/ne3na/OneDrive/Desktop/projectMining/Project folder/Dfindex3',overwrite=True)
index_ref = indexer.index(documentsDf['preprocessed_text'],documentsDf['docno'])
index = pt.IndexFactory.of(index_ref)
# ----------------
bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=10) # using BM25 model to retrieve the results

def extract_expanded_query(x): # take the outputed format of expanded query and filter it to make new scentence as normal
    terms = re.findall(r'(\w+)\^\d+\.\d+',x) 
    expanded_query = ' '.join(terms)
    return expanded_query

def process_query(query_text, index, documentsDf):
    processed_query = preprocessing(query_text)
    print(f"Processed Query: {processed_query}")
    
    expander = pt.rewrite.RM3(index, fb_terms=10, fb_docs=100)
    ret_expanded = bm25 >> expander  # expanding the query using RM3
    
    # Perform the search and check for results
    expanded_results = ret_expanded.search(processed_query)
    if expanded_results.empty:  # No results after expansion
        return {'results': [], 'expanded_query': 'No expanded query could be generated.'}
    
    expanded_query_data = expanded_results.iloc[0]["query"]  # Retrieve expanded query
    expanded_query_as_text = extract_expanded_query(expanded_query_data)
    
    # Search with the expanded query
    final_results = bm25.search(expanded_query_as_text)
    if final_results.empty:  # No results for the expanded query
        return {'results': [], 'expanded_query': expanded_query_as_text}

    # Process results
    results = []
    for _, row in final_results.iterrows():
        docno = row['docno']
        document_text = documentsDf.loc[documentsDf['docno'] == docno, 'text'].iloc[0]
        score = row['score']  # Capture the relevance score
        results.append({'docno': docno, 'document_text': document_text, 'score': score})  # Include the score

    return {'results': results, 'expanded_query': expanded_query_as_text}


@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query_text = request.form['query']
        results_data = process_query(query_text, index, documentsDf)
        # Check if there are no results and pass a message
        if not results_data['results']:
            return render_template(
                './page.html',
                results=None,
                expanded_query=results_data['expanded_query'],
                message="No documents found for the entered query."
            )
        return render_template(
            './page.html',
            results=results_data['results'],
            expanded_query=results_data['expanded_query'],
            message=None
        )
    return render_template('./page.html', results=None, message=None)

if __name__ == '__main__':
    app.run(debug=True)