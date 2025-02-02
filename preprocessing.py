from imports import * # all imports 


class Preprocessing:
    """Preprocessing query, exclude stop words, stemming, and make correction for spelling"""
    
    def __init__(self):
        # stemmer and stop words list
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    '''--------------------------- Main ---------------------------'''
    def preprocessing(self, scentence):
        tokens = word_tokenize(scentence) # tokenize the sentence 
        preprocessed_tokens = [] 
        
        for token in tokens:
            # charachters cleaning
            cleaned_token = re.sub(r"[^a-zA-Z0-9 ]", "", token)
            cleaned_token = re.sub(r"\?\s", " ", cleaned_token)
            if cleaned_token in self.stop_words:
                continue
            # stem and lower-case
            token_stem = self.stemmer.stem(cleaned_token).lower()
            if token_stem != '?' and len(token_stem) > 1 and not token_stem.isdigit():
                preprocessed_tokens.append(token_stem)
                
        return ' '.join(preprocessed_tokens)
    '''----------------------------------------------------------------'''


    
    '''--------------------------- SymSpell ---------------------------'''
    def initialize_symspell(self) -> SymSpell:
        ''' Initialize the SymSpell spell checker, used to correct spelling of words'''
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt") # directly access the file / can be accessed from the repo or downloaded locally
        max_edit_distance = 2
        prefix_length = 7
        sym_spell = SymSpell(max_edit_distance, prefix_length)
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        return sym_spell

    def correct_spelling(self, text, cleaner): # cleaner -->  symspell
        """
        Correct spelling of a text using SymSpell, tool used to correct spelling is passed through the parameter, so can be changed
        """
        suggestions = cleaner.lookup_compound(text, max_edit_distance=2)  # multiple words
        if suggestions:
            return suggestions[0].term  
        return text
    '''----------------------------------------------------------------'''

# # test

# preprocessor = Preprocessing() 

# sym_spell = preprocessor.initialize_symspell()

# input_text = "Ths is a smple txt with speling erors."
# expected_output = "This is a simple text with spelling errors."
# preprcoess =  preprocessor.preprocessing(input_text)
# corrected = preprocessor.correct_spelling(input_text, sym_spell)
# finaltxt =  preprocessor.preprocessing(corrected)

# print(finaltxt)