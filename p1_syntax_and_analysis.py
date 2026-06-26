import pandas as pd
import nltk
import os
import math
import spacy
from spacy.symbols import nsubj, VERB
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
nltk.download('punkt_tab')
nltk.download('cmudict')

def read_novels(folder_path):
    # create a pandas dataframe with the cols:  text, title, author, year
    df = pd.DataFrame(columns=['text', 'title', 'author', 'year'])
    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.txt'):
            continue
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        # extract title, author, and year from file name
        file_name = file_name[:-4]  # remove .txt extension
        title, author, year = file_name.split('-')
        # create a temporary dataframe with the extracted information
        temp_df = pd.DataFrame({'text': [text], 'title': [title], 'author': [author], 'year': [year]})
        # concatenate the temporary dataframe with the main dataframe
        df = pd.concat([df, temp_df], ignore_index=True)
    df.sort_values(by='year', inplace=True)
    return df

def nltk_ttr(df):
    # create a dictionary mapping title to its type-token ratio
    # use NLTK library only 
    # do not include punctuation and ignore case when counting types
    ttr_dict = {}
    for _, row in df.iterrows():
        text = row['text']
        # tokenize the text using NLTK
        tokens = nltk.word_tokenize(text)
        # filter out punctuation and convert to lower case
        tokens = [token.lower() for token in tokens if token.isalpha()]
        # calculate type-token ratio
        types = set(tokens)
        ttr = len(types) / len(tokens) if len(tokens) > 0 else 0
        ttr_dict[row['title']] = ttr
    return ttr_dict

def flesch_kincaid(df):
    # create a dictionary mapping title to its Flesch-Kincaid reading ease score
    # use NLTK library for tokenization and CMU pronouncing dictionary for syllable counting
    # do not include punctuation and ignore case when counting syllables
    fk_dict = {}
    cmu = nltk.corpus.cmudict.dict()
    for _, row in df.iterrows():
        text = row['text']
        # tokenize the text using NLTK
        tokens = nltk.word_tokenize(text)
        # filter out punctuation and convert to lower case
        tokens = [token.lower() for token in tokens if token.isalpha()]
        # count syllables using CMU pronouncing dictionary
        syllable_count = 0
        for token in tokens:
            syllables = cmu.get(token)
            if syllables:
                syllable_count += min(len(syllable) for syllable in syllables)

        # calculate Flesch-Kincaid reading ease score
        words = len(tokens)
        sentences = len(nltk.sent_tokenize(text))
        fk_score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllable_count / words) if sentences > 0 and words > 0 else 0
        fk_dict[row['title']] = fk_score

    return fk_dict

def parse(df, pickle_file='parsed_novels.pkl'):
    # add a new dataframe column called 'parsed' that contains the parsed and tokenized Doc object for each text using spaCy nlp method
    # serialise the resulting df using the pickle format
    # return the df 
    # load the df from the pickle file and use it for the remainder of the coursework part
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])
    df['parsed'] =df['text'].apply(nlp)
    df.to_pickle(pickle_file)
    return df

def calculate_pmi(subj_freq_count : int, verb_freq : nltk.FreqDist, subj_verb_freq : nltk.FreqDist, subj_verb_list : list, total_count : int):
    """
    Calculate Pointwise Mutual Information (PMI) for a word pair.

    Parameters:
        subj_freq_count (int): Frequency count of the subject.  
        verb_freq (nltk.FreqDist): Frequency distribution of verbs.  
        subj_verb_freq (nltk.FreqDist): Frequency distribution of subject-verb pairs
        subj_verb_list (list): List of verbs associated with the subject.  
        total_count (int): Total number of tokens in the document.
    """
    # Calculate Pointwise Mutual Information (PMI) for a word pair
    pmi = {}

    # Use unique verb strings so repeated token objects don't appear multiple times
    verbs = subj_verb_list
    for verb in verbs:
        verb_freq_count = verb_freq.freq(verb)
        # Calculate PMI for this word pair
        p_word1 = subj_freq_count / total_count
        p_word2 = verb_freq_count / total_count
        
        subj_verb_freq_count = subj_verb_freq.freq(verb)
        p_word_pair = subj_verb_freq_count / total_count

        if p_word1 > 0 and p_word2 > 0 and p_word_pair > 0:
            pmi[verb] = math.log(p_word_pair / (p_word1 * p_word2), 2)
        else:
            pmi[verb] = 0
    # order the verbs by PMI score in descending order
    pmi = dict(sorted(pmi.items(), key=lambda item: item[1], reverse=True))
    for verb, score in pmi.items():
        print(f"Verb: {verb}, PMI: {score}")
    return pmi

def extract_features(df):
    # title of each novel, and the 10 most common syntatic subjects
    # title of each novel and a list of verbs most likely to occur with the subject 'he' ordered by Pointwise Mutual Information (PMI) score
    # title of each novel and a list of verbs most likely to occur with the subject 'she' ordered by Pointwise Mutual Information (PMI) score
    for _, row in df.iterrows():
        title = row['title'] 
        parsed_doc = row['parsed']
        # extract the 10 most common syntactic subjects
        # lemma is more linguistically meaningful than the token text, so we will use lemma for the subject
        subjects = [token.lemma_.lower() for token in parsed_doc if token.dep_ == 'nsubj']
        verbs = [token.lemma_.lower() for token in parsed_doc if token.pos_ == 'VERB']
        subject_freq = nltk.FreqDist(subjects)
        verb_freq = nltk.FreqDist(verbs)
        most_common_subjects = subject_freq.most_common(10)
        print(f"Title: {title}")
        print("Most common syntactic subjects:", most_common_subjects)

        # extract verbs most likely to occur with the subject 'he' and 'she' ordered by PMI score
        he_freq = subject_freq.freq('he')
        she_freq = subject_freq.freq('she')

        he_verbs = []
        she_verbs = []
        for possible_subject in parsed_doc:
            if possible_subject.dep == nsubj and possible_subject.lemma_.lower() == 'he' and possible_subject.head.pos == VERB:
                he_verbs.append(possible_subject.head.lemma_.lower())
            elif possible_subject.dep == nsubj and possible_subject.lemma_.lower() == 'she' and possible_subject.head.pos == VERB:
                she_verbs.append(possible_subject.head.lemma_.lower())
        
        # calculate PMI score for each verb
        he_verb_freq = nltk.FreqDist(he_verbs)
        she_verb_freq = nltk.FreqDist(she_verbs)
        
        print(f"PMI scores for subject 'he'")
        he_verb_pmi = calculate_pmi(he_freq, verb_freq, he_verb_freq, set(he_verbs), len(parsed_doc))
        
        print(f"PMI scores for subject 'she':")
        she_verb_pmi = calculate_pmi(she_freq, verb_freq, she_verb_freq, set(she_verbs), len(parsed_doc))

def main():
    folder_path = 'cw-pack-2026/texts/novels'
    df = read_novels(folder_path)
    df = parse(df)
    df = pd.read_pickle('parsed_novels.pkl')
    ttr_dict = nltk_ttr(df)
    # print(ttr_dict)
    fk_dict = flesch_kincaid(df)
    extract_features(df)
    # print(fk_dict)

if __name__ == '__main__':
    main()