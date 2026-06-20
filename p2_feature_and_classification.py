import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

def read_data(file_path):
    """
    Read the CSV file and return a pandas dataframe with the following modifications:
    - Rename 'Labour (Co-op)' to 'Labour' in the 'party' column
    - Remove any rows where the value of the 'party' column is not in the 4 most common party names
    - Remove the 'speakername' column   
    - Remove any rows where the text in the 'speech' column is less than 1000 characters long
    - Print the shape of the dataframe after filtering

    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The modified dataframe.
    """
    df = pd.read_csv(file_path)
    # rename 'Labour (Co-op)' to 'Labour' in the 'party' column
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    # remove any rows where the value of the 'party' column is not in the 4 most common party names 
    most_common_parties = df['party'].value_counts().nlargest(4).index.tolist()
    df = df[df['party'].isin(most_common_parties)]
    # remove the speaker value
    df = df.drop(columns=['speakername'])
    # remove any rows where the text in the speech col is less than 1000 chars long
    df = df[df['speech'].str.len() >= 1000]

    print("Dataframe shape after filtering: ", df.shape)
    return df

def vectorize_text_and_train_models(df, stop_words=None, custom_tokenizer=None):
    """
    Vectorize the speeches using TF-IDF vectorizer and train Random Forest and SVM classifiers on the training set.

    Args:
        df (pd.DataFrame): The dataframe containing the speeches and party labels.

    Returns:
        None
    """
    # vectorize the speeches using TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=3000, tokenizer=custom_tokenizer)
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']

    random_state = 26
    # split the data into train and test data using stratified sampling with random seed of 26
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    
    def fit_and_report(model, model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)
        print(f"{model_name} Macro F1 Score: ", f1)
        print(f"{model_name} Classification Report: \n", report)
        return model

    random_forest_model = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight='balanced')
    fit_and_report(random_forest_model, "Random Forest")

    svm_model = SVC(kernel='linear', random_state=random_state, class_weight='balanced')
    fit_and_report(svm_model, "SVM")

def custom_tokenizer(text):
    '''
    (e)
    The text goes through some preprocessing steps. The text is converted to lowercase and puctauation is removed. 

    Stop words are removed within the cutom tokenizer, and the stop_words parameter in tf-idf is set to None.
    This is because TF-IDF uses its own stop word removal mechanism, which may interfere with the custom tokenizer's
    preprocessing and generate warmings indicating inconsistencies between stop-word list and custom
    preprocessing pipeline. 

    Before obtain lemma, pos tags are obtained for each token. This helps the lemmatizer determine the correct base form of 
    each word, leading to more accurate normatlization a better understanding of the text. 
    '''
    def penn_to_wordnet(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV  # adverb
        else:
            return wordnet.NOUN # sensible fallback

    # tokenize the text using NLTK
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]

    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()
    def lemmatize_with_pos(tokens):
        tagged = nltk.pos_tag(tokens)
        lemmas = []
        for word, penn in tagged:
            wn_pos = penn_to_wordnet(penn)
            if word not in stop_words:
                lemmas.append(lemmatizer.lemmatize(word, wn_pos))
        return lemmas, tagged

    lemmas, _ = lemmatize_with_pos(tokens)
    
    return lemmas


# think of a good customer tokenizer and compare results 
def main():
    file_path = r'..\cw-pack-2026\cw-pack-2026\texts\hansard500.csv'
    df = read_data(file_path)

    print("Results without custom_tokenizer:")
    vectorize_text_and_train_models(df, stop_words="english")

    print("Results with custom_tokenizer:")
    vectorize_text_and_train_models(df, stop_words=None, custom_tokenizer=custom_tokenizer)

    # based on the output, custom_tokenizer improves performance of random forest model but has no effect on SVC classifier 
    # for the random forest model, the macro f1 score without custom tokenizer is 34%, while with custom tokenizer, it is increased to 42%
    # precision scores for conservative and labour parties is higher when custom tokenizer is applied.
    # scottish national is never correctly predicted likey due to the small training instance of this class. 
if __name__ == '__main__':
    main()