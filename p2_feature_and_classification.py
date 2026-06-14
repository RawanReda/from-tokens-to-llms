import pandas as pd 
import numpy as np

# sklearn imports for TF-IDF and classifiers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report


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

def vectorize_text_and_train_models(df):
    """
    Vectorize the speeches using TF-IDF vectorizer and train Random Forest and SVM classifiers on the training set.

    Args:
        df (pd.DataFrame): The dataframe containing the speeches and party labels.

    Returns:
        None
    """
    # vectorize the speeches using TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']

    random_state = 26
    # split the data into train and test data using stratified sampling with random seed of 26
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    
    # train randomforest with n_estimators=300 and SVM with linear kernel classifiers on the training set 
    random_forest_model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    random_forest_model.fit(X_train, y_train)
    y_pred_rf = random_forest_model.predict(X_test)
    random_forest_f1_score = f1_score(y_test, y_pred_rf, average='macro', zero_division=0)
    classification_report_rf = classification_report(y_test, y_pred_rf, zero_division=0)
    print("Random Forest Macro F1 Score: ", random_forest_f1_score)
    print("Random Forest Classification Report: \n", classification_report_rf)

    svm_model = SVC(kernel='linear', random_state=random_state)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    svm_f1_score = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
    classification_report_svm = classification_report(y_test, y_pred_svm, zero_division=0)
    print("SVM Macro F1 Score: ", svm_f1_score)
    print("SVM Classification Report: \n", classification_report_svm)


def main():
    file_path = r'..\cw-pack-2026\cw-pack-2026\texts\hansard500.csv'
    df = read_data(file_path)
    vectorize_text_and_train_models(df)


if __name__ == '__main__':
    main()