import pandas as pd
import nltk
import os

nltk.download('punkt_tab')

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
    for index, row in df.iterrows():
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


def main():
    folder_path = r'..\cw-pack-2026\cw-pack-2026\texts\novels'
    df = read_novels(folder_path)
    ttr_dict = nltk_ttr(df)
    print(ttr_dict)

if __name__ == '__main__':
    main()