import pandas as pd 


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

def main():
    file_path = r'..\cw-pack-2026\cw-pack-2026\texts\hansard500.csv'
    df = read_data(file_path)


if __name__ == '__main__':
    main()