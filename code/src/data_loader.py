import pandas as pd
import os
import numpy as np

def load_and_clean_data(input_file_path, output_filename="Factiva_v1_noDup.csv"):
    """
    Loads the raw Excel export from input_file_path, renames columns, cleans body text, removes duplicates, 
    and saves to CSV in the same directory as the input file.
    """
    print(f"Loading data from {input_file_path}...")
    data_path = os.path.dirname(input_file_path)
    
    # read the excel file containing all articles from the factiva export
    raw_export_C = pd.read_excel(input_file_path) 
    
    # convert all Chinese columns names to English
    raw_export_E = raw_export_C.rename(columns={
        '排序号': 'index',
        '文档id': 'doc_id',
        '出版社': 'publisher',
        '标题': 'title',
        '日期': 'date',
        '时间': 'time',
        '作者': 'author',
        '字数': 'word_count',
        '语言': 'language',
        '公司': 'company',
        '行业': 'industry',
        '主题': 'subject',
        '地区': 'region',
        '所在版面': 'page',
        'ART': 'ART',
        '摘要': 'abstract',
        '全文_1': 'body_1',
        '全文_2': 'body_2',
        '全文_3': 'body_3',
        '全文_4': 'body_4',
        '全文_5': 'body_5',
        '全文_6': 'body_6',
        '全文_7': 'body_7',
        '全文_8': 'body_8',
        '全文_9': 'body_9',
        '全文_10': 'body_10',
        '全文_11': 'body_11',
        '全文_12':  'body_12',
        '全文_13':  'body_13'
    })

    # then drop the unnecessary columns
    refined_export = raw_export_E.drop(columns=['index', 'doc_id', 'time', 'author', 'language', 'company', 'industry',
                                  'subject', 'region', 'page', 'ART', 'abstract'])

    print(f"Refined shape: {refined_export.shape}")

    # concatenate all body columns into a single body column
    refined_export['body'] = refined_export[['body_1', 'body_2', 'body_3', 'body_4', 'body_5',
                                             'body_6', 'body_7', 'body_8', 'body_9', 'body_10',
                                             'body_11', 'body_12', 'body_13']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    # everything looks good. Now drop the individual body part columns
    final_export = refined_export.drop(columns=['body_1', 'body_2', 'body_3', 'body_4', 'body_5',
                                                'body_6', 'body_7', 'body_8', 'body_9', 'body_10',
                                                'body_11', 'body_12', 'body_13'])

    print(f"Final export shape: {final_export.shape}")

    # drop duplicates: if they have the same title and that the first 50 characters of the body text are the same
    final_export.loc[:, "primary_body"] = final_export['body'].str[:50]
    final_export_noDup = final_export.drop_duplicates(subset=['title', 'primary_body']).copy()

    print(f"Shape after removing duplicates: {final_export_noDup.shape}")

    # print how many rows were deleted due to duplication
    num_duplicates = final_export.shape[0] - final_export_noDup.shape[0]
    print(f"Number of duplicate rows removed: {num_duplicates}")

    # small clean: delete the word "word" from the word_count column and convert to integer
    final_export_noDup['word_count'] = final_export_noDup['word_count'].astype(str).str.replace(' words', '').astype(int)

    # small clean: convert the date column to datetime format
    final_export_noDup['date'] = pd.to_datetime(final_export_noDup['date'], format='%Y/%m/%d')

    # save the cleaned dataframe to a new csv file
    final_export_noDup.reset_index(drop = True, inplace = True)
    
    output_path = os.path.join(data_path, output_filename)
    final_export_noDup.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")
    
    return final_export_noDup, final_export.shape[0], num_duplicates

def load_cleaned_data(data_path, filename="Factiva_v1_noDup.csv"):
    """
    Loads the already cleaned CSV file.
    """
    file_path = os.path.join(data_path, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please run load_and_clean_data first.")
    
    print(f"Loading cleaned data from {file_path}...")
    df = pd.read_csv(file_path, parse_dates=['date'])
    return df
