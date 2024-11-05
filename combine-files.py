
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


full = './Data/2015/full/div-b'
national_full = pd.read_csv(full + '/2015 DivB National Full.csv')
pt_full = pd.read_csv(full + '/2015 DivB PT Full.csv')


pt_full.head()


def data_preprocessing(column):
    column = column.str.replace('', ' ') \
                    .str.strip() \
                    .str.replace(r'^[\da-zA-Z]+\)', ' ', regex=True) \
                    .str.replace('\n', ' ') \
                    .mask((column == '-') | (column == '_')) \
                    .str.replace('\x93', ' ') \
                    .str.replace('\x94', ' ') \
                    .str.replace('\x96', ' ') \
                    .str.replace('\x97', ' ') \
                    .str.replace(r' [\da-zA-Z]+\) ', ' ', regex=True) \
                    .str.replace(r'\s{2,}', ' ', regex=True) \
                    .str.replace('?', ' ') \
                    .str.replace(r'^\d+(\.\d+)*[A-Za-z]*\.?\s*', '', regex=True) \
                    .str.strip()              
    return column


provinces_and_territories = ['AB', 'BC', 'NS', 'NU', 'ON', 'PE', 'SK']
code_books = {'B': 'Building', 'P': 'Plumbing', 'F': 'Fire', 'E': 'Energy'}

output_path = './split-up'

national_2015_df = pd.read_excel('./Data/2015/full/National Codes 2015 sentences.xlsx')

national_2015_df['FRAG_DOCUMENT'] = data_preprocessing(national_2015_df['FRAG_DOCUMENT'])
national_2015_df['ARTICLE_TITLE'] = data_preprocessing(national_2015_df['ARTICLE_TITLE'])

national_2015_df_B = national_2015_df[national_2015_df['DIVISION'] == 'B']

national_2015_df_B = national_2015_df_B.copy()
national_2015_df_B.dropna(subset=['FRAG_DOCUMENT'], inplace=True)

print("Number of text sentences in")    
print(f'National Codes: {national_2015_df_B.shape[0]}')

national_2015_df_B.to_csv('./Data/2015/full/div-b/2015 DivB National Full new.csv', index=True, index_label='Full Index')

def split_files(national=True):

    def save_file(df, pt, national):
        if not national:
            df['Full Index'] = df.index

        for book, code_type in code_books.items():
            book_df = df[df['DOCTYPE'] == book]
            book_df.to_csv(f'./{output_path}/{code_type.lower()}/2015 {pt} {code_type}.csv', index=False)
        
    if national:
        df = pd.read_csv(full + '/2015 DivB National Full new.csv')
        save_file(df, 'National', national)
    else:
        pt_full = pd.read_csv(full + '/2015 DivB PT Full.csv')
        for pt in provinces_and_territories:
            df = pt_full[pt_full['PT'] == pt]
            save_file(df, pt, national)


split_files(national=False)
split_files(national=True)


# combine OBC and OPC and call it OCC
# combine NBC, NEC and NPC and call it NCC

def combine_files():
    obc = pd.read_csv('./split-up/building/2015 ON Building.csv')
    opc = pd.read_csv('./split-up/plumbing/2015 ON Plumbing.csv')

    # Combine OBC and OPC
    occ = pd.concat([obc, opc])
    print(occ.shape)

    # Remove any rows with duplicate 'PT Sentence Text' and 'PT Sentence Number' values
    occ1 = occ.drop_duplicates(subset=['PT Sentence Text', 'PT Sentence Number'])
    occ1.to_csv('./split-up/combined/2015 ON Combined.csv', index=False)
    print(occ1.shape)

    # Combine NBC, NEC and NPC
    nbc = pd.read_csv('./split-up/building/2015 National Building.csv')
    nebc = pd.read_csv('./split-up/energy/2015 National Energy.csv')
    npc = pd.read_csv('./split-up/plumbing/2015 National Plumbing.csv')
    print(nbc.shape, nebc.shape, npc.shape)

    ncc = pd.concat([nbc, nebc, npc])

    ncc.to_csv('./split-up/combined/2015 National Combined.csv', index=False)
    print(ncc.shape)

combine_files()


