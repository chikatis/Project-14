
# ## Import the necessary libraries


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# import dask.dataframe as dd


# #### Declare the Variables and Parameters


SEED = 42


# ## Data Preprocessing (Variations data)


# #### Read the data


# Paths to the Variations in P/T and National Codes
variation_folder_path = './Data/2015/variations/old-data'

alberta_building = variation_folder_path + '/CCT Codes Comparison Import Alberta Building v2.csv'
bc_buildingA = variation_folder_path + '/CCT Codes Comparison Import BC Building DIV A.csv'
bc_buildingB = variation_folder_path + '/CCT Codes Comparison Import BC Building Div B.csv'
energy = variation_folder_path + '/CCT Codes Comparison Import Energy Code.csv'
fire = variation_folder_path + '/CCT Codes Comparison Import Fire Code.csv'
nl_building = variation_folder_path + '/CCT Codes Comparison Import NL Building.csv'
on_building_1_and_3 = variation_folder_path + '/CCT Codes Comparison Import ON Building Part 1 and 3.csv'
on_building_4_to_7 = variation_folder_path + '/CCT Codes Comparison Import ON Building Part 4 to 7.csv'
on_building_9 = variation_folder_path + '/CCT Codes Comparison Import ON Building Part 9 v2.csv'
on_building_8_to_12_no_9 = variation_folder_path + '/CCT Codes Comparison Import ON Building Parts 8 10 11 12.csv'
pei_building = variation_folder_path + '/CCT Codes Comparison Import PEI Building.csv'
plumbing = variation_folder_path + '/CCT Codes Comparison Import Plumbing.csv'
qc_building = variation_folder_path + '/CCT Codes Comparison Import QC Building.csv'
sk_building = variation_folder_path + '/CCT Codes Comparison Import SK Building.csv'


# Paths to the full P/T and National Codes
full_2015_folder_path = './Data/2015/full'

full_national_2015 = full_2015_folder_path + '/National Codes 2015 sentences.xlsx'
full_pt_2015 = full_2015_folder_path + '/PT Sentence Data 2015.xlsx'


# ##### Function to read the data from the file, drop any rows that are completely empty, and print the shape of the dataframe


# Function to read in the data, drop any rows that are completely empty, and print the shape of the dataframe
def read_data(file_path, code_type):
    df = pd.read_csv(file_path, encoding='latin1')
    df.dropna(how='all', inplace=True)
    df['Code Type'] = code_type
    print(df.shape)
    print(df.columns)
    return df


# ##### Read the Variations files into dataframes, add a column with *Code Type*, and check for the shape and column names


alberta_building_df = read_data(alberta_building, 'Building')


bc_buildingA_df = read_data(bc_buildingA, 'Building')


bc_buildingB_df = read_data(bc_buildingB, 'Building')


energy_df = read_data(energy, 'Energy')


fire_df = read_data(fire, 'Fire')


nl_building_df = read_data(nl_building, 'Building')


on_building_1_and_3_df = read_data(on_building_1_and_3, 'Building')


on_building_4_to_7_df = read_data(on_building_4_to_7, 'Building')


on_building_9_df = read_data(on_building_9, 'Building')


on_building_8_to_12_no_9_df = read_data(on_building_8_to_12_no_9, 'Building')


pei_building_df = read_data(pei_building, 'Building')


plumbing_df = read_data(plumbing, 'Plumbing')


qc_building_df = read_data(qc_building, 'Building')


sk_building_df = read_data(sk_building, 'Building')


# #### Check if all the columns of the dataframes, with the same number of columns, are the same


# ##### Dataframes with 34 columns


# print(alberta_building_df.columns == bc_buildingB_df.columns )
# print(bc_buildingB_df.columns == pei_building_df.columns)
# print(pei_building_df.columns == sk_building_df.columns)


# All the dataframes with 34 columns are the same.


# ##### Dataframes with 27 columns


# print(bc_buildingA_df.columns == energy_df.columns)
# print(energy_df.columns == fire_df.columns)
# print(fire_df.columns == nl_building_df.columns)
# These have the same columns /\



# These do not have the same columns \/
# print(nl_building_df.columns == on_building_1_and_3_df.columns)
# print(on_building_1_and_3_df.columns == on_building_4_to_7_df.columns)
# print(on_building_4_to_7_df.columns == on_building_8_to_12_no_9_df.columns)
# print(on_building_8_to_12_no_9_df.columns == on_building_9_df.columns)
# print(on_building_9_df.columns == plumbing_df.columns)
# print(plumbing_df.columns == qc_building_df.columns)


# Get the column order from nl_building_df
column_order = nl_building_df.columns

# Reindex the columns of the other dataframes to match the column order of nl_building_df
on_building_1_and_3_df = on_building_1_and_3_df.reindex(columns=column_order)
on_building_4_to_7_df = on_building_4_to_7_df.reindex(columns=column_order)
on_building_8_to_12_no_9_df = on_building_8_to_12_no_9_df.reindex(columns=column_order)
on_building_9_df = on_building_9_df.reindex(columns=column_order)
plumbing_df = plumbing_df.reindex(columns=column_order)
qc_building_df = qc_building_df.reindex(columns=column_order)

# Check if the columns are the same
# print(nl_building_df.columns == on_building_1_and_3_df.columns)
# print(on_building_1_and_3_df.columns == on_building_4_to_7_df.columns)
# print(on_building_4_to_7_df.columns == on_building_8_to_12_no_9_df.columns)
# print(on_building_8_to_12_no_9_df.columns == on_building_9_df.columns)
# print(on_building_9_df.columns == plumbing_df.columns)
# print(plumbing_df.columns == qc_building_df.columns)


# #### Combine all the dataframes into *variation_df*


# ##### DataFrame with 34 columns


variation_df_1 = pd.concat([alberta_building_df, bc_buildingB_df, pei_building_df, sk_building_df], ignore_index=True)
variation_df_1.shape


# ##### DataFrame with 27 columns


variation_df_2 = pd.concat([bc_buildingA_df, energy_df, fire_df, nl_building_df, on_building_1_and_3_df, on_building_4_to_7_df, on_building_8_to_12_no_9_df, on_building_9_df, plumbing_df, qc_building_df], ignore_index=True)
variation_df_2.shape


# Check for the column names in the two dataframes
print(variation_df_1.columns)
print(variation_df_2.columns)



# ##### Change the names of columns in the *variation_df_1* dataframe to match with those in the *variation_df_2* dataframe


# Removing the 'Matched ' and '?' from the column names
variation_df_1.columns = variation_df_1.columns.str.replace('Matched ', '') \
    .str.replace('?', '')

len(variation_df_2.columns)


# ##### Combine the *variation_df_1* and *variation_df_2* into one dataframe (fill the new columns with Nan values)


variation_df = pd.concat([variation_df_1, variation_df_2], axis=0, ignore_index=True)
variation_df.head()


variation_df.value_counts('Province/Territory')


# Let us change the *PEI* to *PE*


variation_df['Province/Territory'] = variation_df['Province/Territory'].replace({'PEI': 'PE'})
print(variation_df.value_counts('Province/Territory'))
print(f'The shape of the variation_df is {variation_df.shape}')


# #### Fix the PE dataframe
# For Prince Edward Island, the P/T Sentence Texts that are supposed to be in 'P/T Sentence Text' column are all in the 'P/T Article Title (FR)' column, except for one row.


# Isolate the PE rows
pe_rows = variation_df[variation_df['Province/Territory'] == 'PE'].copy()

# Function to replace the 'P/T Sentence Text' with the 'P/T Article Title (FR)' 
def fix_columns(df):
    mask = (df['P/T Sentence Text'].isna()) & (df['P/T Article Title (FR)'].notna())
    df.loc[mask, 'P/T Sentence Text'] = df.loc[mask, 'P/T Article Title (FR)']
    df.loc[mask, 'P/T Article Title (FR)'] = ""
    return df


pe_rows = fix_columns(pe_rows)

# Update the variation_df with the fixed PE rows
variation_df.update(pe_rows)

variation_df.shape


# ### Data Preprocessing


# Remove leading and trailing white spaces, any quotation marks, any weird characters, leading numerical or alphabetical bullet points, and any new line characters
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


# ##### Use the function to process the text data in the variation_df


variation_df['National Sentence Text'] = data_preprocessing(variation_df['National Sentence Text'])
variation_df['P/T Sentence Text'] = data_preprocessing(variation_df['P/T Sentence Text'])


variation_df['National Sentence Text'].value_counts()


len(variation_df['P/T Sentence Text'][(variation_df['Province/Territory'] == 'PE') & (variation_df['Difference Type'] == 'P/T Only')])


# ##### Save the combined data into a csv file


# (Please uncomment the code cell below to do that)


variation_df.to_csv('./Data/2015/variations/old-data/Combined 2015 Variations Data.csv', index=False)


# For now, we will be working with Division B since most of the sentences are not missing.


# ##### Check for the column names in variation_df and the value counts of P/T Division


variation_df['P/T Division'].value_counts()


variation_df['National Division'].value_counts()


# ### Isolate the rows with Division B


variation_df_B = variation_df[variation_df['National Division'] == 'Div B']
variation_df_B.shape


variation_df_B['Difference Type'].value_counts()


variation_df_B[(variation_df_B['National Sentence Text'].isna()) & (variation_df_B['P/T Sentence Text'].notna()) & (variation_df_B['Variation'] == 'Yes') & (variation_df_B['Difference Type'].isna())]['Province/Territory'].value_counts()


# ## Split the 2015 variation data based on the Province/Territory


# Function to fill the 'Difference Type' column bassed on National Sentence Text and P/T Sentence Text
def fill_difference_type(row):
    if pd.isna(row['Difference Type']):
        if row['Variation'] == 'Yes':
            if pd.notna(row['National Sentence Text']) and pd.isna(row['P/T Sentence Text']):
                return 'National Only'
            elif pd.isna(row['National Sentence Text']) and pd.notna(row['P/T Sentence Text']):
                return 'P/T Only'
        elif row['Variation'] == 'No':
            if pd.notna(row['National Sentence Text']) and pd.notna(row['P/T Sentence Text']):
                return 'Common Sentence'
    return row['Difference Type']

variation_df_B.loc[:, 'Difference Type'] = variation_df_B.apply(fill_difference_type, axis=1)
variation_df_B['Difference Type'].value_counts()


variation_df_B[(variation_df_B['Difference Type'] == 'Common Sentence') & (variation_df_B['P/T Sentence Text'].isna()) & (variation_df_B['National Sentence Text'].isna())]['Province/Territory'].value_counts()


# Function to remove all the rows in individual P/T variations with missing values for each 'Difference Type'
def remove_missing_values(df):
    # Iterate through the rows and apply the conditions to each row individually
    def row_filter(row):
        if pd.isna(row['Difference Type']):
            return not pd.isna(row['Difference Type'])
        elif row['Difference Type'] == 'National Only':
            return not pd.isna(row['National Sentence Text'])
        elif row['Difference Type'] == 'P/T Only':
            return not pd.isna(row['P/T Sentence Text'])
        else:
            return not (pd.isna(row['National Sentence Text']) or pd.isna(row['P/T Sentence Text']))

    filtered_df = df[df.apply(row_filter, axis=1)]
    return filtered_df

# Apply the function to the DataFrame
variation_df_B = remove_missing_values(variation_df_B)
variation_df_B.shape


variation_df_B.to_csv('./Data/2015/variations/old-data/Cleaned 2015 DivB Variations.csv', index=False)


# ##### Alberta


# We must also check for the non-NaN National sentence text values in each of these P/T to make sure all the sentences are present in each of them.


divb_path = './Data/2015/variations/old-data/div-b'


ab_df = variation_df_B[variation_df_B['Province/Territory'] == 'AB']
ab_df.to_csv(divb_path + '/2015 DivB AB Variations.csv', index=False)
print(ab_df.shape)


# ##### British Columbia


bc_df = variation_df_B[variation_df_B['Province/Territory'] == 'BC']
bc_df.to_csv(divb_path + '/2015 DivB BC Variations.csv', index=False)
print(bc_df.shape)


# ##### Newfoundland and Labrador 
# (We won't be using NL since it does not exist in the full code dataset)


nl_df = variation_df_B[variation_df_B['Province/Territory'] == 'NL']
print(nl_df.shape)


# ##### Nova Scotia


ns_df = variation_df_B[variation_df_B['Province/Territory'] == 'NS']
# ns_df = ns_df[(ns_df['Code Type'] != 'Energy') & (ns_df['Code Type'] != 'Plumbing')]
ns_df.to_csv(divb_path + '/2015 DivB NS Variations.csv', index=False)
print(ns_df.shape)


# ##### Nunavut


nu_df = variation_df_B[variation_df_B['Province/Territory'] == 'NU']
nu_df.to_csv(divb_path + '/2015 DivB NU Variations.csv', index=False)
print(nu_df.shape)


# ##### Ontario


on_df = variation_df_B[variation_df_B['Province/Territory'] == 'ON']
on_df.to_csv(divb_path + '/2015 DivB ON Variations.csv', index=False)
print(on_df.shape)


# ##### Prince Edward Island


pe_df = variation_df_B[variation_df_B['Province/Territory'] == 'PE']
pe_df.to_csv(divb_path + '/2015 DivB PE Variations.csv', index=False)
print(pe_df.shape)


# ##### Quebec 
# (Won't be using QC for now since it is in French)


qc_df = variation_df_B[variation_df_B['Province/Territory'] == 'QC']
print(qc_df.shape)


# ##### Saskatchewan


sk_df = variation_df_B[variation_df_B['Province/Territory'] == 'SK']
# sk_df = sk_df[(sk_df['Code Type'] != 'Plumbing')]
sk_df.to_csv(divb_path + '/2015 DivB SK Variations.csv', index=False)
print(sk_df.shape)


# ## Data Preprocessing (Full code data)


national_2015_df = pd.read_excel(full_national_2015)
pt_2015_df = pd.read_excel(full_pt_2015)


print(pt_2015_df.columns)
print(national_2015_df.columns)


pt_2015_df['PT Sentence Text'][~pt_2015_df['PT Sentence Text'].isna() & pt_2015_df['PT Sentence Text'].str.contains('"')]
pt_2015_df['PT Sentence Text'][73]


# ### Preprocess full code data


pt_2015_df['PT Sentence Text'] = data_preprocessing(pt_2015_df['PT Sentence Text'])
national_2015_df['FRAG_DOCUMENT'] = data_preprocessing(national_2015_df['FRAG_DOCUMENT'])

pt_2015_df['ARTICLE_TITLE'] = data_preprocessing(pt_2015_df['ARTICLE_TITLE'])
national_2015_df['ARTICLE_TITLE'] = data_preprocessing(national_2015_df['ARTICLE_TITLE'])


national_2015_df['FRAG_DOCUMENT'][~national_2015_df['FRAG_DOCUMENT'].isna() & national_2015_df['FRAG_DOCUMENT'].str.contains('')]


# ### Isolate rows from full code data with Division B


national_2015_df['DIVISION'].value_counts()


pt_2015_df['DIVISION'].value_counts()


national_2015_df_B = national_2015_df[national_2015_df['DIVISION'] == 'B']
pt_2015_df_B = pt_2015_df[pt_2015_df['DIVISION'] == 'B']


print("Shape of the dataframes")
print(f"P/T Codes: {pt_2015_df_B.shape}")
print(f"National Codes: {national_2015_df_B.shape}\n")

print("Number of missing sentences in")
print(f"P/T Codes: {pt_2015_df_B['PT Sentence Text'].isna().sum()}")
print(f"National Codes: {national_2015_df_B['FRAG_DOCUMENT'].isna().sum()}")


# ### Remove the empty sentences and store the sentence texts in a dataframe


pt_2015_df_B = pt_2015_df_B.copy()
national_2015_df_B = national_2015_df_B.copy()

pt_2015_df_B.dropna(subset=['PT Sentence Text'], inplace=True)
national_2015_df_B.dropna(subset=['FRAG_DOCUMENT'], inplace=True)

print("Number of text sentences in")    
print(f'P/T Codes: {pt_2015_df_B.shape[0]}')
print(f'National Codes: {national_2015_df_B.shape[0]}')

national_2015_df_B.to_csv('./Data/2015/full/div-b/2015 DivB National Full.csv', index=False)
pt_2015_df_B.to_csv('./Data/2015/full/div-b/2015 DivB PT Full.csv', index=False)


# ## Split the national sentence texts into train and test sets


national_2015_df_B = national_2015_df_B.copy()
national_2015_df_B['unique_id'] = national_2015_df_B['FRAG_DOCUMENT'] + national_2015_df_B['ARTICLE_TITLE']

# Remove duplicates based on the unique identifier
national_2015_df_B = national_2015_df_B.drop_duplicates(subset='unique_id')

# Split the unique national sentences into train and test sets
unique_train, unique_test = train_test_split(national_2015_df_B['FRAG_DOCUMENT'].unique(), test_size=0.2, random_state=SEED)

# Split the full national division B dataset into train/test based on the unique sentences
national_train = national_2015_df_B[national_2015_df_B['FRAG_DOCUMENT'].isin(unique_train)]
national_test = national_2015_df_B[national_2015_df_B['FRAG_DOCUMENT'].isin(unique_test)]

# Print the shapes of the train and test sets and check for common sentences between the two
print(f"Train: {national_train.shape}")
print(f"Test: {national_test.shape}")
print(f"Common sentences between the National train/test: {len(set(national_train['FRAG_DOCUMENT']) & set(national_test['FRAG_DOCUMENT']))}")


national_2015_df_B.shape


# ## Split the 2015 individual P/T variations data into train/test sets


# def sentence_similarity(s1, s2, threshold):
#     if isinstance(s1, str) and isinstance(s2, str):
#         words1 = s1.strip().split()
#         words2 = s2.strip().split()
#         # common_words = set(words1) & set(words2)
        
#         common_words = [word for word in words1 if word in words2]
#         max_len = max(len(words1), len(words2))
#         similarity = len(common_words) / max_len

#         if max_len <= 20:
#             threshold = (max_len - 4) / max_len
            
#         return similarity >= threshold
#     else:
#         return False


# #### Function to preprocess the text fields


def text_preprocessing(column):
    column = column.str.replace('', ' ') \
                    .str.strip() \
                    .str.replace(r'^[\da-zA-Z]+\)', ' ', regex=True) \
                    .str.replace('\n', ' ') \
                    .str.replace('\x93', ' ') \
                    .str.replace('\x94', ' ') \
                    .str.replace('\x96', ' ') \
                    .str.replace('\x97', ' ') \
                    .str.replace(r' [\da-zA-Z]+\) ', ' ', regex=True) \
                    .str.replace(r'\s{2,}', ' ', regex=True) \
                    .str.replace('?', ' ') \
                    .str.replace(r'^\d+(\.\d+)*[A-Za-z]*\.?\s*', '', regex=True) \
                    .str.strip()
    # .str.replace(r'\s*\(.*?\)$', '', regex=True) \

    return column


# #### Isolate national sentence texts from the National Full code


national_full = national_2015_df_B.copy()
national_full = national_full[['FRAG_DOCUMENT']]
national_full.rename(columns={'FRAG_DOCUMENT': 'National Full'}, inplace=True)
national_full['Processed National Full'] = text_preprocessing(national_full['National Full'])


# #### Compute the similarity between National sentences in full and variations data


# Function to compute the similarties between National sentences in National Full data amd those in P/T data
def compute_similarity(national, pt):
    words1 = national.strip().split()
    words2 = pt.strip().split()
    common_words = [word for word in words1 if word in words2]
    max_len = max(len(words1), len(words2))
    similarity = len(common_words) / max_len
    return similarity


# For each P/T, find the similarity between each National sentence in National Full and each sentence in P/T data
def find_similarity(pt):
    results = []
    full = pd.read_csv(f'./Data/2015/variations/div-b/2015 DivB {pt} Variations.csv')
    full = full[['National Sentence Text']]
    full = full[full['National Sentence Text'].notna()]
    full.rename(columns={'National Sentence Text': f'National in {pt}'}, inplace=True)
    full[f'Processed National in {pt}'] = text_preprocessing(full[f'National in {pt}'])

    # Compare each processed sentence in national_full with all processed sentences in pt_full
    for i, row1 in national_full.iterrows():
        for j, row2 in full.iterrows():
            similarity = compute_similarity(row1['Processed National Full'], row2[f'Processed National in {pt}'])
            results.append({
                'National Full': row1['National Full'],
                f'National in {pt}': row2[f'National in {pt}'],
                'Similarity': similarity
            })
    
    # Create a new dataframe with the results
    similarity_df = pd.DataFrame(results)

    # Sort the dataframe by similarity in descending order
    similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)

    # Filter out the rows to keep only the highest similarity score for each unique sentence in national_full
    unique_national_sentences = set()
    filtered_results = []

    for index, row in similarity_df.iterrows():
        if row['National Full'] not in unique_national_sentences:
            filtered_results.append(row)
            unique_national_sentences.add(row['National Full'])

    
    filtered_similarity_df = pd.DataFrame(filtered_results)


     # Filter out the rows to keep only the highest similarity score for each unique sentence in pt
    unique_variations_sentences = set()
    final_filtered_results = []

    for index, row in filtered_similarity_df.iterrows():
        if row[f'National in {pt}'] not in unique_variations_sentences:
            final_filtered_results.append(row)
            unique_variations_sentences.add(row[f'National in {pt}'])

    # Save the final results to a CSV file
    final_filtered_similarity_df = pd.DataFrame(final_filtered_results)
    final_filtered_similarity_df.to_csv(f'./Data/2015/national-similarity/2015 {pt} National Similarity.csv', index=False)
    print(f'{pt} similarity saved successfully')


# # Store all P/T names in a list
# pt_name = ['AB', 'BC', 'NS', 'NU', 'ON', 'PE', 'SK']

# for pt in pt_name:
#     find_similarity(pt)


# #### Isolate P/T Sentence Texts from the P/T full code


# Read the full P/T data
pt_full = pd.read_csv('./Data/2015/full/div-b/2015 DivB PT Full.csv')

# Function to read and preprocess the full P/T data
def read_pt_full(pt_name):
    pt = pt_full[pt_full['PT'] == pt_name]
    pt = pt[['PT Sentence Text']]
    pt.rename(columns={'PT Sentence Text': f'{pt_name} Full'}, inplace=True)
    pt[f'Processed {pt_name} Full'] = text_preprocessing(pt[f'{pt_name} Full'])

    return pt


# Read the full P/T data into their respective dataframes
ab_full = read_pt_full('AB')
bc_full = read_pt_full('BC')
ns_full = read_pt_full('NS')
nu_full = read_pt_full('NU')
on_full = read_pt_full('ON')
pe_full = read_pt_full('PE')
sk_full = read_pt_full('SK')


# #### Compute the similarity between P/T sentences in full and variations data


def compute_pt_similarity(full, variation):
    if isinstance(full, str) and isinstance(variation, str):
        words1 = full.strip().split()
        words2 = variation.strip().split()
        common_words = [word for word in words1 if word in words2]
        max_len = max(len(words1), len(words2))
        similarity = len(common_words) / max_len
        return similarity
    else:
        return 0

# For each P/T, find the similarity between each P/T sentence in Full and each sentence in variation P/T data
def find_pt_similarity(pt_name, pt_full):
    results = []
    variations = pd.read_csv(f'./Data/2015/variations/div-b/2015 DivB {pt_name} Variations.csv')
    variations = variations[['P/T Sentence Text']]
    variations = variations[variations['P/T Sentence Text'].notna()]
    variations.rename(columns={'P/T Sentence Text': f'{pt_name} Variations'}, inplace=True)
    variations[f'Processed {pt_name} Variations'] = text_preprocessing(variations[f'{pt_name} Variations'])

    # Compare each processed sentence in variations with all processed sentences in Full
    for i, row1 in pt_full.iterrows():
        for j, row2 in variations.iterrows():
            similarity = compute_pt_similarity(row1[f'Processed {pt_name} Full'], row2[f'Processed {pt_name} Variations'])
            results.append({
                f'{pt_name} Full': row1[f'{pt_name} Full'],
                f'{pt_name} Variations': row2[f'{pt_name} Variations'],
                'Similarity': similarity
            })
    
    # Create a new dataframe with the results
    similarity_df = pd.DataFrame(results)

    # Sort the dataframe by similarity in descending order
    similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)

    # Filter out the rows to keep only the highest similarity score for each unique sentence in national_full
    unique_full_sentences = set()
    filtered_results = []

    for index, row in similarity_df.iterrows():
        if row[f'{pt_name} Full'] not in unique_full_sentences:
            filtered_results.append(row)
            unique_full_sentences.add(row[f'{pt_name} Full'])

    
    filtered_similarity_df = pd.DataFrame(filtered_results)


     # Filter out the rows to keep only the highest similarity score for each unique sentence in pt
    unique_variations_sentences = set()
    final_filtered_results = []

    for index, row in filtered_similarity_df.iterrows():
        if row[f'{pt_name} Variations'] not in unique_variations_sentences:
            final_filtered_results.append(row)
            unique_variations_sentences.add(row[f'{pt_name} Variations'])

    # Save the final results to a CSV file
    final_filtered_similarity_df = pd.DataFrame(final_filtered_results)
    final_filtered_similarity_df.to_csv(f'./Data/2015/pt-similarity/2015 {pt_name} PT Similarity.csv', index=False)
    print(f'{pt_name} similarity saved successfully')


# pt_names1 = ['AB', 'BC', 'NS', 'NU']

# for pt_name in pt_names1:
#     find_pt_similarity(pt_name, eval(f'{pt_name.lower()}_full'))



# pt_names2 = ['ON']

# for pt_name in pt_names2:
#     find_pt_similarity(pt_name, eval(f'{pt_name.lower()}_full'))


# pt_names3 = ['PE', 'SK']

# for pt_name in pt_names3:
#     find_pt_similarity(pt_name, eval(f'{pt_name.lower()}_full'))


# ##### Function to split individual Province/Territories data into train/test sets and save them as csv files


# The function below only tries to match only the national sentence texts in the variations data and the national sentence texts in the full code data. 


def split_and_save_data(df, string, national_threshold, pt_threshold):

    # Creating a duplicate dataframe to work with
    ddf = df.copy()

    # Reading similiarity file for the P/T
    national_similarity = pd.read_csv(f'./Data/2015/national-similarity/2015 {string} National Similarity.csv')
    pt_similarity = pd.read_csv(f'./Data/2015/pt-similarity/2015 {string} PT Similarity.csv')

    # Isolate rows with similarity scores above the threshold
    national_similarity = national_similarity[national_similarity['Similarity'] >= national_threshold]
    pt_similarity = pt_similarity[pt_similarity['Similarity'] >= pt_threshold]

    # Split the similarity dataframe into train and test sets
    train_sim = national_similarity[national_similarity['National Full'].isin(unique_train)]
    test_sim = national_similarity[national_similarity['National Full'].isin(unique_test)]

    # Splitting the variations data into train and test sets
    train = ddf[ddf['National Sentence Text'].isin(train_sim[f'National in {string}'])]
    train = train[(train['P/T Sentence Text'].isin(pt_similarity[f'{string} Variations'])) | (train['P/T Sentence Text'].isna())]
    ddf = ddf[~ddf.index.isin(train.index)]

    test = ddf[ddf['National Sentence Text'].isin(test_sim[f'National in {string}'])]
    train = train[(train['P/T Sentence Text'].isin(pt_similarity[f'{string} Variations'])) | (train['P/T Sentence Text'].isna())]
    ddf = ddf[~ddf.index.isin(test.index)]


    # # Isolating the train/test sentences in full national code data with exact match and removing them from the dataframe
    # train = ddf[ddf['National Sentence Text'].isin(unique_train)]
    # ddf = ddf[~ddf.index.isin(train.index)]

    # test = ddf[ddf['National Sentence Text'].isin(unique_test)]
    # ddf = ddf[~ddf.index.isin(test.index)]


    # # Checking for sentences with a certain threshold match in the train and test data and removing them from the dataframe
    # train = pd.concat([train, ddf[ddf['National Sentence Text'].apply(lambda x: any(sentence_similarity(x, s, threshold) for s in unique_train))]])
    # ddf = ddf[~ddf.index.isin(train.index)]
    
    # test = pd.concat([test, ddf[ddf['National Sentence Text'].apply(lambda x: any(sentence_similarity(x, s, threshold) for s in unique_test))]])
    # ddf = ddf[~ddf.index.isin(test.index)]


    # Checking for empty National Sentence Texts and combining all three dataframes
    empty = ddf[(ddf['National Sentence Text'].isna()) & (ddf['P/T Sentence Text'].isin(pt_similarity[f'{string} Variations'])) ]
    main = pd.concat([train, test, empty])
    other_national = ddf[~ddf.index.isin(main.index)]

    # # Isolating the national sentence texts in the variations data but not in the full data
    #  other_national = df[~df.index.isin(main.index)]

    empty_train = pd.DataFrame()
    empty_test = pd.DataFrame()
    
    if empty.shape[0] != 0:
        total_train = int(np.round(0.8 * main.shape[0]))
        total_test = main.shape[0] - total_train

        empty_train_len = total_train - train.shape[0]
        empty_test_len = total_test - test.shape[0]

        empty_train, empty_test = train_test_split(empty, train_size=empty_train_len, test_size=empty_test_len, random_state=SEED)

    train_set = pd.concat([train, empty_train])
    test_set = pd.concat([test, empty_test])

    print(f"{string}")
    print(f"Full Data: {df.shape[0]}")
    print(f"Train: {train_set.shape[0]}")
    print(f"Test: {test_set.shape[0]}")
    print(f"National or P/T sentences in variations data but not in full data: {other_national.shape[0]}")

    # Check if there are any common sentences between the train and test data
    print(f"Common sentences between train and test: {(set(train['National Sentence Text']) & set(test['National Sentence Text']))}")

    # Saving the train and test dataframes as csv files
    train_set.to_csv(f'./Data/2015/pt-train-test-sets/{string} Train.csv', index=False)
    test_set.to_csv(f'./Data/2015/pt-train-test-sets/{string} Test.csv', index=False)

    return train_set, test_set, other_national


# ### Use function to split Variation data into train/test for individual Province/Territories


ab_train, ab_test, ab_other = split_and_save_data(ab_df, 'AB', 0.7, 0.51)


bc_train, bc_test, bc_other = split_and_save_data(bc_df, 'BC', 0, 0.84)


ns_train, ns_test, ns_other = split_and_save_data(ns_df, 'NS', 0, 0.62)


nu_train, nu_test, nu_other = split_and_save_data(nu_df, 'NU', 0, 0)


on_train, on_test, on_other = split_and_save_data(on_df, 'ON', 0.55, 0.54)


pe_train, pe_test, pe_other = split_and_save_data(pe_df, 'PE', 0, 0)


sk_train, sk_test, sk_other = split_and_save_data(sk_df, 'SK', 0, 0.92)


# ## Save remaining files


# ### Save the full code national train/test sets as csv files


national_train.to_csv('./Data/2015/full/div-b/national-train-test/2015 National Train.csv', index=False)
national_test.to_csv('./Data/2015/full/div-b/national-train-test/2015 National Test.csv', index=False)


# ### Save the left out data as csv files


def save_leftout_data(df, string):
    df.to_csv(f'./Data/2015/pt-train-test-sets/{string} Leftout.csv', index=False)


pt_name = ['AB', 'BC', 'NS', 'NU', 'PE', 'SK']

for pt in pt_name:
    save_leftout_data(eval(f'{pt.lower()}_other'), pt)


# #### Save the individual PT full codes


def save_pt_full(string):
    df = pt_full[pt_full['PT'] == string]
    df.to_csv(f'./Data/2015/full/div-b/pt-full/2015 DivB {string} Full.csv', index=False)

pt_names = ['AB', 'BC', 'NS', 'NU', 'ON', 'PE', 'SK']

for pt in pt_names:
    save_pt_full(pt)


# # Approach 1


def token_similarity(national, pt, threshold=0.9):
    if isinstance(national, str) and isinstance(pt, str):
        words1 = national.strip().split()
        words2 = pt.strip().split()
        common_words = [word for word in words1 if word in words2]
        
        max_len = max(len(words1), len(words2))
        similarity = len(common_words) / max_len
          
        return similarity >= threshold
    else:
        return False
    


def save_data(df, string):
    # Creating a duplicate dataframe to work with
    ddf = df.copy()

    # Isolating the train/test sentences in full national code data with exact match and removing them from the dataframe
    train = ddf[ddf['National Sentence Text'].isin(unique_train)]
    ddf = ddf[~ddf.index.isin(train.index)]

    test = ddf[ddf['National Sentence Text'].isin(unique_test)]
    ddf = ddf[~ddf.index.isin(test.index)]

    # Checking for sentences with a 90% match in the train and test data and removing them from the dataframe
    train = pd.concat([train, ddf[ddf['National Sentence Text'].apply(lambda x: any(token_similarity(s, x) for s in unique_train))]])
    ddf = ddf[~ddf.index.isin(train.index)]
    
    test = pd.concat([test, ddf[ddf['National Sentence Text'].apply(lambda x: any(token_similarity(s, x) for s in unique_test))]])
    ddf = ddf[~ddf.index.isin(test.index)]

    # Checking for empty National Sentence Texts and combining all three dataframes
    empty = ddf[ddf['National Sentence Text'].isna()]
    main = pd.concat([train, test, empty])
    other_national = ddf[~ddf.index.isin(main.index)]

    empty_train = pd.DataFrame()
    empty_test = pd.DataFrame()
    
    if empty.shape[0] != 0:
        total_train = int(np.round(0.8 * main.shape[0]))
        total_test = main.shape[0] - total_train

        empty_train_len = total_train - train.shape[0]
        empty_test_len = total_test - test.shape[0]

        empty_train, empty_test = train_test_split(empty, train_size=empty_train_len, test_size=empty_test_len, random_state=SEED)

    train_set = pd.concat([train, empty_train])
    test_set = pd.concat([test, empty_test])
    

    print(f"{string}")
    print(f"Full Data: {df.shape[0]}")
    print(f"Train: {train_set.shape[0]}")
    print(f"Test: {test_set.shape[0]}")
    print(f"National sentences in variations data but not in full data: {other_national.shape[0]}")

    # Check if there are any common sentences between the train and test data
    print(f"Common sentences between train and test: {(set(train['National Sentence Text']) & set(test['National Sentence Text']))}\n")

    # Saving the train and test dataframes as csv files
    train_set.to_csv(f'./Data/pt-train-test-sets/old-sets/Old {string} Train.csv', index=False)
    test_set.to_csv(f'./Data/pt-train-test-sets/old-sets/Old {string} Test.csv', index=False)



pt_name = ['AB', 'BC', 'NS', 'NU', 'ON', 'PE', 'SK']

for pt in pt_name:
    save_data(eval(f'{pt.lower()}_df'), pt)


def sentence_similarity(national, pt, threshold=0.9):
    if isinstance(national, str) and isinstance(pt, str):
        words1 = national.strip().split()
        words2 = pt.strip().split()
        common_words = [word for word in words1 if word in words2]
        
        max_len = max(len(words1), len(words2))
        similarity = len(common_words) / max_len
            
        return similarity
    else:
        return 0
    

def save_similarity_scores(string, threshold=0.7):
    results = []

    df = pd.read_csv(f'./Data/2015-divb/code-variations-2015-divb/DivB {string} Variations 2015.csv')
    df['National Sentence Text'] = text_preprocessing(df['National Sentence Text'])

    national_full['National Full'] = text_preprocessing(national_full['National Full'])
    national_unique = national_full['National Full'].unique()

    # exact_match = df[df['National Sentence Text'].isin(national_unique)]
    # df = df[~df.index.isin(exact_match.index)]
    
    
    if df.shape[0] != 0:
        for pt_index, pt_row in df.iterrows():
            pt_sentence = pt_row['National Sentence Text']
            
            for national_sentence in national_unique:
                if national_sentence == pt_sentence:
                    similarity = 1
                    results.append({
                        'National in Full': national_sentence,
                        f'National in {string}': pt_sentence,
                        'Similarity': similarity
                    })
                else:
                    similarity = sentence_similarity(national_sentence, pt_sentence, threshold)
                    
                    if (similarity >= threshold):
                        results.append({
                            'National in Full': national_sentence,
                            f'National in {string}': pt_sentence,
                            'Similarity': similarity
                        })
    
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Similarity', ascending=False)
    else:
        results_df = pd.DataFrame()
    
    results_df.to_csv(f'./Data/old-similarity/Old {string} Similarity.csv', index=False)
    print(f'{string} similarity scores saved')


for pt in pt_name:
    save_similarity_scores(pt)


# Evaluation using random sampling
old_test = pd.read_csv('./Data/pt-train-test-sets/old-sets/Old ON Test.csv')
new_test = pd.read_csv('./Data/pt-train-test-sets/ON Test.csv')

old_test = old_test[old_test['National Sentence Text'].notna()]
new_test = new_test[new_test['National Sentence Text'].notna()]

# Sample about 30 sentences from both the old and new test sets
old_sample = old_test['National Sentence Text'].sample(30, random_state=SEED)
new_sample = new_test['National Sentence Text'].sample(30, random_state=SEED)

old_sample.to_csv('./Data/samples/Old ON Test Samples.csv', index=False)
new_sample.to_csv('./Data/samples/New ON Test Samples.csv', index=False)


import pandas as pd

# Merging all the similarity scores files from Approach 2 into one single file
pt_thresholds = {'AB': 0.7, 'BC': 0, 'NS': 0, 'NU': 0, 'ON': 0.6, 'PE': 0, 'SK': 0}

for pt, threshold in pt_thresholds.items():
    similarity = pd.read_csv(f'./Data/new-similarity/{pt} Similarity.csv')
    similarity = similarity[similarity['Similarity'] >= threshold]
    similarity.to_csv(f'./Data/new-similarity/{pt} Similarity.csv', index=False)

# Initialize the final dataframe
appended_df = pd.DataFrame()

# Read and merge all files
for pt, threshold in pt_thresholds.items():
    pt_similarity = pd.read_csv(f'./Data/new-similarity/{pt} Similarity.csv')
    pt_similarity = pt_similarity[['National Full', f'National in {pt}']]
    if appended_df.empty:
        appended_df = pt_similarity
    else:
        appended_df = pd.merge(appended_df, pt_similarity, on='National Full', how='outer')

# Fill NaN values with empty strings
appended_df.fillna('', inplace=True)

# Save the final dataframe to a CSV file
appended_df.to_csv('./Data/new-similarity/Merged Similarity.csv', index=False)

# Check for duplicates
duplicates = appended_df[appended_df.duplicated(subset='National Full')]
print(duplicates)



# #### Computing the similarity scores of the 2015 P/T Code sentences in Full and Variations


# Read the full P/T data
pt_full = pd.read_csv('./Data/2015-divb/code-full-2015-divb/DivB PT Full 2015.csv')

# Function to read and preprocess the full P/T data
def read_pt_full(pt_name):
    pt = pt_full[pt_full['PT'] == pt_name]
    pt = pt[['PT Sentence Text']]
    pt.rename(columns={'PT Sentence Text': f'{pt_name} Full'}, inplace=True)
    pt[f'Processed {pt_name} Full'] = text_preprocessing(pt[f'{pt_name} Full'])

    return pt

# Read the full P/T data into their respective dataframes
ab_full = read_pt_full('AB')
bc_full = read_pt_full('BC')
ns_full = read_pt_full('NS')
nu_full = read_pt_full('NU')
on_full = read_pt_full('ON')
pe_full = read_pt_full('PE')
sk_full = read_pt_full('SK')


# Function to compute the similarties between P/T sentences in Full data and those in variation P/T data
def compute_similarity(full, variations):
    if isinstance(full, str) and isinstance(variations, str):
        words1 = full.strip().split()
        words2 = variations.strip().split()
        common_words = [word for word in words1 if word in words2]
        max_len = max(len(words1), len(words2))
        similarity = len(common_words) / max_len
        return similarity
    else:
        return 0


# For each P/T, find the similarity between each P/T sentence in Full and each sentence in variation P/T data
def find_similarity(pt_name, pt_full):
    results = []
    train = pd.read_csv(f'./Data/pt-train-test-sets/{pt_name} Train.csv')
    test = pd.read_csv(f'./Data/pt-train-test-sets/{pt_name} Test.csv')
    variations = pd.concat([train, test], ignore_index=True)
    variations = variations[['P/T Sentence Text']]
    variations = variations[variations['P/T Sentence Text'].notna()]
    variations.rename(columns={'P/T Sentence Text': f'{pt_name} Variations'}, inplace=True)
    variations[f'Processed {pt_name} Variations'] = text_preprocessing(variations[f'{pt_name} Variations'])

    # Compare each processed sentence in variations with all processed sentences in Full
    for i, row1 in pt_full.iterrows():
        for j, row2 in variations.iterrows():
            similarity = compute_similarity(row1[f'Processed {pt_name} Full'], row2[f'Processed {pt_name} Variations'])
            results.append({
                f'{pt_name} Full': row1[f'{pt_name} Full'],
                f'{pt_name} Variations': row2[f'{pt_name} Variations'],
                'Similarity': similarity
            })
    
    # Create a new dataframe with the results
    similarity_df = pd.DataFrame(results)

    # Sort the dataframe by similarity in descending order
    similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)

    # Filter out the rows to keep only the highest similarity score for each unique sentence in national_full
    unique_full_sentences = set()
    filtered_results = []

    for index, row in similarity_df.iterrows():
        if row[f'{pt_name} Full'] not in unique_full_sentences:
            filtered_results.append(row)
            unique_full_sentences.add(row[f'{pt_name} Full'])

    
    filtered_similarity_df = pd.DataFrame(filtered_results)


     # Filter out the rows to keep only the highest similarity score for each unique sentence in pt
    unique_variations_sentences = set()
    final_filtered_results = []

    for index, row in filtered_similarity_df.iterrows():
        if row[f'{pt_name} Variations'] not in unique_variations_sentences:
            final_filtered_results.append(row)
            unique_variations_sentences.add(row[f'{pt_name} Variations'])

    # Save the final results to a CSV file
    final_filtered_similarity_df = pd.DataFrame(final_filtered_results)
    final_filtered_similarity_df.to_csv(f'./Data/pt-similarity/{pt_name} PT Similarity.csv', index=False)
    print(f'{pt_name} similarity saved successfully')


# Store all P/T names in a list
# pt_name = ['AB', 'BC', 'NS', 'NU', 'ON', 'PE', 'SK']
pt_name = ['PE', 'SK']

# Save all the similarity scores for each P/T
for pt in pt_name:
    find_similarity(pt, eval(f'{pt.lower()}_full'))


# For each P/T, find the similarity between each P/T sentence in Full and each sentence in variation P/T data
def find_on_similarity(pt_name, pt_full):
    results = []
    # train = pd.read_csv(f'./Data/pt-train-test-sets/{pt_name} Train.csv')
    test = pd.read_csv(f'./Data/pt-train-test-sets/{pt_name} Test.csv')
    variations = test.copy()
    variations = variations[['P/T Sentence Text']]
    variations = variations[variations['P/T Sentence Text'].notna()]
    variations.rename(columns={'P/T Sentence Text': f'{pt_name} Variations'}, inplace=True)
    variations[f'Processed {pt_name} Variations'] = text_preprocessing(variations[f'{pt_name} Variations'])

    # Compare each processed sentence in variations with all processed sentences in Full
    for i, row1 in pt_full.iterrows():
        for j, row2 in variations.iterrows():
            similarity = compute_similarity(row1[f'Processed {pt_name} Full'], row2[f'Processed {pt_name} Variations'])
            results.append({
                f'{pt_name} Full': row1[f'{pt_name} Full'],
                f'{pt_name} Variations': row2[f'{pt_name} Variations'],
                'Similarity': similarity
            })
    
    # Create a new dataframe with the results
    similarity_df = pd.DataFrame(results)

    # Sort the dataframe by similarity in descending order
    similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)

    # Filter out the rows to keep only the highest similarity score for each unique sentence in national_full
    unique_full_sentences = set()
    filtered_results = []

    for index, row in similarity_df.iterrows():
        if row[f'{pt_name} Full'] not in unique_full_sentences:
            filtered_results.append(row)
            unique_full_sentences.add(row[f'{pt_name} Full'])

    
    filtered_similarity_df = pd.DataFrame(filtered_results)


     # Filter out the rows to keep only the highest similarity score for each unique sentence in pt
    unique_variations_sentences = set()
    final_filtered_results = []

    for index, row in filtered_similarity_df.iterrows():
        if row[f'{pt_name} Variations'] not in unique_variations_sentences:
            final_filtered_results.append(row)
            unique_variations_sentences.add(row[f'{pt_name} Variations'])

    # Save the final results to a CSV file
    final_filtered_similarity_df = pd.DataFrame(final_filtered_results)
    final_filtered_similarity_df.to_csv(f'./Data/pt-similarity/{pt_name} PT Test Similarity.csv', index=False)
    print(f'{pt_name} similarity saved successfully')


find_on_similarity('ON', on_full)


# #### BC train and test sets and ON train sets have rows which are mislabelled
# Sentences with the following properties in BC train and test sets are changed to 'Difference Type' == 'Common Sentence':
# * 'Difference Type' == 'P/T Only' or 'National Only'
# * 'Variation' == 'No'
# * non-missing 'National Sentence Text' and 'P/T Sentence Text'


bc_train = pd.read_csv('./Data/2015/pt-train-test-sets/BC Train.csv')
bc_test = pd.read_csv('./Data/2015/pt-train-test-sets/BC Test.csv')
on_train = pd.read_csv('./Data/2015/pt-train-test-sets/ON Train.csv')

def change_labels(df):
    df.loc[
        ((df['Difference Type'].isin(['P/T Only', 'National Only'])) &
        (df['Variation'] == 'No') &
        df['National Sentence Text'].notna() &
        df['P/T Sentence Text'].notna()),
        'Difference Type'
    ] = 'Common Sentence'

    return df

bc_train = change_labels(bc_train)
bc_train.to_csv('./Data/2015/pt-train-test-sets/BC Train.csv', index=False)

bc_test = change_labels(bc_test)
bc_test.to_csv('./Data/2015/pt-train-test-sets/BC Test.csv', index=False)

on_train = change_labels(on_train)



