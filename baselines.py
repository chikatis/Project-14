# ## Baseline models
# #### Importing the necessary libraries
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize.punkt import PunktSentenceTokenizer
import nltk
nltk.download('punkt')


# #### File Paths
full_code_2015 = './Data/2015/full/div-b'
pt_full_2015 = full_code_2015 + '/pt-full'

national_full_path = full_code_2015 + '/2015 National Full.csv' 

national_similarity_path = './Data/2015/national-similarity'
pt_similarity_path = './Data/2015/pt-similarity'

train_test_sets = './Data/2015/pt-train-test-sets'


# ### Function to preprocess sentence texts
# Preprocess the sentences using NLTK's Punkt tokenizer
tokenizer = PunktSentenceTokenizer()

def preprocess(sentences):
    # return [' '.join(tokenizer.tokenize(sentence.lower().replace(r'[\)]', '. ').replace(r'\)(?=[^\w\s])', ') '))) for sentence in sentences]
    preprocessed_sentences = []
    for sentence in sentences:
        if pd.isna(sentence):
            preprocessed_sentences.append('')
            continue
        
        # Convert to lowercase
        sentence = sentence.lower()

        # Ensure spaces after periods and closing parentheses if not followed by a digit or a character
        sentence = re.sub(r'\.(?=[^\d\s])', '. ', sentence)
        sentence = re.sub(r'(\))(?=[a-zA-Z])', ') ', sentence)

        # Tokenize the sentence
        tokens = tokenizer.tokenize(sentence)
        preprocessed_sentences.append(' '.join(tokens))
    return preprocessed_sentences


# Function to map the labels
def map_labels(row):
    if row['Difference Type'] == 'Common Sentence':
        if row['Variation'] == 'Yes':
            return 'Variation'
        else:
            return 'Common Sentence'
    return row['Difference Type']


# Function to calculate the cosine similarity between the national and PT sentences
def baseline_models(national_df, pt_df):
    # Extract the sentence texts columns
    national_sentences = national_df['FRAG_DOCUMENT']
    pt_sentences = pt_df['PT Sentence Text']

    national_df['Preprocessed Sentences'] = preprocess(national_sentences)
    pt_df['Preprocessed Sentences'] = preprocess(pt_sentences)

    # Calculate cosine similarity
    def calculate_similarity(matrix_a, matrix_b):
        return cosine_similarity(matrix_a, matrix_b)

    # Create a combined vocabulary
    corpus = list(national_df['Preprocessed Sentences']) + list(pt_df['Preprocessed Sentences'])

    # Bag of Words
    vectorizer_bow = CountVectorizer()
    vectorizer_bow.fit(corpus)
    national_bow = vectorizer_bow.transform(national_df['Preprocessed Sentences'])
    pt_bow = vectorizer_bow.transform(pt_df['Preprocessed Sentences'])

    cosine_similarity_bow = calculate_similarity(national_bow, pt_bow)

    # TF-IDF
    vectorizer_tfidf = TfidfVectorizer()
    vectorizer_tfidf.fit(corpus)
    national_tfidf = vectorizer_tfidf.transform(national_df['Preprocessed Sentences'])
    pt_tfidf = vectorizer_tfidf.transform(pt_df['Preprocessed Sentences'])

    cosine_similarity_tfidf = calculate_similarity(national_tfidf, pt_tfidf)

    # N-Hot Encoding
    vocabulary_n_hot = list(vectorizer_bow.get_feature_names_out())

    def n_hot_encode(sentences, vocab):
        n_hot_vectors = np.zeros((len(sentences), len(vocab)))
        for i, sentence in enumerate(sentences):
            for word in sentence.split():
                if word in vocab:
                        n_hot_vectors[i, vocab.index(word)] = 1
        return n_hot_vectors
    
    national_n_hot = n_hot_encode(national_df['Preprocessed Sentences'], vocabulary_n_hot)
    pt_n_hot = n_hot_encode(pt_df['Preprocessed Sentences'], vocabulary_n_hot)

    cosine_similarity_n_hot = calculate_similarity(national_n_hot, pt_n_hot)

    cosine_similarity_matrices = {
        'Bag of Words': cosine_similarity_bow,
        'TF-IDF': cosine_similarity_tfidf,
        'N-Hot': cosine_similarity_n_hot
    }

    return cosine_similarity_matrices


# Function to get the best possible matches
def get_matchings(similarity_matrix, national_df, pt_df):
    national_ind, pt_ind = linear_sum_assignment(similarity_matrix, maximize=True)

    unmatched_similarity_list = []

    unmatched_national_ind = [i for i in range(len(national_df)) if i not in national_ind]
    unmatched_pt_ind = [j for j in range(len(pt_df)) if j not in pt_ind]

    if len(unmatched_national_ind) > 0:
        len_unmatched_national_ind = len(unmatched_national_ind)
        unmatched_similarity_list.extend(list(zip(
            national_df['Full Index'].iloc[unmatched_national_ind], 
            national_df['FRAG_DOCUMENT'].iloc[unmatched_national_ind], 
            [None] * len_unmatched_national_ind,
            [''] * len_unmatched_national_ind,
            np.zeros(len_unmatched_national_ind))))
        
    if len(unmatched_pt_ind) > 0:
        len_unmatched_pt_ind = len(unmatched_pt_ind)
        unmatched_similarity_list.extend(list(zip(
            [None] * len_unmatched_pt_ind,
            [''] * len_unmatched_pt_ind,
            pt_df['Full Index'].iloc[unmatched_pt_ind],
            pt_df['PT Sentence Text'].iloc[unmatched_pt_ind],
            np.zeros(len_unmatched_pt_ind))))

    similarity_list = list(zip(
        national_df['Full Index'].iloc[national_ind], 
        national_df['FRAG_DOCUMENT'].iloc[national_ind], 
        pt_df['Full Index'].iloc[pt_ind], 
        pt_df['PT Sentence Text'].iloc[pt_ind], 
        similarity_matrix[national_ind, pt_ind]))
    
    similarity_list.extend(unmatched_similarity_list)

    similarity_df = pd.DataFrame(similarity_list, columns=['National Identifier', 'National Sentence Text', 'P/T Identifier', 'P/T Sentence Text', 'Similarity'])

    return similarity_df


# Function to get the alignments based on threshold
def get_alignments(similarity_df, national_similarity, pt_similarity, string, alignment_threshold):
    final_alignments = []

    for _, row in similarity_df.iterrows():
        national_identifier = row['National Identifier']
        pt_identifier = row['P/T Identifier']
        cosine_similarity = row['Similarity']

        matched_national_rows = national_similarity[national_similarity['Full Index'] == national_identifier]
        matched_pt_rows = pt_similarity[pt_similarity['Full Index'] == pt_identifier]

        matched_national_sentence = matched_national_rows[f'National in {string}'].values[0] if not matched_national_rows.empty else ''
        
        matched_pt_sentence = matched_pt_rows[f'{string} Variations'].values[0] if not matched_pt_rows.empty else ''

        # Keep the alignments if the cosine similarity is above the threshold
        if cosine_similarity >= alignment_threshold:
            final_alignments.append({
                'National Sentence Text': row['National Sentence Text'],
                'Matched National Sentence (Variation)': matched_national_sentence,
                'P/T Sentence Text': row['P/T Sentence Text'],
                'Matched P/T Sentence (Variation)': matched_pt_sentence,
                'Similarity': cosine_similarity
            })

        # Otherwise split up the alignments
        else:
            if row['P/T Sentence Text'] == '':
                final_alignments.append({
                    'National Sentence Text': row['National Sentence Text'],
                    'Matched National Sentence (Variation)': matched_national_sentence,
                    'P/T Sentence Text': '',
                    'Matched P/T Sentence (Variation)': '',
                    'Similarity': 0.0
                })
            elif row['National Sentence Text'] == '':
                final_alignments.append({
                    'National Sentence Text': '',
                    'Matched National Sentence (Variation)': '',
                    'P/T Sentence Text': row['P/T Sentence Text'],
                    'Matched P/T Sentence (Variation)': matched_pt_sentence,
                    'Similarity': 0.0
                })
            else:
                final_alignments.append({
                    'National Sentence Text': row['National Sentence Text'],
                    'Matched National Sentence (Variation)': matched_national_sentence,
                    'P/T Sentence Text': '',
                    'Matched P/T Sentence (Variation)': '',
                    'Similarity': 0.0
                })

                final_alignments.append({
                    'National Sentence Text': '',
                    'Matched National Sentence (Variation)': '',
                    'P/T Sentence Text': row['P/T Sentence Text'],
                    'Matched P/T Sentence (Variation)': matched_pt_sentence,
                    'Similarity': 0.0
                })

    final_alignments_df = pd.DataFrame(final_alignments)
    return final_alignments_df


# Function useful for filtering existing alignments using a higher threshold
def recursion_final_alignments(alignments_df, alignment_threshold):
    final_alignments = []
    for _, row in alignments_df.iterrows():
        if row['Similarity'] >= alignment_threshold:
            final_alignments.append({
                'National Sentence Text': row['National Sentence Text'],
                'Matched National Sentence (Variation)': row['Matched National Sentence (Variation)'],
                'P/T Sentence Text': row['P/T Sentence Text'],
                'Matched P/T Sentence (Variation)': row['Matched P/T Sentence (Variation)'],
                'Similarity': row['Similarity']
            })
        else:
            if row['P/T Sentence Text'] == '':
                final_alignments.append({
                    'National Sentence Text': row['National Sentence Text'],
                    'Matched National Sentence (Variation)': row['Matched National Sentence (Variation)'],
                    'P/T Sentence Text': '',
                    'Matched P/T Sentence (Variation)': '',
                    'Similarity': 0.0
                })
            elif row['National Sentence Text'] == '':
                final_alignments.append({
                'National Sentence Text': '',
                'Matched National Sentence (Variation)': '',
                'P/T Sentence Text': row['P/T Sentence Text'],
                'Matched P/T Sentence (Variation)': row['Matched P/T Sentence (Variation)'],
                'Similarity': 0.0
                })
            else:
                final_alignments.append({
                    'National Sentence Text': row['National Sentence Text'],
                    'Matched National Sentence (Variation)': row['Matched National Sentence (Variation)'],
                    'P/T Sentence Text': '',
                    'Matched P/T Sentence (Variation)': '',
                    'Similarity': 0.0
                })

                final_alignments.append({
                    'National Sentence Text': '',
                    'Matched National Sentence (Variation)': '',
                    'P/T Sentence Text': row['P/T Sentence Text'],
                    'Matched P/T Sentence (Variation)': row['Matched P/T Sentence (Variation)'],
                    'Similarity': 0.0
                })

    final_alignments_df = pd.DataFrame(final_alignments)
    return final_alignments_df


# Function to calculate the alignment error rate
def calculate_alignment_error_rate(alignments_df, test_df, train=False, pt=None, model_name=None):
    total_correct_alignments = 0
    total_predicted_alignments = 0
    total_actual_alignments = 0
    misalignments = []
    misalignments_test = []
    predicted_test_alignments = []

    for _, test_row in test_df.iterrows():
        matching_rows = []
        if test_row['Label'] == 'National Only':
            matching_rows = alignments_df[
                (alignments_df['Matched National Sentence (Variation)'] == test_row['National Sentence Text'])]
            precise_alignments = alignments_df[
                (alignments_df['Matched National Sentence (Variation)'] == test_row['National Sentence Text']) &
                ((alignments_df['P/T Sentence Text'] == '') | alignments_df['P/T Sentence Text'].isna())]
        elif test_row['Label'] == 'P/T Only':
            matching_rows = alignments_df[
                (alignments_df['Matched P/T Sentence (Variation)'] == test_row['P/T Sentence Text'])]
            precise_alignments = alignments_df[
                (alignments_df['Matched P/T Sentence (Variation)'] == test_row['P/T Sentence Text']) &
                ((alignments_df['National Sentence Text'] == '') | alignments_df['National Sentence Text'].isna())]
        else:
            matching_rows = alignments_df[
                (alignments_df['Matched National Sentence (Variation)'] == test_row['National Sentence Text']) |
                (alignments_df['Matched P/T Sentence (Variation)'] == test_row['P/T Sentence Text'])]
            precise_alignments = alignments_df[
                (alignments_df['Matched National Sentence (Variation)'] == test_row['National Sentence Text']) &
                (alignments_df['Matched P/T Sentence (Variation)'] == test_row['P/T Sentence Text'])]
            
        if not matching_rows.empty:
            total_predicted_alignments += len(matching_rows)

        if not precise_alignments.empty:
            total_correct_alignments += 1
        else:
            if (not train) and (model_name is not None) and (pt is not None):
                misalignments.append(matching_rows)
                misalignments_test.append(test_row.to_frame().T)

                if not matching_rows.empty:
                    predicted_test_alignments.append(matching_rows)
                    

                if misalignments:
                    all_misalignments_df = pd.concat(misalignments, ignore_index=True)
                    all_misalignments_df.to_csv(f'./baseline-data/misalignments/{pt}/{pt} {model_name} Misalignments.csv', index=False)

                if misalignments_test:
                    all_misalignments_test_df = pd.concat(misalignments_test, ignore_index=True)
                    all_misalignments_test_df.to_csv(f'./baseline-data/misalignments/{pt}/{pt} {model_name} Misalignments Test.csv', index=False)

        total_actual_alignments += 1

        if total_actual_alignments + total_predicted_alignments > 0:
            total_alignment_error_rate = 1 - ((2 * total_correct_alignments) / (total_predicted_alignments + total_actual_alignments))
        else:
            total_alignment_error_rate = float('inf')
            
    return total_alignment_error_rate


# #### Read P/T full code data into dataframes
def get_data(string):
    data = pd.read_csv(pt_full_2015 + f'/2015 DivB {string} Full.csv')
    data = data.drop_duplicates(subset=['PT Sentence Text'], keep='first').reset_index(drop=True)
    return data


# #### Implement the baseline models for each of the P/Ts
# Province/Territory names with their respective national similarity, pt similarity, and alignment thresholds
province_territories = {'AB': (0.7, 0.51), 'BC': (0, 0.84), 'NS': (0, 0.62), 'NU': (0, 0), 'ON': (0.55, 0.54), 'PE': (0, 0.85), 'SK': (0, 0.92)}
model_names = ['Bag of Words', 'TF-IDF', 'N-Hot']

national_df = pd.read_csv(national_full_path)


# Initialize the thresholds and error rates
thresholds = np.arange(0.0, 1.01, 0.01)
best_thresholds = {}

for pt, (national_threshold, pt_threshold) in province_territories.items():
    # Get the data for the province/territory
    pt_df = get_data(pt)

    # Get the matched similarities
    national_similarity = pd.read_csv(national_similarity_path + f'/2015 {pt} National Similarity.csv')
    pt_similarity = pd.read_csv(pt_similarity_path + f'/2015 {pt} PT Similarity.csv')

    national_similarity = national_similarity[national_similarity['Similarity'] >= national_threshold]
    pt_similarity = pt_similarity[pt_similarity['Similarity'] >= pt_threshold]

    # Get the cosine similarity matrices
    cosine_similarity_matrices = baseline_models(national_df, pt_df)

    # Load the train set
    train_df = pd.read_csv(train_test_sets + f'/{pt} Train.csv')
    train_df['Label'] = train_df.apply(map_labels, axis=1)

    # Initialize the best thresholds and error rates
    best_thresholds[pt] = {}

    print(f'{pt}:')
    
    plt.figure(figsize=(4, 3))

    # Calculate the optimal threshold for each model using train sets
    for model_name in model_names:
        cosine_similarity_matrix = cosine_similarity_matrices[model_name]
        similarity_df = get_matchings(cosine_similarity_matrix, national_df, pt_df)
        alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt, 0)

        error_rates = []

        # For each threshold, calculate the alignment error rate
        for threshold in thresholds:
            alignments_df = recursion_final_alignments(alignments_df, threshold)
            alignment_error_rate = calculate_alignment_error_rate(alignments_df, train_df)
            error_rates.append(alignment_error_rate)

        # Get the best threshold and error rate
        best_threshold = thresholds[np.argmin(error_rates)]
        best_thresholds[pt][model_name] = best_threshold

        plt.plot(thresholds, error_rates, label=model_name)

    plt.xlabel('Threshold', fontsize=7)
    plt.ylabel('Alignment Error Rate', fontsize=7)
    plt.title(f'AER vs. Threshold for {pt}', fontsize=9)
    plt.legend()
    plt.show()

print('Best Thresholds:')
print(best_thresholds)



# Use the optimal thresholds on the test sets
test_aers = {}
test_alignments = {}

for pt, (national_threshold, pt_threshold) in province_territories.items():
    # Get the data for the province/territory
    pt_df = get_data(pt)

    # Get the matched similarities
    national_similarity = pd.read_csv(national_similarity_path + f'/2015 {pt} National Similarity.csv')
    pt_similarity = pd.read_csv(pt_similarity_path + f'/2015 {pt} PT Similarity.csv')

    national_similarity = national_similarity[national_similarity['Similarity'] >= national_threshold]
    pt_similarity = pt_similarity[pt_similarity['Similarity'] >= pt_threshold]

    # Get the cosine similarity matrices
    cosine_similarity_matrices = baseline_models(national_df, pt_df)

    # Load the test set and map the labels
    test_df = pd.read_csv(train_test_sets + f'/{pt} Test.csv')
    test_df['Label'] = test_df.apply(map_labels, axis=1)

    # Initialize the best thresholds and error rates
    test_aers[pt] = {}
    test_alignments[pt] = {}

    print(f'{pt}:')

    # Calculate the optimal threshold for each model using train sets
    for model_name in model_names:
        cosine_similarity_matrix = cosine_similarity_matrices[model_name]
        similarity_df = get_matchings(cosine_similarity_matrix, national_df, pt_df)
        alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt, 0)

        best_threshold = best_thresholds[pt][model_name]

        # Use the best threshold on test set
        test_alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt, best_threshold)
        test_alignments[pt][model_name] = test_alignments_df

        test_aer = calculate_alignment_error_rate(test_alignments_df, test_df, train=False, pt=pt, model_name=model_name)
        test_aers[pt][model_name] = test_aer

        print(f'{model_name}:')
        print(f'  Best Threshold: {best_threshold}')
        print(f'  Test Alignment Error Rate: {test_aer}\n')

print('Best Thresholds:')
print(best_thresholds)
print('Test AERs:')
print(test_aers)