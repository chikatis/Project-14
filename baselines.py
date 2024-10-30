
# ## Import necessary libraries


import re
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize.punkt import PunktSentenceTokenizer
import nltk
nltk.download('punkt')


# ## Load the files required for National and P/T Sentence Texts


# Paths to the data
full_code_2015 = './split-up'

national_similarity_path = './Data/2015/old/old-national-similarity'
pt_similarity_path = './Data/2015/old/old-pt-similarity'

train_test_sets = './Data/2015/old/old-pt-train-test-sets'

baselines = './baseline-data'


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

    # Identify and drop duplicates from national_df
    national_duplicates = national_df[national_df.duplicated(subset='FRAG_DOCUMENT', keep='first')]
    national_duplicate_indices = national_duplicates.index.tolist()
    national_df = national_df.drop_duplicates(subset='FRAG_DOCUMENT', keep='first')

    # Identify and drop duplicates from pt_df
    pt_duplicates = pt_df[pt_df.duplicated(subset='PT Sentence Text', keep='first')]
    pt_duplicate_indices = pt_duplicates.index.tolist()
    pt_df = pt_df.drop_duplicates(subset='PT Sentence Text', keep='first')
    
    # Adjust indices 
    national_duplicate_indices = [i for i in national_duplicate_indices]
    pt_duplicate_indices = [i for i in pt_duplicate_indices]

    # Drop the corresponding rows and columns from all the matrices
    cosine_similarity_bow = np.delete(cosine_similarity_bow, national_duplicate_indices, axis=0)
    cosine_similarity_bow = np.delete(cosine_similarity_bow, pt_duplicate_indices, axis=1)

    cosine_similarity_tfidf = np.delete(cosine_similarity_tfidf, national_duplicate_indices, axis=0)
    cosine_similarity_tfidf = np.delete(cosine_similarity_tfidf, pt_duplicate_indices, axis=1)

    cosine_similarity_n_hot = np.delete(cosine_similarity_n_hot, national_duplicate_indices, axis=0)
    cosine_similarity_n_hot = np.delete(cosine_similarity_n_hot, pt_duplicate_indices, axis=1)

    cosine_similarity_matrices = {
        'Bag of Words': cosine_similarity_bow,
        'TF-IDF': cosine_similarity_tfidf,
        '1-Hot': cosine_similarity_n_hot
    }

    return cosine_similarity_matrices, national_df, pt_df



# Function to get the best possible matches using the Hungarian algorithm
def get_matchings(similarity_matrix, national_df, pt_df):
    national_ind, pt_ind = linear_sum_assignment(similarity_matrix, maximize=True)

    unmatched_similarity_list = []

    # Store the unmatched indices
    unmatched_national_ind = [i for i in range(len(national_df)) if i not in national_ind]
    unmatched_pt_ind = [j for j in range(len(pt_df)) if j not in pt_ind]

    # Add unmatched sentences to the similarity list
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

    # Add matched sentences to the similarity list
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
        # Keep the alignments if the cosine similarity is above the threshold
        if row['Similarity'] >= alignment_threshold:
            final_alignments.append({
                'National Sentence Text': row['National Sentence Text'],
                'Matched National Sentence (Variation)': row['Matched National Sentence (Variation)'],
                'P/T Sentence Text': row['P/T Sentence Text'],
                'Matched P/T Sentence (Variation)': row['Matched P/T Sentence (Variation)'],
                'Similarity': row['Similarity']
            })
        # Otherwise split up the alignments
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
def calculate_alignment_error_rate(alignments_df, test_df, code_book, train=False, pt=None, model_name=None):
    misalignments = []
    misalignments_test = []

    correct_alignments = 0
    predicted_alignments = 0
    actual_alignments = 0

    if code_book == 'Combined':
        test_df = test_df[(test_df['Code Type'] == 'Building') | (test_df['Code Type'] == 'Plumbing') | (test_df['Code Type'] == 'Energy')]
    else:
        test_df = test_df[test_df['Code Type'] == code_book]

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
            predicted_alignments += len(matching_rows)

        if not precise_alignments.empty:
            correct_alignments += 1
        else:
            # If using threshold values to filter alignments for test data, save the misalignments
            if (not train) and (model_name is not None) and (pt is not None):
                parts = model_name.split('/')

                if len(parts) > 1:
                    model = parts[-1]
                else:
                    model = model_name

                misalignments.append(matching_rows)
                misalignments_test.append(test_row.to_frame().T)

                if misalignments:
                    all_misalignments_df = pd.concat(misalignments, ignore_index=True)
                    all_misalignments_df.to_csv(f'./baseline-data/{model_name}/misalignments/{pt}/{pt} {code_book} {model} Misalignments.csv', index=False)

                if misalignments_test:
                    all_misalignments_test_df = pd.concat(misalignments_test, ignore_index=True)
                    all_misalignments_test_df.to_csv(f'./baseline-data/{model_name}/misalignments/{pt}/{pt} {code_book} {model} Misalignments Test.csv', index=False)

        actual_alignments += 1


    if actual_alignments + predicted_alignments > 0:
        aer = 1 - ((2 * correct_alignments) / (predicted_alignments + actual_alignments))
    else:
        aer = float('inf')
            
    return aer, correct_alignments, predicted_alignments, actual_alignments


def calculate_aer(correct, predicted, actual):
    if actual + predicted > 0:
        aer = 1 - ((2 * correct) / (predicted + actual))
    else:
        aer = float('inf')
    return aer


# Province/Territory names with their respective national similarity and pt similarity thresholds
province_territories = {'AB': (0.7, 0.51), 'BC': (0, 0.84), 'NS': (0, 0.62), 'NU': (0, 0), 'ON': (0.55, 0.54), 'PE': (0, 0.85), 'SK': (0, 0.92)}


def implement_model(model_type):
    parts = model_type.split('/')

    if len(parts) > 1:
        model = parts[-1]

    else:
        model = model_type

    # Define the directory to save the graphs
    graph_dir = f'./{model_type.lower()}-data/graphs'
    os.makedirs(graph_dir, exist_ok=True)

    
    # Initialize the thresholds and error rates
    thresholds = np.arange(0.0, 1.01, 0.01)
    test_alignments = {}
    aer_results = {}

    test_alignments['Bag of Words'] = {}
    test_alignments['TF-IDF'] = {}
    test_alignments['1-Hot'] = {}

    aer_results['Bag of Words'] = {}
    aer_results['TF-IDF'] = {}
    aer_results['1-Hot'] = {}

    bow_correct_alignments = 0
    bow_predicted_alignments = 0
    bow_actual_alignments = 0

    tfidf_correct_alignments = 0
    tfidf_predicted_alignments = 0
    tfidf_actual_alignments = 0

    one_hot_correct_alignments = 0
    one_hot_predicted_alignments = 0
    one_hot_actual_alignments = 0

    for path, folders, files in os.walk(full_code_2015):
        for folder in folders:
            code_type = folder
            national_df = pd.read_csv(f'{full_code_2015}/{code_type}/2015 National {code_type.capitalize()}.csv')

            for file in os.listdir(f'{full_code_2015}/{code_type}'):
                if file != f'2015 National {code_type.capitalize()}.csv':
                    pt_df = pd.read_csv(f'{full_code_2015}/{code_type}/{file}')
                    pt = file.split(' ')[1]

                    # Get the matched similarities
                    national_similarity = pd.read_csv(national_similarity_path + f'/2015 {pt} National Similarity.csv')
                    pt_similarity = pd.read_csv(pt_similarity_path + f'/2015 {pt} PT Similarity.csv')

                    # Get the thresholds for the province/territory
                    national_threshold, pt_threshold = province_territories[pt]

                    national_similarity = national_similarity[national_similarity['Similarity'] >= national_threshold]
                    pt_similarity = pt_similarity[pt_similarity['Similarity'] >= pt_threshold]

                    # Get the similarity score matrix
                    similarity_score_matrices, national_df, pt_df = baseline_models(national_df, pt_df)

                    print(f'{pt} {code_type.capitalize()}:')
                        
                    plt.figure(figsize=(8, 6))

                    for model_name in similarity_score_matrices:
                        print(f'    {model_name}:')

                        similarity_score_matrix = similarity_score_matrices[model_name]

                        # Load the train set
                        train_df = pd.read_csv(train_test_sets + f'/{pt} Train.csv')
                        test_df = pd.read_csv(train_test_sets + f'/{pt} Test.csv')

                        train_df['Label'] = train_df.apply(map_labels, axis=1)
                        test_df['Label'] = test_df.apply(map_labels, axis=1)

                        
                        similarity_df = get_matchings(similarity_score_matrix, national_df, pt_df)
                        alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt, 0)

                        error_rates = []

                        # For each threshold, calculate the alignment error rate
                        for threshold in thresholds:
                            alignments_df = recursion_final_alignments(alignments_df, threshold)
                            alignment_error_rate, c, p, a = calculate_alignment_error_rate(alignments_df, train_df, code_book=f'{code_type.capitalize()}', train=True, pt=f'{pt}', model_name=model_name)
                            error_rates.append(alignment_error_rate)

                        plt.plot(thresholds, error_rates, label=model_name)

                        # Find the minimum alignment error rate
                        min_error_rate = min(error_rates)

                        # Get the best threshold (last occurence of the minimum alignment error rate)
                        best_threshold = [thresholds[i] for i, error_rate in enumerate(error_rates) if error_rate == min_error_rate]
                        best_threshold = best_threshold[-1]

                        # # # Get the best threshold (first occurence of the minimum alignment error rate)
                        # best_threshold = thresholds[np.argmin(error_rates)]
                        # best_thresholds[pt_code] = best_threshold

                        # Store the training AER
                        train_aer = min_error_rate
                        
                        # Use the best threshold on test set
                        test_alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt, best_threshold)
                        test_alignments[model_name][f'{pt} {code_type.capitalize()}'] = test_alignments_df


                        aer, correct_alignments, predicted_alignments, actual_alignments = calculate_alignment_error_rate(test_alignments_df, test_df, code_book=f'{code_type.capitalize()}', train=False, pt=f'{pt}', model_name=model_name)
                        
                        # Store the results in the dictionary
                        aer_results[model_name][f'{pt} {code_type.capitalize()}'] = {
                            'Train AER': train_aer,
                            'Threshold': best_threshold,
                            'Test AER': aer
                        }   
                        
                        if model_name == '1-Hot':
                            one_hot_correct_alignments += correct_alignments
                            one_hot_predicted_alignments += predicted_alignments
                            one_hot_actual_alignments += actual_alignments

                        elif model_name == 'TF-IDF':
                            tfidf_correct_alignments += correct_alignments
                            tfidf_predicted_alignments += predicted_alignments
                            tfidf_actual_alignments += actual_alignments

                        elif model_name == 'Bag of Words':
                            bow_correct_alignments += correct_alignments
                            bow_predicted_alignments += predicted_alignments
                            bow_actual_alignments += actual_alignments
                            

                        print(f'        Train AER: {train_aer}')
                        print(f'        Best Threshold: {best_threshold}')
                        print(f'        Test AER: {aer}\n')

                    plt.xlabel('Threshold', fontsize=12)
                    plt.ylabel('Alignment Error Rate', fontsize=12)
                    plt.xlim(-0.1, 1.1)
                    plt.ylim(-0.1, 1.1)
                    plt.title(f'AER vs. Threshold for {pt} {code_type.capitalize()}', fontsize=14)
                    plt.legend()

                    # Save the graph
                    plt.savefig(graph_dir + f'/{pt} {code_type.capitalize()} AER vs. Threshold.png', dpi=500)
                    plt.close()     
                    # plt.show()

    bow_aer = calculate_aer(bow_correct_alignments, bow_predicted_alignments, bow_actual_alignments)
    print(f'BoW Alignment Error Rate: {bow_aer}\n')

    tfidf_aer = calculate_aer(tfidf_correct_alignments, tfidf_predicted_alignments, tfidf_actual_alignments)
    print(f'TF-IDF Alignment Error Rate: {tfidf_aer}\n')

    one_hot_aer = calculate_aer(one_hot_correct_alignments, one_hot_predicted_alignments, one_hot_actual_alignments)
    print(f'1-Hot Alignment Error Rate: {one_hot_aer}\n')

    # Save the aer_results dictionary to a JSON file
    bow_results_file = f'./{model_type.lower()}-data/Bag of Words/results/{model_name} AER Results.json'
    with open(bow_results_file, 'w') as f:
        json.dump(aer_results['Bag of Words'], f, indent=4)

    tfidf_results_file = f'./{model_type.lower()}-data/TF-IDF/results/{model_name} AER Results.json'
    with open(tfidf_results_file, 'w') as f:
        json.dump(aer_results['TF-IDF'], f, indent=4)

    one_hot_results_file = f'./{model_type.lower()}-data/1-Hot/results/{model_name} AER Results.json'
    with open(one_hot_results_file, 'w') as f:
        json.dump(aer_results['1-Hot'], f, indent=4)

    return test_alignments, bow_aer, tfidf_aer, one_hot_aer, aer_results


baseline_test_alignments, bow_aer, tfidf_aer, one_hot_aer, aer_results = implement_model('Baseline')


for model_name in baseline_test_alignments:
    for pt_code in baseline_test_alignments[model_name]:
        test_alignments_df = baseline_test_alignments[model_name][pt_code]
        test_alignments_df.to_csv(f'./baseline-data/{model_name}/results/alignments/{pt_code} {model_name} Alignments.csv', index=False)


