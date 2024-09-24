
# ## Import necessary libraries
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import linear_sum_assignment
from itertools import product, islice
from comet import download_model, load_from_checkpoint
from huggingface_hub import login


# Set logging level
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ## Load the files required for National and P/T Sentence Texts
# Paths to the data
full_code_2015 = './Data/2015/full/div-b'
pt_full_2015 = full_code_2015 + '/pt-full'
national_full_path = full_code_2015 + '/2015 National Full.csv' 

national_similarity_path = './Data/2015/old/old-national-similarity'
pt_similarity_path = './Data/2015/old/old-pt-similarity'

train_test_sets = './Data/2015/old/old-pt-train-test-sets'


# Function to load P/T data
def get_data(string):
    data = pd.read_csv(pt_full_2015 + f'/2015 DivB {string} Full.csv')
    data = data.drop_duplicates(subset=['PT Sentence Text'], keep='first').reset_index(drop=True)
    return data


# Function to map the labels
def map_labels(row):
    if row['Difference Type'] == 'Common Sentence':
        if row['Variation'] == 'Yes':
            return 'Variation'
        else:
            return 'Common Sentence'
    return row['Difference Type']

# Authenticate with Hugging Face 
login(token="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")

comet_model_path = download_model('Unbabel/wmt22-cometkiwi-da')
comet_model = load_from_checkpoint(comet_model_path, strict=False)

# Function to generate embeddings and compute cosine similarity
def generate_embeddings(national_df, pt_df, model_name):

    national_sentences = national_df['FRAG_DOCUMENT'][0:8].tolist()
    pt_sentences = pt_df['PT Sentence Text'][0:8].tolist()

    n = len(national_sentences)
    m = len(pt_sentences)
    scores_matrix = np.zeros((n, m))
    
    batch_size = 2
    
    for i, s1 in enumerate(national_sentences):
        seg_scores = []
        # Process pt_sentences in batches
        for batch_start in range(0, m, batch_size):
            batch_pt_sentences = pt_sentences[batch_start:batch_start+batch_size]
            data = [{"src": s1, "mt": s2} for s2 in batch_pt_sentences]
            outputs = comet_model.predict(data, progress_bar=False)
            seg_scores.extend(outputs['scores'])
        
        # Store the scores in the scores_matrix
        scores_matrix[i, :] = seg_scores
    
    return scores_matrix


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
def calculate_alignment_error_rate(alignments_df, test_df, train=False, pt=None, model_name=None):
    total_correct_alignments = 0
    total_predicted_alignments = 0
    total_actual_alignments = 0
    misalignments = []
    misalignments_test = []

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
            # If using threshold values to filter alignments for test data, save the misalignments
            if (not train) and (model_name is not None) and (pt is not None):
                parts = model_name.split('/')

                if len(parts) > 1:
                    model = parts[-1]

                misalignments.append(matching_rows)
                misalignments_test.append(test_row.to_frame().T)

                if misalignments:
                    all_misalignments_df = pd.concat(misalignments, ignore_index=True)
                    all_misalignments_df.to_csv(f'./comet-data/misalignments/{pt}/{pt} {model} Misalignments.csv', index=False)

                if misalignments_test:
                    all_misalignments_test_df = pd.concat(misalignments_test, ignore_index=True)
                    all_misalignments_test_df.to_csv(f'./comet-data/misalignments/{pt}/{pt} {model} Misalignments Test.csv', index=False)

        total_actual_alignments += 1

        if total_actual_alignments + total_predicted_alignments > 0:
            total_alignment_error_rate = 1 - ((2 * total_correct_alignments) / (total_predicted_alignments + total_actual_alignments))
        else:
            total_alignment_error_rate = float('inf')
            
    return total_alignment_error_rate


# Province/Territory names with their respective national similarity, pt similarity, and alignment thresholds
province_territories = {'AB': (0.7, 0.51), 'BC': (0, 0.84), 'NS': (0, 0.62), 'NU': (0, 0), 'ON': (0.55, 0.54), 'PE': (0, 0.85), 'SK': (0, 0.92)}
# province_territories = {'NS': (0, 0.62), 'NU': (0, 0), 'PE': (0, 0.85)}

national_df = pd.read_csv(national_full_path)


def implement_model(model_name):
    parts = model_name.split('/')

    if len(parts) > 1:
        model = parts[-1]

    # Define the directory to save the graphs
    graph_dir = './comet-data/graphs'
    
    # Initialize the thresholds and error rates
    thresholds = np.arange(0.0, 1.01, 0.01)
    test_aers = {}
    best_thresholds = {}
    test_alignments = {}

    for pt, (national_threshold, pt_threshold) in province_territories.items():
        # Get the data for the province/territory
        pt_df = get_data(pt)

        # Get the matched similarities
        national_similarity = pd.read_csv(national_similarity_path + f'/2015 {pt} National Similarity.csv')
        pt_similarity = pd.read_csv(pt_similarity_path + f'/2015 {pt} PT Similarity.csv')

        national_similarity = national_similarity[national_similarity['Similarity'] >= national_threshold]
        pt_similarity = pt_similarity[pt_similarity['Similarity'] >= pt_threshold]

        # Get the cosine similarity matrix
        comet_scores = generate_embeddings(national_df, pt_df, model_name)

        # Load the train set
        train_df = pd.read_csv(train_test_sets + f'/{pt} Train.csv')
        test_df = pd.read_csv(train_test_sets + f'/{pt} Test.csv')

        train_df['Label'] = train_df.apply(map_labels, axis=1)
        test_df['Label'] = test_df.apply(map_labels, axis=1)

        print(f'{pt}:')
        
        plt.figure(figsize=(8, 6))

        
        similarity_df = get_matchings(comet_scores, national_df, pt_df)
        alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt, 0)

        error_rates = []

        # For each threshold, calculate the alignment error rate
        for threshold in thresholds:
            alignments_df = recursion_final_alignments(alignments_df, threshold)
            alignment_error_rate = calculate_alignment_error_rate(alignments_df, train_df)
            error_rates.append(alignment_error_rate)

        # # Find the minimum alignment error rate
        # min_error_rate = min(error_rates)

        # # Get the best threshold (last occurence of the minimum alignment error rate)
        # best_thresholds = [thresholds[i] for i, error_rate in enumerate(error_rates) if error_rate == min_error_rate]
        # best_threshold = best_thresholds[-1]

        # # Get the best threshold (first occurence of the minimum alignment error rate)
        best_threshold = thresholds[np.argmin(error_rates)]
        best_thresholds[pt] = best_threshold

        # Use the best threshold on test set
        test_alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt, best_threshold)
        test_alignments[pt] = test_alignments_df

        test_aer = calculate_alignment_error_rate(test_alignments_df, test_df, train=False, pt=pt, model_name=model_name)
        test_aers[pt] = test_aer

        print(f'{model_name}:')
        print(f'  Best Threshold: {best_threshold}')
        print(f'  Test Alignment Error Rate: {test_aer}\n')

        plt.plot(thresholds, error_rates)

        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Alignment Error Rate', fontsize=12)
        plt.title(f'COMET AER vs. Threshold for {pt}', fontsize=14)
        
        # Save the graph
        plt.savefig(graph_dir + f'/{pt} {model} AER vs. Threshold.png', dpi=500)
        plt.close()


    return best_thresholds, test_aers, test_alignments


def save_outputs(best_threshold, test_aers, test_alignments, model_name):
    output_data = {
        'best_threshold': best_threshold,
        'test_aers': test_aers,
        'test_alignments': {k: v.to_dict() for k, v in test_alignments.items()}  # Convert DataFrames to dictionaries
    }

    # Define the output path
    output_file = f'./comet-data/results/{model_name.split("/")[-1]} Results.json'
    
    # Save the data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)


comet_best_threshold, comet_test_aers, comet_test_alignments = implement_model('Unbabel/wmt22-cometkiwi-da')

# Save the results to a JSON file
save_outputs(comet_best_threshold, comet_test_aers, comet_test_alignments, 'Unbabel/wmt22-cometkiwi-da')