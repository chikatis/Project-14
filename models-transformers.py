# ## Import necessary libraries

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util


# ## Load the files required for National and P/T Sentence Texts

# Paths to the data
full_code_2015 = './split-up'

national_similarity_path = './Data/2015/old/old-national-similarity'
pt_similarity_path = './Data/2015/old/old-pt-similarity'

train_test_sets = './Data/2015/old/old-pt-train-test-sets'

# Function to map the labels
def map_labels(row):
    if row['Difference Type'] == 'Common Sentence':
        if row['Variation'] == 'Yes':
            return 'Variation'
        else:
            return 'Common Sentence'
    return row['Difference Type']

# Function to generate similarity scores
def generate_scores(national_df, pt_df, model_name):

    if model_name == 'Lajavaness/bilingual-embedding-large':
        model = SentenceTransformer(model_name, trust_remote_code=True)
    else:
        model = SentenceTransformer(model_name)

    # Generate embeddings
    national_embeddings = model.encode(national_df['FRAG_DOCUMENT'].tolist(), convert_to_tensor=True)
    pt_embeddings = model.encode(pt_df['PT Sentence Text'].tolist(), convert_to_tensor=True)

    # Compute cosine similarity
    # similarity_matrix = cosine_similarity(national_embeddings.cpu(), pt_embeddings.cpu())
    scores_matrix = util.pytorch_cos_sim(national_embeddings, pt_embeddings)

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

    # Drop the corresponding rows and columns from the scores matrix
    scores_matrix = np.delete(scores_matrix, national_duplicate_indices, axis=0)
    scores_matrix = np.delete(scores_matrix, pt_duplicate_indices, axis=1)

    return scores_matrix, national_df, pt_df



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
def calculate_alignment_error_rate(alignments_df, test_df, code_book, train=False, pt=None, model_name=None, model_type=None):
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
            if (not train) and (model_name is not None) and (pt is not None) and (model_type is not None):
                parts = model_name.split('/')

                if len(parts) > 1:
                    model = parts[-1]
                else:
                    model = model_name

                misalignments.append(matching_rows)
                misalignments_test.append(test_row.to_frame().T)

                if misalignments:
                    all_misalignments_df = pd.concat(misalignments, ignore_index=True)
                    all_misalignments_df.to_csv(f'./{model_type.lower()}-data/{model_name}/misalignments/{pt}/{pt} {code_book} {model} Misalignments.csv', index=False)

                if misalignments_test:
                    all_misalignments_test_df = pd.concat(misalignments_test, ignore_index=True)
                    all_misalignments_test_df.to_csv(f'./{model_type.lower()}-data/{model_name}/misalignments/{pt}/{pt} {code_book} {model} Misalignments Test.csv', index=False)

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

def implement_model(model_type, model_name):
    parts = model_name.split('/')

    if len(parts) > 1:
        model = parts[-1]

    else:
        model = model_name

    # Define the directory to save the graphs
    graph_dir = f'./{model_type.lower()}-data/{model.lower()}/graphs'
    os.makedirs(graph_dir, exist_ok=True)
    
    # Initialize the thresholds and error rates
    thresholds = np.arange(0.0, 1.01, 0.01)
    # test_aers = {}
    # best_thresholds = {}
    test_alignments = {}
    aer_results = {}

    total_correct_alignments = 0
    total_predicted_alignments = 0
    total_actual_alignments = 0

    total_train_correct_alignments = 0
    total_train_predicted_alignments = 0
    total_train_actual_alignments = 0


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
                    similarity_score_matrix, national_df, pt_df = generate_scores(national_df, pt_df, model_name)

                    # Load the train set
                    train_df = pd.read_csv(train_test_sets + f'/{pt} Train.csv')
                    test_df = pd.read_csv(train_test_sets + f'/{pt} Test.csv')

                    train_df['Label'] = train_df.apply(map_labels, axis=1)
                    test_df['Label'] = test_df.apply(map_labels, axis=1)

                    print(f'{pt} {code_type.capitalize()}:')
                    
                    plt.figure(figsize=(8, 6))
                    
                    similarity_df = get_matchings(similarity_score_matrix, national_df, pt_df)
                    alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt, 0)

                    error_rates = []

                    # For each threshold, calculate the alignment error rate
                    for threshold in thresholds:
                        alignments_df = recursion_final_alignments(alignments_df, threshold)
                        alignment_error_rate, c, p, a = calculate_alignment_error_rate(alignments_df, train_df, code_book=f'{code_type.capitalize()}', train=True, pt=f'{pt}', model_name=model, model_type=model_type)
                        error_rates.append(alignment_error_rate)

                    # Find the minimum alignment error rate
                    min_error_rate = min(error_rates)

                    # Get the best threshold (last occurence of the minimum alignment error rate)
                    best_threshold = [thresholds[i] for i, error_rate in enumerate(error_rates) if error_rate == min_error_rate]
                    best_threshold = best_threshold[-1]

                    # # # Get the best threshold (first occurence of the minimum alignment error rate)
                    # best_threshold = thresholds[np.argmin(error_rates)]

                    # Store the training AER
                    train_aer = min_error_rate

                    # Use the best threshold on test set
                    test_alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt, best_threshold)
                    test_alignments[f'{pt} {code_type.capitalize()}'] = test_alignments_df


                    aer, correct_alignments, predicted_alignments, actual_alignments = calculate_alignment_error_rate(test_alignments_df, test_df, code_book=f'{code_type.capitalize()}', train=False, pt=f'{pt}', model_name=model_name)
                    train_aer, train_correct_alignments, train_predicted_alignments, train_actual_alignments = calculate_alignment_error_rate(test_alignments_df, train_df, code_book=f'{code_type.capitalize()}', train=True)
                    
                    # Store the results in the dictionary
                    aer_results[f'{pt} {code_type.capitalize()}'] = {
                        'Train AER': train_aer,
                        'Threshold': best_threshold,
                        'Test AER': aer
                    }

                    if aer == float('inf'):
                        total_correct_alignments += 0
                        total_predicted_alignments += 0
                        total_actual_alignments += 0
                    else:
                        total_correct_alignments += correct_alignments
                        total_predicted_alignments += predicted_alignments
                        total_actual_alignments += actual_alignments


                    if train_aer == float('inf'):
                        total_train_correct_alignments += 0
                        total_train_predicted_alignments += 0
                        total_train_actual_alignments += 0
                    else:
                        total_train_correct_alignments += train_correct_alignments
                        total_train_predicted_alignments += train_predicted_alignments
                        total_train_actual_alignments += train_actual_alignments

                    print(f'{model_name}:')
                    print(f'  Train AER: {train_aer}')
                    print(f'  Best Threshold: {best_threshold}')
                    print(f'  Test AER: {aer}\n')

                    plt.plot(thresholds, error_rates)

                    plt.xlabel('Threshold', fontsize=12)
                    plt.ylabel('Alignment Error Rate', fontsize=12)
                    plt.xlim(-0.1, 1.1)
                    plt.ylim(-0.1, 1.1)
                    plt.title(f'{model} AER vs. Threshold for {pt} {code_type.capitalize()}', fontsize=14)
                    
                    # Save the graph
                    plt.savefig(graph_dir + f'/{pt} {code_type.capitalize()} {model} AER vs. Threshold.png', dpi=500)
                    plt.close()

    total_test_aer = calculate_aer(total_correct_alignments, total_predicted_alignments, total_actual_alignments)
    print(f'Total Test AER: {total_test_aer}\n')

    total_train_aer = calculate_aer(total_train_correct_alignments, total_train_predicted_alignments, total_train_actual_alignments)
    print(f'Total Train AER: {total_train_aer}\n')

    aer_results['Total Test AER'] = total_test_aer
    aer_results['Total Train AER'] = total_train_aer

    # Save the aer_results dictionary to a JSON file
    results_file = f'./{model_type.lower()}-data/{model}/results/{model} AER Results.json'
    with open(results_file, 'w') as f:
        json.dump(aer_results, f, indent=4)

    return test_alignments, total_test_aer, aer_results

model_names = ['Lajavaness/bilingual-embedding-large', 'intfloat/multilingual-e5-large-instruct', 'mixedbread-ai/mxbai-embed-large-v1', 'avsolatorio/GIST-Embedding-v0']

bilingual_test_alignments, bilingual_total_aer, bilingual_aer_results = implement_model("transformers", model_names[0])

# Save all the alignments from Bilingual model
for key, value in bilingual_test_alignments.items():
    value.to_csv(f'./transformers-data/bilingual-embedding-large/results/alignments/{key} Alignments.csv', index=False)

multilingual_test_alignments, multilingual_total_aer, multiingual_aer_results = implement_model("transformers", model_names[1])

# Save all the alignments from the Multilingual model
for key, value in multilingual_test_alignments.items():
    value.to_csv(f'./transformers-data/multilingual-e5-large-instruct/results/{key} Alignments.csv', index=False)

mxbai_test_alignments, mxbai_total_aer, mxbai_aer_results = implement_model("transformers", model_names[2])

# Save all the alignments from the MxBAI model
for key, value in mxbai_test_alignments.items():
    value.to_csv(f'./transformers-data/mxbai-embed-large-v1/results/{key} Alignments.csv', index=False)


gist_test_alignments, gist_total_aer, gist_aer_results = implement_model("transformers", model_names[3])

# Save all the alignments from the GIST model
for key, value in gist_test_alignments.items():
    value.to_csv(f'./transformers-data/GIST-Embedding-v0/results/{key} Alignments.csv', index=False)
