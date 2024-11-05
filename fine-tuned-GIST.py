# Import the necessary libraries
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader


# # Declare variables and parameters
# model_name = 'avsolatorio/GIST-Embedding-v0'
# batch_size = 4
# num_epochs = 1
# SEED = 42


# # Load the data
# pt_names = ['AB', 'BC', 'NS', 'NU', 'ON', 'PE', 'SK']

# # Read the data 
# train_set_path = './Data/2015/pt-train-test-sets'

# # Initialize an empty list to hold all InputExamples
# train_examples = []
# val_datasets = {}

# for pt in pt_names:
#     train_data = pd.read_csv(f'{train_set_path}/{pt} Train.csv')
#     train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=SEED)
#     val_datasets[pt] = val_data

#     national_only_sentences = []
#     pt_only_sentences = []

#     for _, row in train_data.iterrows():
#         # Create positive pairs (similar sentences)
#         if row['Difference Type'] == 'Common Sentence':
#             train_examples.append(InputExample(texts=[row['National Sentence Text'], row['P/T Sentence Text']], label=1))

#         elif row['Difference Type'] == 'National Only':
#             national_only_sentences.append(row['National Sentence Text'])
#         elif row['Difference Type'] == 'P/T Only':
#             pt_only_sentences.append(row['P/T Sentence Text'])

#     # Create negative pairs by combining National Only and P/T Only sentences
#     min_length = min(len(national_only_sentences), len(pt_only_sentences))
#     for i in range(min_length):
#         train_examples.append(InputExample(texts=[national_only_sentences[i], pt_only_sentences[i]], label=0))

# print(len(train_examples))


# # Initialize the SentenceTransformer model
# model = SentenceTransformer(model_name)

# # Create a DataLoader for the training data
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
# train_loss = losses.ContrastiveLoss(model)


# # Fine-tune the model
# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=num_epochs, 
#     output_path='./transformers-data/gist-embedding-v0-fine-tuned',
#     show_progress_bar=True
# )


model = SentenceTransformer('./gist-embedding-v0-fine-tuned')

sentence1 = "Means of egress shall be provided in buildings in conformance with the NBC."
sentence2 = "Means of egress shall be provided in buildings in conformance with the British Columbia Building Code."

embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity([embedding1], [embedding2])


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


# Function to generate embeddings and compute cosine similarity
def generate_embeddings(national_df, pt_df, model):
    model = SentenceTransformer(model)
    
    # Generate embeddings
    national_embeddings = model.encode(national_df['FRAG_DOCUMENT'].tolist(), convert_to_tensor=True)
    pt_embeddings = model.encode(pt_df['PT Sentence Text'].tolist(), convert_to_tensor=True)

    # Compute cosine similarity
    # similarity_matrix = cosine_similarity(national_embeddings.cpu(), pt_embeddings.cpu())
    similarity_matrix = util.pytorch_cos_sim(national_embeddings, pt_embeddings)

    return similarity_matrix


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
                    all_misalignments_df.to_csv(f'./finetuned-data/misalignments/{pt}/{pt} {code_book} Fine-tuned Misalignments.csv', index=False)

                if misalignments_test:
                    all_misalignments_test_df = pd.concat(misalignments_test, ignore_index=True)
                    all_misalignments_test_df.to_csv(f'./finetuned-data/misalignments/{pt}/{pt} {code_book} Fine-tuned Misalignments Test.csv', index=False)

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


# Province/Territory names with their respective national similarity, pt similarity, and alignment thresholds
province_territories = {'AB': (0.7, 0.51), 'BC': (0, 0.84), 'NS': (0, 0.62), 'NU': (0, 0), 'ON': (0.55, 0.54), 'PE': (0, 0.85), 'SK': (0, 0.92)}


def implement_model(model_name):
    test_sets = {'combined': ['National', 'ON'],
                 'building': ['National', 'AB', 'BC', 'NS', 'NU', 'PE', 'SK'],
                 'fire': ['National', 'AB', 'BC', 'NS', 'ON', 'SK'],
                 'plumbing': ['National', 'BC', 'NU', 'PE']}
    parts = model_name.split('/')

    if len(parts) > 1:
        model = parts[-1]

    else:
        model = model_name

    # Define the directory to save the graphs
    graph_dir = f'./finetuned-data/graphs'
    os.makedirs(graph_dir, exist_ok=True)
    
    # Initialize the thresholds and error rates
    thresholds = np.arange(0.0, 1.01, 0.01)
    test_alignments = {}
    aer_results = {}

    total_correct_alignments = 0
    total_predicted_alignments = 0
    total_actual_alignments = 0

    total_train_correct_alignments = 0
    total_train_predicted_alignments = 0
    total_train_actual_alignments = 0

    for code_type, provinces in test_sets.items():
        national_df = pd.read_csv(f'{full_code_2015}/{code_type}/2015 {provinces[0]} {code_type.capitalize()}.csv')
        for pt_code in provinces[1:]:
            pt_df = pd.read_csv(f'{full_code_2015}/{code_type}/2015 {pt_code} {code_type.capitalize()}.csv')

            # Get the matched similarities
            national_similarity = pd.read_csv(national_similarity_path + f'/2015 {pt_code} National Similarity.csv')
            pt_similarity = pd.read_csv(pt_similarity_path + f'/2015 {pt_code} PT Similarity.csv')

            # Get the thresholds for the province/territory
            national_threshold, pt_threshold = province_territories[pt_code]

            national_similarity = national_similarity[national_similarity['Similarity'] >= national_threshold]
            pt_similarity = pt_similarity[pt_similarity['Similarity'] >= pt_threshold]

            # Get the similarity score matrix
            similarity_matrix = generate_embeddings(national_df, pt_df, model_name)

            # Load the train set
            train_df = pd.read_csv(train_test_sets + f'/{pt_code} Train.csv')
            test_df = pd.read_csv(train_test_sets + f'/{pt_code} Test.csv')

            train_df['Label'] = train_df.apply(map_labels, axis=1)
            test_df['Label'] = test_df.apply(map_labels, axis=1)

            print(f'{pt_code} {code_type.capitalize()}:')
            
            plt.figure(figsize=(8, 6))

            
            similarity_df = get_matchings(similarity_matrix, national_df, pt_df)
            alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt_code, 0)

            error_rates = []

            # For each threshold, calculate the alignment error rate
            for threshold in thresholds:
                alignments_df = recursion_final_alignments(alignments_df, threshold)
                alignment_error_rate, c, p, a = calculate_alignment_error_rate(alignments_df, train_df, code_book=f'{code_type.capitalize()}', train=True, pt=f'{pt_code}', model_name=model_name)
                error_rates.append(alignment_error_rate)

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
            test_alignments_df = get_alignments(similarity_df, national_similarity, pt_similarity, pt_code, best_threshold)
            test_alignments[f'{pt_code} {code_type.capitalize()}'] = test_alignments_df


            aer, correct_alignments, predicted_alignments, actual_alignments = calculate_alignment_error_rate(test_alignments_df, test_df, code_book=f'{code_type.capitalize()}', train=False, pt=f'{pt_code}', model_name=model_name)
            train_aer, train_correct_alignments, train_predicted_alignments, train_actual_alignments = calculate_alignment_error_rate(test_alignments_df, train_df, code_book=f'{code_type.capitalize()}', train=True)
            
            # Store the results in the dictionary
            aer_results[f'{pt_code} {code_type.capitalize()}'] = {
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
            plt.title(f'{model_name} AER vs. Threshold for {pt_code} {code_type.capitalize()}', fontsize=14)
            
            # Save the graph
            plt.savefig(graph_dir + f'/{pt_code} {code_type.capitalize()} Fine-tuned AER vs. Threshold.png', dpi=500)
            plt.close()

    total_test_aer = calculate_aer(total_correct_alignments, total_predicted_alignments, total_actual_alignments)
    print(f'Total Test AER: {total_test_aer}\n')

    total_train_aer = calculate_aer(total_train_correct_alignments, total_train_predicted_alignments, total_train_actual_alignments)
    print(f'Total Train AER: {total_train_aer}\n')

    aer_results['Total Test AER'] = total_test_aer
    aer_results['Total Train AER'] = total_train_aer
    
    # Save the aer_results dictionary to a JSON file
    results_file = f'./finetuned-data/results/Fine-tuned AER Results.json'
    with open(results_file, 'w') as f:
        json.dump(aer_results, f, indent=4)

    return test_alignments, total_test_aer, total_train_aer, aer_results


finetuned_alignments, finetuned_test_aer, finetuned_train_aer, finetuned_results = implement_model('./gist-embedding-v0-fine-tuned')

# Save all the alignments from BERTScore
for key, value in finetuned_alignments.items():
    value.to_csv(f'./finetuned-data/results/alignments/{key} Alignments.csv', index=False)


