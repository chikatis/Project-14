
# ## Import necessary libraries
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# ## Load the files required for National and P/T Sentence Texts
# Paths to the data
full_code_2015 = './sorted-split-up'

national_similarity_path = './Data/2015/old/old-national-similarity'
pt_similarity_path = './Data/2015/old/old-pt-similarity'

train_test_sets = './Data/2015/old/old-pt-train-test-sets'


# Path to VecAlign data
vecalign_alignments_path = './vecalign-data/final-alignments'


# Function get the information from the filename
def get_file_info(filename):
    file_info = (filename.split('_')[2]).split('.')
    code_type = file_info[0]
    pt_code = ((file_info[1].split('-'))[1]).upper()
    return code_type, pt_code


# Function to map the labels
def map_labels(row):
    if row['Difference Type'] == 'Common Sentence':
        if row['Variation'] == 'Yes':
            return 'Variation'
        else:
            return 'Common Sentence'
    return row['Difference Type']


def final_alignments(alignments_df, national_df, pt_df, national_similarity, pt_similarity, string):
    final_alignments = []
    for _, row in alignments_df.iterrows():
        national_index = row[0]
        pt_index = row[1]

        if (national_index == -1) and (pt_index != -1):
            pt_full_index = pt_df['Full Index'].iloc[pt_index -1]
            matched_pt_rows = pt_similarity[pt_similarity['Full Index'] == pt_full_index]
            matched_pt_sentence = matched_pt_rows[f'{string} Variations'].values[0] if not matched_pt_rows.empty else ''

            final_alignments.append({
                'National Sentence Text': '',
                'Matched National Sentence (Variation)': '',
                'P/T Sentence Text': pt_df['PT Sentence Text'].iloc[pt_index - 1],
                'Matched P/T Sentence (Variation)': matched_pt_sentence,
                'Similarity': 0.0
            })

        elif (pt_index == -1) and (national_index != -1):
            national_full_index = national_df['Full Index'].iloc[national_index - 1]
            matched_national_rows = national_similarity[national_similarity['Full Index'] == national_full_index]
            matched_national_sentence = matched_national_rows[f'National in {string}'].values[0] if not matched_national_rows.empty else ''

            final_alignments.append({
                'National Sentence Text': national_df['FRAG_DOCUMENT'].iloc[national_index - 1],
                'Matched National Sentence (Variation)': matched_national_sentence,
                'P/T Sentence Text': '',
                'Matched P/T Sentence (Variation)': '',
                'Similarity': 0.0
            })

        elif (national_index != -1) and (pt_index != -1):
            national_full_index = national_df['Full Index'].iloc[national_index - 1]
            matched_national_rows = national_similarity[national_similarity['Full Index'] == national_full_index]
            matched_national_sentence = matched_national_rows[f'National in {string}'].values[0] if not matched_national_rows.empty else ''

            pt_full_index = pt_df['Full Index'].iloc[pt_index - 1]
            matched_pt_rows = pt_similarity[pt_similarity['Full Index'] == pt_full_index]
            matched_pt_sentence = matched_pt_rows[f'{string} Variations'].values[0] if not matched_pt_rows.empty else ''

            final_alignments.append({
                'National Sentence Text': national_df['FRAG_DOCUMENT'].iloc[national_index - 1],
                'Matched National Sentence (Variation)': matched_national_sentence,
                'P/T Sentence Text': pt_df['PT Sentence Text'].iloc[pt_index - 1],
                'Matched P/T Sentence (Variation)': matched_pt_sentence,
                'Similarity': 1.0
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
                    all_misalignments_df.to_csv(f'./{model_name.lower()}-data/misalignments/{pt}/{pt} {code_book} {model} Misalignments.csv', index=False)

                if misalignments_test:
                    all_misalignments_test_df = pd.concat(misalignments_test, ignore_index=True)
                    all_misalignments_test_df.to_csv(f'./{model_name.lower()}-data/misalignments/{pt}/{pt} {code_book} {model} Misalignments Test.csv', index=False)

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


def implement_model(model_name):

    aer_results = {}
    final_test_alignments = {}

    total_correct_alignments = 0
    total_predicted_alignments = 0
    total_actual_alignments = 0

    total_train_correct_alignments = 0
    total_train_predicted_alignments = 0
    total_train_actual_alignments = 0

    main_scores_path = vecalign_alignments_path

    for filename in os.listdir(main_scores_path):
        aligned_df = pd.read_csv(main_scores_path + f'/{filename}', sep='\t', header=None)
        code_type, pt_code = get_file_info(filename)

        # Load the data
        national_df = pd.read_csv(f'{full_code_2015}/{code_type}/2015 National Sorted {code_type.capitalize()}.csv')
        pt_df = pd.read_csv(f'{full_code_2015}/{code_type}/2015 {pt_code} Sorted {code_type.capitalize()}.csv')

        # Get the matched similarities
        national_similarity = pd.read_csv(national_similarity_path + f'/2015 {pt_code} National Similarity.csv')
        pt_similarity = pd.read_csv(pt_similarity_path + f'/2015 {pt_code} PT Similarity.csv')

        # Get the thresholds for the province/territory
        national_threshold, pt_threshold = province_territories[pt_code]

        national_similarity = national_similarity[national_similarity['Similarity'] >= national_threshold]
        pt_similarity = pt_similarity[pt_similarity['Similarity'] >= pt_threshold]

        # Load the train set
        train_df = pd.read_csv(train_test_sets + f'/{pt_code} Train.csv')
        test_df = pd.read_csv(train_test_sets + f'/{pt_code} Test.csv')

        train_df['Label'] = train_df.apply(map_labels, axis=1)
        test_df['Label'] = test_df.apply(map_labels, axis=1)

        print(f'{pt_code} {code_type.capitalize()}:')
        
        alignments_df = final_alignments(aligned_df, national_df, pt_df, national_similarity, pt_similarity, pt_code)
        train_aer, train_correct_alignments, train_predicted_alignments, train_actual_alignments = calculate_alignment_error_rate(alignments_df, train_df, code_book=f'{code_type.capitalize()}', train=True)
        test_aer, correct_alignments, predicted_alignments, actual_alignments = calculate_alignment_error_rate(alignments_df, test_df, code_book=f'{code_type.capitalize()}', train=False, pt=f'{pt_code}', model_name=model_name)
        
        final_test_alignments[f'{pt_code} {code_type.capitalize()}'] = alignments_df

        # Store the results in the dictionary
        aer_results[f'{pt_code} {code_type.capitalize()}'] = {
            'Train AER': train_aer,
            'Test AER': test_aer
        }

        if test_aer == float('inf'):
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

        print(f'    Train AER: {train_aer}')
        print(f'    Test AER: {test_aer}\n')

    total_test_aer = calculate_aer(total_correct_alignments, total_predicted_alignments, total_actual_alignments)
    print(f'Total Test AER: {total_test_aer}\n')

    total_train_aer = calculate_aer(total_train_correct_alignments, total_train_predicted_alignments, total_train_actual_alignments)
    print(f'Total Train AER: {total_train_aer}\n')

    aer_results['Total Test AER'] = total_test_aer
    aer_results['Total Train AER'] = total_train_aer
    
    # Save the aer_results dictionary to a JSON file
    results_file = f'./{model_name.lower()}-data/results/{model_name} AER Results.json'
    with open(results_file, 'w') as f:
        json.dump(aer_results, f, indent=4)

    return final_test_alignments, total_test_aer, total_train_aer, aer_results


vecalign_test_alignments, vecalign_test_aers, vecalign_train_aers, vecalign_results, = implement_model('Vecalign')

# Save all the alignments from Vecalign
for key, value in vecalign_test_alignments.items():
    value.to_csv(f'./vecalign-data/results/alignments/{key} Alignments.csv', index=False)
