
import pandas as pd
import numpy as np
import os


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

def map_labels(row):
    if row['Difference Type'] == 'Common Sentence':
        if row['Variation'] == 'Yes':
            return 'Variation'
        else:
            return 'Common Sentence'
    return row['Difference Type']


train_test_sets = './Data/2015/old/old-pt-train-test-sets'

model_names = ['1-Hot', 'Bag of Words', 'TF-IDF']

for model in model_names:
    total_correct_alignments = 0
    total_predicted_alignments = 0
    total_actual_alignments = 0

    print(f'{model}: ')
    for file in os.listdir(f'./baseline-data/{model}/results/alignments'):
        alignments_df = pd.read_csv(f'./baseline-data/{model}/results/alignments/{file}')
        pt_name = file.split(' ')[0]
        code_book = file.split(' ')[1]
        
        train_df = pd.read_csv(f'{train_test_sets}/{pt_name} Train.csv')
        train_df['Label'] = train_df.apply(map_labels, axis=1)

        aer, correct_alignments, predicted_alignments, actual_alignments = calculate_alignment_error_rate(alignments_df, train_df, code_book, train=True, pt=pt_name, model_name=model)

        print(f'    {pt_name} {code_book}:')
        print(f'        AER: {aer}')
        total_correct_alignments += correct_alignments
        total_predicted_alignments += predicted_alignments
        total_actual_alignments += actual_alignments

    total_train_aer = calculate_aer(total_correct_alignments, total_predicted_alignments, total_actual_alignments)
    print(f'Total Train AER: {total_train_aer}\n')


model_types = ['bertscore', 'comet', 'labse', 'laser']

for model_type in model_types:
    total_correct_alignments = 0
    total_predicted_alignments = 0
    total_actual_alignments = 0

    print(f'{model_type}: ')
    for file in os.listdir(f'./{model_type}-data/results/alignments'):
        alignments_df = pd.read_csv(f'./{model_type}-data/results/alignments/{file}')
        pt_name = file.split(' ')[0]
        code_book = file.split(' ')[1]
        
        train_df = pd.read_csv(f'{train_test_sets}/{pt_name} Train.csv')
        train_df['Label'] = train_df.apply(map_labels, axis=1)

        aer, correct_alignments, predicted_alignments, actual_alignments = calculate_alignment_error_rate(alignments_df, train_df, code_book, train=True, pt=pt_name, model_name=model)

        print(f'    {pt_name} {code_book}:')
        print(f'        AER: {aer}')
        total_correct_alignments += correct_alignments
        total_predicted_alignments += predicted_alignments
        total_actual_alignments += actual_alignments

    total_train_aer = calculate_aer(total_correct_alignments, total_predicted_alignments, total_actual_alignments)
    print(f'Total Train AER: {total_train_aer}\n')


transformers_models = ['bilingual-embedding-large', 'multilingual-e5-large-instruct', 'mxbai-embed-large-v1']

for model in transformers_models:
    total_correct_alignments = 0
    total_predicted_alignments = 0
    total_actual_alignments = 0

    print(f'{model}: ')
    for file in os.listdir(f'./transformers-data/{model}/results/alignments'):
        alignments_df = pd.read_csv(f'./transformers-data/{model}/results/alignments/{file}')
        pt_name = file.split(' ')[0]
        code_book = file.split(' ')[1]
        
        train_df = pd.read_csv(f'{train_test_sets}/{pt_name} Train.csv')
        train_df['Label'] = train_df.apply(map_labels, axis=1)

        aer, correct_alignments, predicted_alignments, actual_alignments = calculate_alignment_error_rate(alignments_df, train_df, code_book, train=True, pt=pt_name, model_name=model)

        print(f'    {pt_name} {code_book}:')
        print(f'        AER: {aer}')
        total_correct_alignments += correct_alignments
        total_predicted_alignments += predicted_alignments
        total_actual_alignments += actual_alignments

    total_train_aer = calculate_aer(total_correct_alignments, total_predicted_alignments, total_actual_alignments)
    print(f'Total Train AER: {total_train_aer}\n')


