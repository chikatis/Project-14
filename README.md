# Project 14 - Construction Codes Variations

## Table of Contents
- [Data](#data) - Contains all the 2015 and the 2020 datasets.
  - [2015](#2015) - Entry point of the application.
    - [changed_labels] - Contains the rows whose `Difference Type` have been changed by me.
    - [full] - Contains data pertaining to the sentences taken from the code books.
      - [divb] - Isolated data for Division B.
        - [national-train-test] - Files with the data pertaining to the sentences from the National Model Codes into Train/Test.
        - [pt-full] - Files with data pertaining to sentences from each Provincial/Territorial Code Book, split based on each P/T.
        - [2015 DivB National Full new.csv] - Data of sentence texts from Division B of National Model Codes, with indices. (will be useful to compare with the sentences in the variations files)
        - [2015 DivB National Full.csv] - Data of sentence texts from Division B of National Model Codes, without indices.
        - [2015 DivB PT Full.csv] - Data of sentence texts from Division B of each of the P/T Codes.
        - [2015 National Full.csv] - Data of sentence texts from Division B of National Model Codes, with indices. (kept this file just in case)
      - [National Codes 2015 sentences.xlsx] - Source file for data corresponding to the sentences from National Model Codes.
      - [PT Sentence Data 2015.xlsx] - Source file for data corresponding to the sentences from all P/T Code Books.
    - [old] - 
      - [old-national-similarity] - Files with columns: `Full Index`: the index of the national sentence in the 2015 full National sentence data .xlsx file, `National Full`: sentence present in the 2015 National full sentence .xlsx file, `National in {PT}`: sentence present in the 2015 variations file corresponding to the PT, `Similarity`: simple similarity scores between the two sentences with best match being 1.0 and worst being 0.
      - [old-pt-similarity] - Has different files with columns: `Full Index`: the index of the PT sentence in the 2015 full PT sentence data .xlsx file, `{PT} Full`: sentence present in the 2015 PT full sentence .xlsx file, `{PT} Variations`: PT sentence present in the 2015 variations file corresponding to the PT, `Similarity`: simple similarity scores between the two sentences with best match being 1.0 and worst being 0.
      - [old-pt-train-test-sets] - Has three types of files, Train: training data, Test: testing dataset, Leftout: data not added to Train/Test, for each Province/Territory.
    - [variations] - Has different files that have the information regarding the variations in sentences that were manually found for the 2015 data, for each P/T.
      - [cleaned-data] - Various .csv files that end with dates, were QA'd, by handling data quality issues, and handed over in September.
        - [excel files] - Just the two files that were handed in excel format. (retaining the original format)
      - [div-b] - All the 2015 variations files, with Div B, for each P/T.
      - [old-data] - Various .csv files, with all the 2015 variations data, that were initially handed over in the month of May.
      - 
      
    
  - [2020](#2020) - Helper functions used in multiple modules.
- [docs](#docs) - Documentation files for the project.
  - [setup.md](#setupmd) - Guide on how to set up the project.
  - [API.md](#apimd) - API documentation.

## src

This folder contains the main source code for the project.

### index.js

This is the entry point of the application. It initializes the main components.

### utils

The `utils` folder includes reusable helper functions.

## docs

Documentation files that help understand and use the project.

### setup.md

This file provides a step-by-step guide on setting up the project environment.

### API.md

Documentation for the API endpoints exposed by the project.
