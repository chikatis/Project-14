# Project 14 - Construction Codes Variations

## Table of Contents
- **Data** - Contains all the data for 2015 and 2020 Code Books.
  - **2015** - Entry point of the application.
    - **changed_labels** - Contains the rows whose `Difference Type` have been changed by me.
    - **full** - Contains data pertaining to the sentences taken from the code books.
      - **divb** - Isolated data for Division B.
        - **national-train-test** - Files with the data pertaining to the sentences from the National Model Codes into Train/Test.
        - **pt-full** - Files with data pertaining to sentences from each Provincial/Territorial Code Book, split based on each P/T.
        - **2015 DivB National Full new.csv** - Data of sentence texts from Division B of National Model Codes, with indices. (will be useful to compare with the sentences in the variations files)
        - **2015 DivB National Full.csv** - Data of sentence texts from Division B of National Model Codes, without indices.
        - **2015 DivB PT Full.csv** - Data of sentence texts from Division B of each of the P/T Codes.
        - **2015 National Full.csv** - Data of sentence texts from Division B of National Model Codes, with indices. (kept this file just in case)
      - **National Codes 2015 sentences.xlsx** - Source file for data corresponding to the sentences from National Model Codes.
      - **PT Sentence Data 2015.xlsx** - Source file for data corresponding to the sentences from all P/T Code Books.
    - **old** - 
      - **old-national-similarity** - Files with columns: `Full Index`: the index of the national sentence in the 2015 full National sentence data .xlsx file, `National Full`: sentence present in the 2015 National full sentence .xlsx file, `National in {PT}`: sentence present in the 2015 variations file corresponding to the PT, `Similarity`: simple similarity scores between the two sentences with best match being 1.0 and worst being 0.
      - **old-pt-similarity** - Has different files with columns: `Full Index`: the index of the PT sentence in the 2015 full PT sentence data .xlsx file, `{PT} Full`: sentence present in the 2015 PT full sentence .xlsx file, `{PT} Variations`: PT sentence present in the 2015 variations file corresponding to the PT, `Similarity`: simple similarity scores between the two sentences with best match being 1.0 and worst being 0.
      - **old-pt-train-test-sets** - Has three types of files, Train: training data, Test: testing dataset, Leftout: data not added to Train/Test, for each Province/Territory.
    - **variations** - Has different files that have the information regarding the variations in sentences that were manually found for the 2015 data, for each P/T.
      - **cleaned-data** - Various .csv files that end with dates, were QA'd, by handling data quality issues, and handed over in September.
        - **excel files** - Just the two files that were handed in excel format. (retaining the original format)
      - **div-b** - All the 2015 variations files, with Div B, for each P/T.
      - **old-data** - Various .csv files, with all the 2015 variations data, that were initially handed over in the month of May.
        
  - **2020** - Contains XML and CSV files with the data related to the sentence texts taken from the 2020 Code books. The XML or DITA files are source codes, while the CSV files are extracts from these source files.
    - **national** - Files for the National 2020 Model Codes.
      - **csv_files** - The CSV file `nbc2020_p1.csv` contains the extracts from the XML files. In the extracted files, IDs like es000101, ea004131, en000009, must be replaced by their respective numbers. The alphabetical characters in the IDs tell us what kind of data it is. For example, *es* stands for *English Sentence*, *ea* stands for *English Article*, *en* for *English Note*, and so on.
      - **xml_files** - The XML file `nbc2020_p1.xml` is of interest, which contains the data for the 2020 National Building Code.
    - **pt** - Files for 2020 P/T Codes for AB, BC, ON. 
      - **AB**:
          - **csv_files** - CSV files with extracted data from 2023 AB Code Books.
            - **abc2023_p1.csv** - CSV file with extracted data for the 2023 Alberta Building Code Book. In the extracted files, IDs like es000101, ea004131, en000009, must be replaced by their respective numbers. The alphabetical characters in the IDs tell us what kind of data it is. For example, *es* stands for *English Sentence*, *ea* stands for *English Article*, *en* for *English Note*, and so on.
            - **afc2023_p1.csv** - CSV file with extracted data for the 2023 Alberta Fire Code Book. In the extracted files, IDs like es000101, ea004131, en000009, must be replaced by their respective numbers, in this file too.
          - **xml_files** - XML source files containing data pertaining to 2023 ABC and 2023 AFC.
            - **abc2023_p1.xml** - XML source file for the 2023 Alberta Building Code Book.
            - **afc2023_p1.xml** - XML source file for the 2023 Alberta Fire Code Book.
      - **BC**:
          - **xml_files** - Contains fragmented parts of the 2020 BC Building Code in the form of DITA files. The data for BC has not been extracted yet.
      - **ON**:
          - **csv_files** - CSV file with the extracted 2024 Ontario Building Code. This extract is complete.
          - **xml_files** - The XML file called `obc2024.xml` is of interest, which contains the data from the 2024 Ontario Building Code.

- **all-results** - An older folder with the results from some models.

- **sorted-split-up** - Each of the National Codes and P/T Codes data split up based on the Code Book: building, combined (important for the Ontario), energy, fire, plumbing. Then, the sentence texts in the files are sorted by the increasing order of their sentence numbers.

- **split-up-indexed** - Each of the National Codes and P/T Codes data split up based on the Code Book: building, combined (important for the Ontario), energy, fire, plumbing. These files also contain indices to make it easier when evaluating the models. (since there are inconsistencies between the same sentences in the full sentence files and the variations files) Note that you might have to change the name of the folder while running scripts.

- **split-up** - Each of the National Codes and P/T Codes data split up based on the Code Book: building, combined (important for the Ontario), energy, fire, plumbing. The order of each row is the same as before and these files do not have the indices. We will not be using this. 

- All folders that end with '-data' contain all the results from the models. The results include alignments of the `National Sentence Text` and `P/T Sentence Text` by the model, the plots of the Alignment Error Rates (AERs) vs. Threshold, the misalignments by the model, (with the alignments made by the model, that were incorrect as given by the test set, and also the misalignments from the test set. The former would end with `Misalignments.csv` and the latter would end with `Misalignments Test.csv`), and the results with the Thresholds used for each P/T and Code Book, individual AERs, the total train AER, and the total test AER. Note that the curves for the three models are plotted in one graph for each Code Book, for each P/T. Note that we won't be having graphs in `vecalign-data`, since the alignments are decided by vecalign.

- **baselines.py** - Python script to convert the National and P/T Sentences into three types of vectors: *Bag-of-Words*, *tf-idf-weighted-BoW*, and *One-Hot Encoding*, Get the cosine similarity scores, use maximum weight matching to get the best matches between them, choose a threshold to break up the alignments from maximum weight matching (it tends to overalign sentences) using the elbow method (over a range of thresholds, get the AER for each threshold on the training set, plot the AERs vs Thresholds, choose the best threshold corresponding to the lowest train AER as the threshold) choose the best threshold, use the threshold on the alignments. These would be the final alignments. Then, use the test set to get the final total test AER.

- **bertscore-and-comet.py** - Script to follow the above process to get the final total test AER. But instead of running the model locally, since they are expensive to run, they were run on a cluster, and the index + 1 of the National and the P/T sentence texts along with all of the scores between them, were given to me. Esentially, the files would have scores between each National sentence and every P/T sentence. The files are very large so they were not uploaded on here. See the Teams channel for the scores files.

- **combine-files.py** - Script to split up the individual P/T Codes or National Codes, based on the Code Book. The script also contains code to combine files to get the Ontario Combined and the National Combined files.

- **extract-from-xml.py** - Script to extract 2020 data from the XML or DITA files and store the extracted data as CSV files.

- **fine-tuned-GIST.py** - Script to fine-tune the GIST-Embedding-v0 model on contrastive training (that uses positive pairs - similar sentences, to pull the similar sentences close to each other in space, and negative pairs - dissimilar sentences, to push apart the dissimilar sentences far apart in space). The rest of the process remains the same to get the AERs.

- **labse-and-laser.py** - Script to get the AERs of the alignments from LaBSE and LASER, using the method mentioned above. Again, it is expensive to run these models. The scores files follow the same structure as those for BERTScore and COMET, and are available on the Teams channel in `experiments` folder.

- **models-transformers.py** - Script to convert the sentences into embeddings based on four different transformer-based models, `bilingual-embedding-large`, `multilingual-e5-large-instruct`, `mxbai-embed-large-v1`, `GIST-Embedding-v0`, then following the steps as above, to get the AERs.

- **preprocessing.py** - Script to clean the sentence texts, prepare the data, and create the train/test sets for each Province/Territory. We also use a method to get the best matches of the same sentences in the full sentence excel files and those in the variations csv files, making our evaluation smoother.

- **total-train-aers.py** - Script used to get the total Train AER for some of the models.

- **transformers.py** - Old script to run the transformer-based models without the `GIST-Embedding-v0` model, and get the AERs.

- **vecalign.py** - Vecalign decides the alignments between the National and the P/T sentences. So, we would not use maximum weight matching and then get the threshold in this case. We would directly compute the train and test AERs on the training and test datasets, repectively. The data for this model can also be found on Teams channel in the `experiments` folder.


## Results
| Category                        | Model                          | Training | Test   |
|---------------------------------|--------------------------------|----------|--------|
| parallel text alignment baseline| vecalign                       | 0.4564   | 0.4568 |
| feature vector-based            | 1-Hot                          | 0.1296   | 0.1426 |
|                                 | Bag-of-Words                   | 0.1402   | 0.1554 |
|                                 | $tf$-$idf$ Weighted BoW        | 0.1233   | 0.1372 |
| sentence embeddings             | LASER2                         | 0.1635   | 0.1783 |
|                                 | LaBSE                          | 0.1352   | 0.1471 |
| task specific fine-tuned        | bilingual-embedding-large      | 0.1183   | **0.1306** |
|                                 | multilingual-e5-large-instruct | 0.1210   | 0.1403 |
|                                 | mxbai-embed-large-v1           | 0.1194   | 0.1366 |
|                                 | GIST-embedding-v0              | **0.1165** | 0.1339 |
| construction codes fine-tuned   | GIST-embedding-v0              | 0.1370   | 0.1522 |
| MT evaluation metric            | BERTScore                      | 0.1427   | 0.1475 |
|                                 | COMET                          | 0.1604   | 0.1622 |
