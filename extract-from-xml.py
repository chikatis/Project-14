
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import re


def text_preprocessing(column):
    column = column.str.replace('', ' ') \
                    .str.strip() \
                    .str.replace('\n', ' ') \
                    .str.replace('\x93', ' ') \
                    .str.replace('\x94', ' ') \
                    .str.replace('\x96', ' ') \
                    .str.replace('\x97', ' ') \
                    .str.replace(r'\s{2,}', ' ', regex=True) \
                    .str.strip()
    # .str.replace(r'\s*\(.*?\)$', '', regex=True) \

    return column


# Function to extract text from an element, including nested elements
def get_full_text(element):
    parts = []
    if element.text:
        parts.append(element.text)
        if element.text.strip().endswith(':'):
            parts.append(' ') # Add a space after a colon
    for subelement in element:
        if subelement.tag == 'ref.int':
            pretext = subelement.get('pretext', '')
            refid = subelement.get('refid', '')
            parts.append(f"{pretext} {refid}".strip())
        elif subelement.tag == 'ref.stand':
            if subelement.text:
                parts.append(subelement.text)
            parts.append(f"{subelement.get('StandID')}")
            punctuation = subelement.get('punc')
            if punctuation == 'period':
                parts.append('.')
            elif punctuation == 'comma':
                parts.append(',')
        elif subelement.tag == 'ref.ext':
            if subelement.text:
                parts.append(subelement.text)
        elif subelement.tag == 'ref.bib':
            parts.append(f"{subelement.get('refid')}")
        elif subelement.tag == 'def.group':
            defterm = subelement.find('defterm').text
            text_defin = get_full_text(subelement.find('text.defin'))
            parts.append(f"{defterm} {text_defin}")
            parts.append(' ')  # Add a space after each definition group
        else:   
                parts.append(get_full_text(subelement))
        if subelement.tail:
            parts.append(subelement.tail)
    return ''.join(parts).strip()


# Extract the text name from the row
def extract_division(column):
    # Split each row based on '-', capitalize the letters, and include everything after the first part
    return column.str.split('-').str[1].str.capitalize()

# Function to number the sections and the subsections based on the text
def number_sections_and_subsections(data, pt=True):
    if pt:
        data['Section ID'] = data['P/T Section ID'].apply(lambda x: x[2:])
        data['Subsection ID'] = data['P/T Subsection ID'].apply(lambda x: x[2:])
    else:
        data['Section ID'] = data['National Section ID'].apply(lambda x: x[2:])
        data['Subsection ID'] = data['National Subsection ID'].apply(lambda x: x[2:])
    
    def renumber_division(division_data):
        section_mapping = {}
        subsection_mapping = {}
        section_counter = 1
        section_key_mapping = {}

        for i, row in division_data.iterrows():
            section_parts = row['Section ID'].split('.')
            subsection_parts = row['Subsection ID'].split('.')

            section_key = section_parts[0]
            if section_key not in section_key_mapping:
                section_key_mapping[section_key] = section_counter
                section_counter += 1

            new_section_number = section_key_mapping[section_key]
            section_key_full = f"{new_section_number}.{section_parts[1]}"
            section_mapping[row['Section ID']] = section_key_full

            subsection_key = f"{section_key_full}.{subsection_parts[2]}"
            subsection_mapping[row['Subsection ID']] = subsection_key

        division_data.loc[:, 'Section Number'] = division_data['Section ID'].map(section_mapping)
        division_data.loc[:, 'Subsection Number'] = division_data['Subsection ID'].map(subsection_mapping)
        return division_data

    div_a = data[data['Division'] == 'A'].copy()
    div_b = data[data['Division'] == 'B'].copy()
    div_c = data[data['Division'] == 'C'].copy()

    div_a = renumber_division(div_a)
    div_b = renumber_division(div_b)
    div_c = renumber_division(div_c)

    result = pd.concat([div_a, div_b, div_c])

    # Drop the 'Section ID' and 'Subsection ID' columns
    result.drop(columns=['Section ID', 'Subsection ID'], inplace=True)

    # Reorder the columns
    if pt:
        result = result[['Division', 'P/T Section ID', 'Section Number', 'P/T Subsection ID', 'Subsection Number', 'P/T Article Title', 'P/T Sentence Number', 'P/T Sentence Text']]
    else:
        result = result[['Division', 'National Section ID', 'Section Number', 'National Subsection ID', 'Subsection Number', 'National Article Title', 'National Sentence Number', 'National Sentence Text']]

    return result


# Function to extract data from XML and write to CSV
def extract_xml_to_csv(xml_file, csv_file, pt=True):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Open the CSV file for writing
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header row
        if pt:
            writer.writerow(['Division', 'P/T Section ID', 'P/T Subsection ID', 'P/T Article Title', 'P/T Sentence Number', 'P/T Sentence Text'])
        else:
            writer.writerow(['Division', 'National Section ID', 'National Subsection ID', 'National Article Title', 'National Sentence Number', 'National Sentence Text'])

        # Iterate through each Division in the XML
        for div in root.iter('OBCode.div'):
            # Extract the division name
            div_name = div.get('id')

            # Iterate through each appnote in the XML
            for part in div.iter('part'):
                # For each section
                for sect in part.iter('section'):
                    section = sect.get('id')
                    for subsect in sect.iter('subsect'):
                        subsection = subsect.get('id')
                        for article in subsect.iter('article'):
                            # Extract the article title
                            article_title = get_full_text(article.find('title'))

                            # Iterate through each sentence in the article
                            for sent in article.iter('sentence'):
                                sentence_id = sent.get('id')

                                # Extract the sentence text
                                sent_text = get_full_text(sent)

                                # Write the data to the CSV file
                                writer.writerow([div_name, section, subsection, article_title, sentence_id, sent_text])

    csv_data = pd.read_csv(csv_file)
    if pt:
        csv_data['Division'] = extract_division(csv_data['Division'])
        csv_data = number_sections_and_subsections(csv_data, pt=True)
        csv_data['P/T Article Title'] = text_preprocessing(csv_data['P/T Article Title'])
        csv_data['P/T Sentence Text'] = text_preprocessing(csv_data['P/T Sentence Text'])
    else:
        csv_data['Division'] = extract_division(csv_data['Division'])
        csv_data = number_sections_and_subsections(csv_data, pt=False)
        csv_data['National Article Title'] = text_preprocessing(csv_data['National Article Title'])
        csv_data['National Sentence Text'] = text_preprocessing(csv_data['National Sentence Text'])

    csv_data.to_csv(csv_file, index=False)


# Define the input XML file and output CSV file
afc_xml_file = './Data/2020/pt/AB/xml-files/afc2023_p1.xml'  # Replace with your XML file path
afc_csv_file = './Data/2020/pt/AB/csv-files/afc2023_p1.csv'  # Replace with your desired CSV file path

# Call the function to extract data and write to CSV
extract_xml_to_csv(afc_xml_file, afc_csv_file)

print(f"Data extracted and written to {afc_csv_file}")


afc_2023 = pd.read_csv(afc_csv_file)
afc_2023.head()


afc_2023['P/T Sentence Text'][1]


afc_2023['Division'].value_counts()


# Define the input XML file and output CSV file
abc_xml_file = './Data/2020/pt/AB/xml-files/abc2023_p1.xml'  # Replace with your XML file path
abc_csv_file = './Data/2020/pt/AB/csv-files/abc2023_p1.csv'  # Replace with your desired CSV file path

# Call the function to extract data and write to CSV
extract_xml_to_csv(abc_xml_file, abc_csv_file)

print(f"Data extracted and written to {abc_csv_file}")


abc_2023 = pd.read_csv(abc_csv_file)
abc_2023.head()


abc_2023['P/T Sentence Text'][1198]


# Define the input XML file and output CSV file
nbc_xml_file = './Data/2020/national/xml-files/nbc2020_p1.xml'  # Replace with your XML file path
nbc_csv_file = './Data/2020/national/csv-files/nbc2020_p1.csv'  # Replace with your desired CSV file path

# Call the function to extract data and write to CSV
extract_xml_to_csv(nbc_xml_file, nbc_csv_file, pt=False)

print(f"Data extracted and written to {nbc_csv_file}")


nbc2020_p1 = pd.read_csv(nbc_csv_file)
nbc2020_p1.head()


nbc2020_p1['National Sentence Text'][29]


nbc2020_p1['Division'].tail()


nbc2020_p1['Division'].value_counts()


# Function to extract text from an element, including nested elements
def extract_text(element):
    text = element.text or ""
    for subelement in element:
        text += ' ' + extract_text(subelement)
        if subelement.tail:
            text += f' {subelement.tail}'
    return text.strip()

# Function to clean the text
def clean_text(column):
    column = column.str.replace(r'^\([\d]+\)', '', regex=True) \
                    .str.replace(' .', '.', regex=False) \
                    .str.replace(' ,', ',', regex=True) \
                    .str.replace('\n', ' ') \
                    .str.replace(r'\s{2,}', ' ', regex=True) \
                    .str.strip()
    return column

# Function to parse the DITA file and extract the required data
def extract_dita_to_csv(xml_file, csv_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    current_division = None
    current_article_title = ''
    current_section_number = ''
    sentence_number = ''
    full_sentence_text = ''
    sentence_id = ''
    processing = False

    number_pattern = re.compile(r'^\d+\.\d+\.\d+\.\d+[A-Za-z]*\.')
    bullet_pattern = re.compile(r'^\([\d]+\)')
    bullet_pattern2 = re.compile(r'^\( [\d]+ \)')
    subbullet_pattern = re.compile(r'^\([A-Za-z]+\)')

    all_styles = []

    for elem in root.iter():
        if elem.tag != 'p':
            continue
        all_styles.append(elem.attrib.get('style-name', ''))

    # Get the unique style names
    all_style_names = set(all_styles)

    # Define the style names you are using
    used_style_names = ['subsubclause-e', 'Nornal', 'equation-e', 'Sdefinition-e', 'subpara-e', 'Ssection-e', 'Psection-e',
                          'equationind1-e', 'section-e', 'subclause-e', 'paragraph-e', 'subection-e', 'Ysection-e', 'Sclause-e', 
                          'equationind2-e', 'equationind3-e', 'defsubclause-e', 'Ssubsection-e', 'firstdef-e', 'defclause-e', 'clause-e', 
                          'definition-e', 'subsection-e', 'ruleb-e']

    for elem in root.iter():
        if elem.tag != 'p':
            continue
        text = extract_text(elem)
        text = text.replace('\t', ' ') 
        text = re.sub(r'\(\s*(\d+)\s*\)', r'(\1)', text).strip()
        style_name = elem.attrib.get('style-name', '')

        # Detect Division
        if ('DIVISION A' == text.upper()[0:10]) and (style_name == 'partnum-e'):
            processing = True
            # Before updating current_division, save any pending sentence
            if full_sentence_text:
                data.append({
                    'ID': sentence_id,
                    'Division': current_division,
                    'Article Title': current_article_title,
                    'Sentence Number': sentence_number,
                    'Sentence Text': full_sentence_text
                })
                # Reset variables
                full_sentence_text = ''
                sentence_number = ''
                sentence_id = ''
                current_article_title = ''
                current_section_number = ''
            current_division = 'A'
            continue
        
        elif ('DIVISION B' == text.upper()[0:10]) and (style_name == 'partnum-e') and processing:
            # Before updating current_division, save any pending sentence
            if full_sentence_text:
                data.append({
                    'ID': sentence_id,
                    'Division': current_division,
                    'Article Title': current_article_title,
                    'Sentence Number': sentence_number,
                    'Sentence Text': full_sentence_text
                })
                # Reset variables
                full_sentence_text = ''
                sentence_number = ''
                sentence_id = ''
                current_article_title = ''
                current_section_number = ''
            current_division = 'B'
            continue

        elif ('DIVISION C' == text.upper()[0:10]) and (style_name == 'partnum-e') and processing:
            # Before updating current_division, save any pending sentence
            if full_sentence_text:
                data.append({
                    'ID': sentence_id,
                    'Division': current_division,
                    'Article Title': current_article_title,
                    'Sentence Number': sentence_number,
                    'Sentence Text': full_sentence_text
                })
                # Reset variables
                full_sentence_text = ''
                sentence_number = ''
                sentence_id = ''
                current_article_title = ''
                current_section_number = ''
            current_division = 'C'
            continue

        if processing and (style_name in used_style_names):
            match = number_pattern.match(text)
            if match:
                # Before updating article title, save any pending sentence
                if full_sentence_text:
                    data.append({
                        'ID': sentence_id,
                        'Division': current_division,
                        'Article Title': current_article_title,
                        'Sentence Number': sentence_number,
                        'Sentence Text': full_sentence_text
                    })
                    # Reset variables
                    full_sentence_text = ''
                    sentence_number = ''
                    sentence_id = ''
                # Update article title and section number
                current_section_number = match.group()
                current_article_title = number_pattern.sub('', text).strip()
                continue

            if current_section_number is not None:
                bullet_match = bullet_pattern.match(text)

                if bullet_match:
                    # Before starting a new sentence, save the current one if any
                    if full_sentence_text:
                        data.append({
                            'ID': sentence_id,
                            'Division': current_division,
                            'Article Title': current_article_title,
                            'Sentence Number': sentence_number,
                            'Sentence Text': full_sentence_text
                        })
                    # Start new sentence
                    sentence_number = f'{current_section_number}{bullet_match.group()}'
                    sentence_text = text.strip()
                    sentence_id = elem.attrib.get('id', '')
                    full_sentence_text = sentence_text
                else:
                    if subbullet_pattern.match(text):
                        # Accumulate subbullet text
                        full_sentence_text += ' ' + text.strip()
                    else:
                        # Accumulate sentence text
                        full_sentence_text += ' ' + text.strip()

    # Append the last sentence after the loop
    if full_sentence_text:
        data.append({
            'ID': sentence_id,
            'Division': current_division,
            'Article Title': current_article_title,
            'Sentence Number': sentence_number,
            'Sentence Text': full_sentence_text
        })

    fieldnames = ['ID', 'Division', 'Article Title', 'Sentence Number', 'Sentence Text']
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    # Clean the 'Sentence Text' column
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['ID', 'Article Title', 'Sentence Number'], how='all')
    df['Sentence Text'] = clean_text(df['Sentence Text'])
    df.to_csv(csv_file, index=False)



def calculate_aer(correct, predicted, actual):
    if actual + predicted > 0:
        aer = 1 - ((2 * correct) / (predicted + actual))
    else:
        aer = float('inf')
    return aer


# Define the input XML file and output CSV file
obc_xml_file = './Data/2020/pt/ON/xml-files/obc2024.xml'  
obc_csv_file = './Data/2020/pt/ON/csv-files/obc2024.csv' 

# Call the function to extract data and write to CSV
extract_dita_to_csv(obc_xml_file, obc_csv_file)

print(f"Data extracted and written to {obc_csv_file}")
