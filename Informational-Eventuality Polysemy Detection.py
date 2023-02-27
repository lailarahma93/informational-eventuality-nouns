#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 22:27:46 2023

@author: laila
"""

import re
import spacy
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

# to open/read the dataset
with open('/Users/laila/Downloads/ind_mixed_2013_30K/ind_mixed_2013_30K-sentences.txt') as f:
    #text = f.read()
    text = list(f)

# to get sentences with informational-eventuality nouns
def search_sentence(regex, data):
    selected_sentences = []
    for sentence in data:
        if re.findall(regex, sentence):
            selected_sentences.append(sentence)
    return selected_sentences

# to get informational-eventuality nouns
def search_noun(regex, data):
    string = ''.join(data)
    target = re.findall(regex, string)
    return target

# to remove punctuation
def remove_punctuation(data):
    chars_to_remove = "!\“#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789\r\n"
    without_punctuation = str.maketrans("", "", chars_to_remove)
    string = ''.join(data)
    remove_punctuation = string.translate(without_punctuation)
    return remove_punctuation

# to stem the nouns
def stemming(data):
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # stemming process
    sentence = data
    output   = stemmer.stem(sentence)    
    return output

# to give POS label
def pos_tagging(data):
    nlp = spacy.load("/Users/laila/Downloads/Models/Spacy2.1.x/NER/id_ud-tag-dep-ner-1.0.0/id_ud-tag-dep-ner/id_ud-tag-dep-ner-1.0.0")
    doc = nlp(data)
    pos = []
    for token in doc:
        if not token.is_punct:
            pos.append(token.pos_)
    return(pos)

# to predict if the noun is informational-eventuality noun or not
# by checking if the ke-an/pe-an is circumfix or part of the base/stem
def detection(data):
    for i in range(len(data)):
        if data.loc[i,"Noun"] == data.loc[i,"Stem"]: #if the noun and the stem are the same, that the ke-an/pe-an is part of the base/stem
            data.loc[i,"Prediction"] = 'NO'
        else:
            data.loc[i,"Prediction"] = 'YES' #if the noun and the stem are different, that means the ke-an/pe-an is circumfix
    return data

def calculate_accuracy(data):
    n_correct = 0   #number of correct predictions
    for i in range(len(data)):
        if data.loc[i,"Prediction"] == data.loc[i,"Gold"]:
            n_correct += 1
    accuracy = n_correct/len(data)
    percentage = accuracy*100
    print("\n",pd.crosstab(data.Prediction,data.Gold))
    return str(percentage)+"%"

def main():
    
    # Extracting sentences containing nouns with circumfix "ke--an"
    sentences_with_keNOUNan = search_sentence('\W[k][e]\w+[a][n]\W+bahwa|\W[k][e]\w+[a][n]\W+kalau|\W[k][e]\w+[a][n]\W+jika', text)
    print('There are', len(sentences_with_keNOUNan), 'sentences containing nouns with circumfix ke--an')
    
    # Extracting sentences containing nouns with circumfix "pe--an"
    sentences_with_peNOUNan = search_sentence(r'\W[p][e]\w+[a][n]\W+bahwa|\W[p][e]\w+[a][n]\W+kalau|\W[p][e]\w+[a][n]\W+jika', text)
    print('There are', len(sentences_with_peNOUNan), 'sentences containing nouns with circumfix pe--an')
    
    # Extracting nouns with circumfix "ke--an", incl. removing punctuation
    keNOUNan = search_noun('\W[k][e]\w+[a][n]\W+bahwa|\W[k][e]\w+[a][n]\W+kalau|\W[k][e]\w+[a][n]\W+jika', text)
    keNOUNan = [s.strip('bahwa') for s in keNOUNan]
    keNOUNan = [s.strip('kalau') for s in keNOUNan]
    keNOUNan = [s.strip('jika') for s in keNOUNan]
    clean_keNOUNan = remove_punctuation(keNOUNan)
    print(clean_keNOUNan)
    
    # Extracting nouns with circumfix "pe--an", incl. removing punctuation
    peNOUNan = search_noun(r'\W[p][e]\w+[a][n]\W+bahwa|\W[p][e]\w+[a][n]\W+kalau|\W[p][e]\w+[a][n]\W+jika', text)
    peNOUNan = [s.strip('bahwa') for s in peNOUNan]
    peNOUNan = [s.strip('kalau') for s in peNOUNan]
    peNOUNan = [s.strip('jika') for s in peNOUNan]
    clean_peNOUNan = remove_punctuation(peNOUNan)
    print(clean_peNOUNan)
    
    # Stemming the nouns with circumfix "ke--an" and POS-tagging the stems
    keNOUNan_stem = stemming(clean_keNOUNan)
    print(keNOUNan_stem)
    keNOUNan_stem_pos = pos_tagging(keNOUNan_stem)
    print(keNOUNan_stem_pos)
    
    # Stemming the nouns with circumfix "pe--an" and POS-tagging the stems
    peNOUNan_stem = stemming(clean_peNOUNan)
    print(peNOUNan_stem)
    peNOUNan_stem_pos = pos_tagging(peNOUNan_stem)
    print(peNOUNan_stem_pos)
 
    # Create dataframe consisting of sentences with ke-an nouns, the nouns, the stems, and the stem POS
    clean_keNOUNan_list = clean_keNOUNan.split('  ')
    keNOUNan_stem_list = keNOUNan_stem.split(' ')
    keNOUNan_table = pd.DataFrame(
    {'Sentence': sentences_with_keNOUNan,
      'Noun': clean_keNOUNan_list,
      'Stem': keNOUNan_stem_list,
      'Stem POS': keNOUNan_stem_pos
    })
    keNOUNan_table[['zero','Sentence']] = keNOUNan_table['Sentence'].str.split("[0-9]*?\t", expand=True)
    keNOUNan_table = keNOUNan_table.drop(columns=['zero'])
    
    # Create dataframe consisting of sentences with pe-an nouns, the nouns, the stems, and the stem POS
    clean_peNOUNan_list = clean_peNOUNan.split(' ')
    clean_peNOUNan_list = [i for i in clean_peNOUNan_list if i]
    peNOUNan_stem_list = peNOUNan_stem.split(' ')
    peNOUNan_table = pd.DataFrame(
    {'Sentence': sentences_with_peNOUNan,
      'Noun': clean_peNOUNan_list,
      'Stem': peNOUNan_stem_list,
      'Stem POS': peNOUNan_stem_pos
    })
    peNOUNan_table[['zero','Sentence']] = peNOUNan_table['Sentence'].str.split("[0-9]*?\t", expand=True)
    peNOUNan_table = peNOUNan_table.drop(columns=['zero'])
    
    # Predicting noun is informational-eventuality nouns or not
    # by checking if the ke-an/pe-an is circumfix or part of the base/stem
    detection_keNOUNan = detection(keNOUNan_table)
    detection_peNOUNan = detection(peNOUNan_table)
    
    # Adding gold annotation to the dataframe
    detection_keNOUNan['Gold'] = ['YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'NO', 'YES', 'YES', 'NO',
                                  'YES', 'NO', 'NO', 'YES', 'YES', 'NO', 'YES', 'YES', 'YES', 'YES', 'YES', 'NO', 'YES', 'YES', 'YES',
                                  'NO', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'NO', 'YES']
    detection_peNOUNan['Gold'] = ['YES', 'YES', 'YES', 'NO', 'NO', 'YES', 'NO', 'YES', 'YES', 'NO', 'YES', 'YES', 'YES', 'YES', 'YES',
                                  'YES', 'NO', 'YES', 'YES', 'YES', 'NO', 'NO', 'NO']
    
    # Saving the dataframe as csv, if needed
    df_keNOUNan = detection_keNOUNan.to_csv('df_keNOUNan.csv') 
    df_peNOUNan = detection_peNOUNan.to_csv('df_peNOUNan.csv')
    
    # Checking the accuracy and creating confusion matrix
    keNOUNan_acc = calculate_accuracy(detection_keNOUNan)
    print(keNOUNan_acc)
    peNOUNan_acc = calculate_accuracy(detection_peNOUNan)
    print(peNOUNan_acc)

main()