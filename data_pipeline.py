
from params import *
import pandas as pd
import numpy as np 
import yaml
from params import * 
import re 
import os  
from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import difflib
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from fuzzywuzzy import process

def import_transcriptions():
    data = pd.read_csv(os.path.join(PATH_DATA,'transcriptions_with_sex.csv'))
    return data 


def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def explode_column(data, column, tags):
    output = data.copy()
    pattern = lambda tag: rf"{tag}: ([^:]+)(?= \w+:|$)"
    for tag in tags:
        output[tag] = output[column].apply(lambda x: re.search(pattern(tag), x).group(1) if re.search(pattern(tag), x) else np.nan)
    return output 

def rename_predictions(data,mapping) :
    output = data.copy()
    output = output.rename(columns = mapping)
    return output 

def encode_freq(x):
    
    if x > 0.75 :
        return 'prenom homme'
    elif x <=0.25:
        return 'prenom femme'
    elif not x:
        return "mixte"
    else :
        return "mixte"
    
def import_freq_name():
    name_freq = pd.read_csv(os.path.join(PATH_DATA,'firstname_with_sex.csv'), sep = ';')
    name_freq['total'] = name_freq[['male','female']].sum(1)
    name_freq['freq_male'] = name_freq['male'] / name_freq['total']
    name_freq['freq_female'] = name_freq['female'] / name_freq['total']
    name_freq["name_sex"] = name_freq["freq_male"].apply(lambda x : encode_freq(x))
    return name_freq


def modify_firstname(data):
    data['firstname'] = data['firstname'].apply(lambda x : str(x).split(' ')[0] if x else None)
    data['firstname'] = data['firstname'].apply(lambda x : str(x).split('-')[0] if x else None)
    data['firstname'] = data['firstname'].apply(lambda x : str(x).replace('ï','i') if x else None)
    data['firstname_lower'] = data['firstname'].apply(lambda x : str.lower(x) if x else None)
    data['firstname_lower']= data['firstname_lower'].apply(lambda x : x.replace(' ','') if x else None)
    return data 


def merge_with_namefreq(data,name_freq):
    merged = data.merge(name_freq.set_index('firstname'),left_on='firstname_lower',right_index=True,how = 'left')
    return merged


def get_list_valid_names(name_freq):
    valid_names = name_freq[((name_freq["freq_female"] - 0.5).abs()>0.25) & (name_freq["total"] >1000)]['firstname'].index.tolist()
    return valid_names

    
def drop_ambiguous(data):
    data = data[~(data['sex']=="ambigu")]
    return data


def encode_label(x):
    if x :
        if x =='homme':
            return 1
        if x == 'femme':
            return 0
    else :
        return None
    

def make_dir(dir_path):
    
    if not os.path.exists(dir_path):
        # Create the directory
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' was created.")
    else:
        print(f"Directory '{dir_path}' already exists.")
    
    


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
    
    
def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def transform_tags(tags):
    tags_clean = {}
    for col in tags :
        tag = tags[col]['start']
        tags_clean[tag] = col
    return tags_clean




def parse_observation(observation, tags):
    parts = re.split('(' + '|'.join(re.escape(key) for key in tags.keys()) + ')', observation)
    parts = [part for part in parts if part]
    result = {col: None for col in tags.values()}  # Initialize all columns with None
    
    tag = None  # Keep track of the current tag
    for part in parts:
        if part in tags:  # If the part is a tag
            tag = tags[part]
        elif tag:  # If the part is not a tag, it is a value for the current tag
            result[tag] = part
            tag = None  # Reset tag for the next iteration
            
    return result


def parse_page(page, tags,name_page):
    page_parsed = page.split('\n')
    all = []
    for obs in page_parsed:
        res = parse_observation(obs,tags)
        all.append(res)
    all  = pd.DataFrame(all)
    all['name_page'] = name_page
    all = all.set_index('name_page',append=True)
    all = all.swaplevel()
    all.index.names = ['page','id']
    return all


def parse_all_pages(data,tags):
    
    results =[]
    for page in tqdm(list(data.keys())) :
        name_page = page.split('-')[-1].replace('.jpg','')
        page_clean = parse_page(data[page],tags,name_page)
        results.append(page_clean)
        
    return pd.concat(results,axis=0).dropna(axis=0,how = 'all').dropna(axis=1,how = 'all')


def complete_missing_names(data:pd.DataFrame,name_freq: pd.DataFrame):
    
    
    data['firstname_lower_enhanced'] = data['firstname_lower']
    missing_index = data[data['freq_male'].isna()].index 
    missing_names = data.loc[missing_index,'firstname_lower'].values.tolist()
    # reco = Recommender(name_freq)
    print('**************** COMPLETING MISSING NAMES with DiffLib **************')
    for i in range(len(missing_index)):
        index = missing_index[i]
        name= missing_names[i]
        # similarity = reco.get_recommendations(name).sort_values(by = "similarity",ascending=False)
        new_name = difflib.get_close_matches(name, name_freq[name_freq['total']>250]['firstname'].unique(),n =2)
        if new_name:
            name_sex = name_freq[name_freq["firstname"]== new_name[0]]['name_sex'].iloc[0]
            data.loc[index,'firstname_lower_enhanced'] = name_sex+' '+ new_name[0] 
    print(data.loc[missing_index][["firstname_lower","firstname_lower_enhanced"]])
    
    
    return data 

def get_predictions_network(names):
    model = load_model('my_model.h5')
    with open('label_encoder.pickle', 'rb') as handle:
        loaded_label_encoder = pickle.load(handle)

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    new_texts = names
    new_texts_seq = tokenizer.texts_to_sequences(new_texts)
    new_texts_padded = pad_sequences(new_texts_seq, padding='post', maxlen=8)
    predictions = model.predict(new_texts_padded)
    predicted_labels = predictions.argmax(axis=-1) 
    predicted_class_names = loaded_label_encoder.inverse_transform(predicted_labels)
    predicted_class_names
    return predicted_class_names



def complete_missing_names_lstm(data:pd.DataFrame,name_freq: pd.DataFrame):
    
    
    data['firstname_lower_lstm'] = data['firstname_lower']
    missing_index = data[data['freq_male'].isna()].index 
    missing_names = data.loc[missing_index,'firstname_lower'].values.tolist()
    predictions = get_predictions_network(missing_names)
    print('**************** COMPLETING MISSING NAMES with LSTM **************')
    for i in range(len(missing_index)):
        index = missing_index[i]
        new_name = predictions[i]
        if new_name:
            name_sex = name_freq[name_freq["firstname"]== new_name]['name_sex'].iloc[0]
            data.loc[index,'firstname_lower_lstm'] = name_sex+' '+ new_name 
    print(data.loc[missing_index][["firstname_lower","firstname_lower_lstm"]])
    
    
    return data 

def complete_missing_names_fuzzy(data:pd.DataFrame,name_freq: pd.DataFrame):

    data['firstname_lower_fuzzy'] = data['firstname_lower']
    missing_index = data[data['freq_male'].isna()].index 
    missing_names = data.loc[missing_index,'firstname_lower'].values.tolist()
    choices = name_freq[name_freq['total']>250]['firstname'].unique()
    print('**************** COMPLETING MISSING NAMES with FuzzyWizzy **************')
    for i in range(len(missing_index)):
        index = missing_index[i]
        name= missing_names[i]
        new_name = process.extract(name,choices= choices,limit=1)[0][0]
        if new_name:
            name_sex = name_freq[name_freq["firstname"]== new_name]['name_sex'].iloc[0]
            data.loc[index,'firstname_lower_fuzzy'] = name_sex+' '+ new_name 
    print(data.loc[missing_index][["firstname_lower","firstname_lower_fuzzy"]])
    
    return data 

def pipeline():
    data = import_transcriptions()
    data = drop_ambiguous(data)

    data_groundtruth = explode_column(data,'groundtruth',tags_groundtruth)
    data_prediction = explode_column(data,'prediction',tags_prediction)
    data_prediction = rename_predictions(data_prediction,mapping)

    data_prediction = modify_firstname(data_prediction)
    data_groundtruth = modify_firstname(data_groundtruth)
    data_prediction['target'] = data_prediction['sex'].apply(encode_label)
    data_groundtruth['target'] = data_groundtruth['sex'].apply(encode_label)

    name_freq = import_freq_name()
    data_prediction = merge_with_namefreq(data_prediction,name_freq)
    data_groundtruth = merge_with_namefreq(data_groundtruth,name_freq)
    data_prediction = complete_missing_names(data_prediction,name_freq)
    data_prediction = complete_missing_names_lstm(data_prediction,name_freq)
    data_prediction = complete_missing_names_fuzzy(data_prediction,name_freq)
    
    
    make_dir(PATH_DATA_TREATED)
    data_prediction.to_parquet(os.path.join(PATH_DATA_TREATED,'data_prediction.pq'))
    data_groundtruth.to_parquet(os.path.join(PATH_DATA_TREATED,'data_groundtruth.pq'))

class Recommender:
    """
    Implements a recommender system using Sentence-BERT for semantic text embedding and cosine
    similarity for generating recommendations based on textual similarity.

    Attributes:
        df (pd.DataFrame): DataFrame containing data for recommendations.
        model (SentenceTransformer): Pre-loaded Sentence-BERT model for text embedding.

    Methods:
        __init__(self, df: pd.DataFrame):
            Initializes the recommender with data and a Sentence-BERT model.

        compute_embeddings(self, texts: List[str]) -> np.ndarray:
            Computes and returns embeddings for a list of text strings.

        get_recommendations(self, query: str, similarity_threshold: float = 0.51) -> pd.DataFrame:
            Returns recommendations for a given query, based on a similarity threshold.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the Recommender with a pandas DataFrame and loads the Sentence-BERT model.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the recommendation data.
        """
        self.df = df
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Computes embeddings for a list of texts using the Sentence-BERT model.

        Parameters:
            texts (List[str]): Texts to encode into embeddings.

        Returns:
            np.ndarray: The computed embeddings.
        """
        return self.model.encode(texts)

    def get_recommendations(self, query: str, similarity_threshold: float = 0.51) -> pd.DataFrame:
        """
        Retrieves DataFrame rows as recommendations based on semantic similarity to the query.

        Parameters:
            query (str): The query text for finding similar items.
            similarity_threshold (float, optional): Threshold for cosine similarity (default: 0.51).

        Returns:
            pd.DataFrame: Recommended items.
        """
        texts = self.df['firstname'].tolist() + [query]
        embeddings = self.compute_embeddings(texts)

        query_embedding = embeddings[-1].reshape(1, -1)
        cosine_sim = cosine_similarity(query_embedding, embeddings[:-1])[0]
        print(cosine_sim)
        recommended_indices = [
            i for i, score in enumerate(cosine_sim)
            ]
        output = self.df 
        output['similarity'] = cosine_sim
        return output
    
if __name__ == "__main__" :
    pipeline()
