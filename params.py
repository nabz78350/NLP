
PATH_DATA = 'raw_data/'
PATH_DATA_TREATED = 'clean_data/'

tags_prediction = ['date_naissance',
                   'prénom',
                   'nom',
                   'lieux_naissance',
                   'relation',
                   'profession',
                   'id',
                   'état_civil',
                   'employeur',
                   ]

tags_groundtruth = [
    'surname',
    'firstname',
    'link',
    'age',
    'occupation',
    'employer',
    'birth_date',
    'lob'
]

mapping = {'nom': 'surname',
            'prénom': 'firstname',
            'relation': 'link',
            'date_naissance': 'age',
            'profession': 'occupation',
            'employeur': 'employer',
            'lieux_naissance': 'lob'}

gender_map = {
    'Garçon': 'homme',
    'Garçon ': 'homme',
    'Homme marié': 'homme',
    'Homme marié ': 'homme',
    'Veuf': 'homme',
    'Veuf ': 'homme',
    'Fille': 'femme',
    'Fille ': 'femme',
    'Femme mariée': 'femme',
    'Femme mariée ': 'femme',
    'Veuve': 'femme',
    'Veuve ': 'femme',
    None: None
}

FEATURES = [
    "firstname_lower",
    "link",
    "employer",
    "occupation",
    "name_sex"]


knn_args = {"n_neighbors":5,
            "p": 2
            }

xgb_args = {"max_depth":6,
            "n_estimators":100,
            "n_jobs":-1
            }

mlp_args = {"vocab_size": 500,
            "embed_size": 500,
            "n_layers": 3,
            "hidden_size":20,
            "hidden_function": "relu",
            "dropout_rate":0.1,
            "max_document_length":20,
            "num_epochs":200,
            "batch_size":32,
            "lr": 3e-4
            
            }