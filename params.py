
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