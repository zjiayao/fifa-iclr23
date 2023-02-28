import os
import requests
import tempfile
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def get_dummied_df(df, bin_cols, excl_cols=[]):
    all_cols = df.columns
    dummy_cols = [c for c in df.columns if df.dtypes[c] == 'O' and c not in bin_cols+excl_cols]
    to_drop_cols = dummy_cols+bin_cols
    d1 = pd.get_dummies(df,columns=dummy_cols, drop_first=False).drop(all_cols,axis=1,errors='ignore')
    d2 = pd.get_dummies(df,columns=bin_cols, drop_first=True).drop(all_cols,axis=1,errors='ignore')
    return pd.concat([df.drop(to_drop_cols, errors='ignore',axis=1), d1,d2],axis=1)

def load_dutch_data(path, seed=1):
    if not isinstance(path, Path):
        path = Path(path)
    dutch_df = pd.read_csv(path)
    dutch_df['label'] = dutch_df['NL2001A_OCC'].isin([1,2]).astype(int)
    dutch_df=dutch_df.rename({'NL2001A_SEX':"gender"},axis=1)
    dutch_df=dutch_df.replace({'gender': {1:'Male', 2:'Female'}})
    dutch_df=dutch_df.drop(['COUNTRY','YEAR','SAMPLE','SERIAL','NL2001A_PERNUM','NL2001A_DWNUM','PERNUM', 'NL2001A_OCC', ],axis=1)
    
    dummy_cols = ['NL2001A_RELATE', 'NL2001A_CITZ','NL2001A_BPL','NL2001A_EDUC','NL2001A_CLASSWK','NL2001A_IND', 'NL2001A_MARST']
    dutch_df[dummy_cols]=dutch_df[dummy_cols].astype(str)
    A = dutch_df['gender'].copy()
    y = dutch_df['label'].copy()
    dutch_df_vals = get_dummied_df(dutch_df, [],['HHWT','PERWT','NL2001A_WEIGHT','label', 'gender']).drop(['gender','label'],axis=1)

    tr_idx, te_idx = train_test_split(np.arange(len(y)), train_size=0.8,random_state=seed)
    return [
        dutch_df_vals.iloc[tr_idx].reset_index(drop=True).astype(float),
        y.iloc[tr_idx].reset_index(drop=True).astype(float),
        A.iloc[tr_idx].reset_index(drop=True),
        dutch_df_vals.iloc[te_idx].reset_index(drop=True).astype(float),
        y.iloc[te_idx].reset_index(drop=True).astype(float),
        A.iloc[te_idx].reset_index(drop=True),
    ]

def load_adult_income_dataset(path):
    if not isinstance(path, Path):
        path = Path(path)
    adult_income_cols = ['age', 
                         'workclass',
                         'fnlwgt',
                         'education',
                         'education-num',
                         'marital-status',
                         'occupation',
                         'relationship',
                         'race',
                         'sex',
                         'capital-gain',
                         'capital-loss',
                         'hours-per-week',
                         'native-country',
                         'label'
                        ]
    adult_income = pd.read_csv(path / "adult.data",header=None)
    adult_income_test = pd.read_csv(path / "adult.test")
    adult_income.columns=adult_income_cols
    adult_income_test=adult_income_test.reset_index()
    adult_income_test.columns=adult_income_cols
    
    adult_income_rename_dict = {'label_ >50K':'label', 'sex_ Male': 'sex', 'label_ >50K.':'label'}
    adult_income_d = get_dummied_df(adult_income,['label'], ['sex']).rename(adult_income_rename_dict,axis=1)
    adult_income_test_d = get_dummied_df(adult_income_test,['label'], ['sex']).rename(adult_income_rename_dict,axis=1)
    tr_cols = adult_income_d.columns
    te_cols = adult_income_test_d.columns
    tr_new_cols = [c for c in te_cols if c not in tr_cols]
    te_new_cols = [c for c in tr_cols if c not in te_cols]
    for c in tr_new_cols:
        adult_income_d[c] = 0.
    for c in te_new_cols:
        adult_income_test_d[c] = 0.

    adult_income_test_d = adult_income_test_d[adult_income_d.columns]
    gender_replace_dict = {' Male':'Male',' Female': 'Female'}
    return [
        adult_income_d.drop(['label','sex'],axis=1),
        adult_income_d['label'].astype(float),
        adult_income_d['sex'].replace(gender_replace_dict),
        adult_income_test_d.drop(['label','sex'],axis=1),
        adult_income_test_d['label'].astype(float),
        adult_income_test_d['sex'].replace(gender_replace_dict),
    ]

def load_lawschool_data(target):
    """ Downloads SEAPHE lawschool data from the SEAPHE webpage.
    For more information refer to http://www.seaphe.org/databases.php

    :param target: the name of the target variable, either pass_bar or zfygpa
    :type target: str
    :return: pandas.DataFrame with columns
    """
    if target not in ['pass_bar', 'zfygpa']:
        raise ValueError("Only pass_bar and zfygpa are supported targets.")

    with tempfile.TemporaryDirectory() as temp_dir:
        response = requests.get("http://www.seaphe.org/databases/LSAC/LSAC_SAS.zip")
        temp_file_name = os.path.join(temp_dir, "LSAC_SAS.zip")
        with open(temp_file_name, "wb") as temp_file:
            temp_file.write(response.content)
        with zipfile.ZipFile(temp_file_name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        data = pd.read_sas(os.path.join(temp_dir, "lsac.sas7bdat"))

        # data contains 'sex', 'gender', and 'male' which are all identical except for the type;
        # map string representation of feature "sex" to 0 for Female and 1 for Male
        data = data.assign(gender=(data["gender"] == b"male") * 1)

        # filter out all records except the ones with the most common two races
        data = data.assign(white=(data["race1"] == b"white") * 1)
        data = data.assign(black=(data["race1"] == b"black") * 1)
        data = data[(data['white'] == 1) | (data['black'] == 1)]

        # encode dropout as 0/1
        data = data.assign(dropout=(data["Dropout"] == b"YES") * 1)

        if target == 'pass_bar':
            # drop NaN records for pass_bar
            data = data[(data['pass_bar'] == 1) | (data['pass_bar'] == 0)]
        elif target == 'zfygpa':
            # drop NaN records for zfygpa
            data = data[np.isfinite(data['zfygpa'])]

        # drop NaN records for features
        data = data[np.isfinite(data["lsat"]) & np.isfinite(data['ugpa'])]

        # Select relevant columns for machine learning.
        # We explicitly leave in age_cat to allow linear classifiers to be non-linear in age
        # TODO: consider using 'fam_inc', 'age', 'parttime', 'dropout'
        data = data[['white', 'black', 'gender', 'lsat', 'ugpa', target]]

    return data

def get_law_school_pass_bar_data():
    # Extract into X, y and A
    data = load_lawschool_data('pass_bar')
    X = data[['lsat', 'ugpa']]
    y = data['pass_bar']
    A = data['white'].apply(lambda x: 'white' if x==1 else 'black').rename('race')

    # Split into test and train, making sure we have sequential indices in the results
    X_train, X_test, y_train, y_test, A_train, A_test = \
        train_test_split(X, y, A, test_size=0.33, random_state=123)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    A_train = A_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    A_test = A_test.reset_index(drop=True)

    return [X_train, y_train, A_train, X_test, y_test, A_test]

