import numpy as np
import pandas as pd

# removes the undesirable instances from the dataframe
def clean(df):
    # dropping records which do not have values in columns standard_value and canonical_smiles
    df = df[df.standard_value.notna()]
    # convert standard value into floats and drop non-postives
    df['standard_value'] = df['standard_value'].apply(lambda x: float(x))
    df = df[df.standard_value > 0.0]
    df = df[df.canonical_smiles.notna()]

    #dropping records with duplicate canonical_smiles values to keep them unique
    df_unique = df.drop_duplicates(['canonical_smiles'])
    selection = ['molecule_chembl_id','canonical_smiles','standard_value']
    df = df_unique[selection]
    df['plC50']= df['standard_value'].apply(lambda x: np.log10(x))

    return df

# prepares the data frame for training
def transform(df):
    n=df.shape[0]

    fp= df['canonical_smiles_fingerprints'].to_numpy().T
    y= df['plC50'].to_numpy().reshape(n,1)

    # This step is to convert the fingerprint object into 2048 columns of numbers in the dataframe.
    x = np.zeros((n,2048),dtype=int)
    for i in range(n):
        for j in range(2048):
            x[i,j] = int(fp[i][j])


    data = pd.DataFrame(np.concatenate((x,y), axis =1))

    return data