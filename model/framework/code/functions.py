'''
Function script for clean importing and calling to
'''
import pandas as pd
import numpy as np
from mordred import Calculator, descriptors
from standardise_smiles import standardize_jumpcp
from rdkit import Chem
import pickle
import csv


def load_model(model_path):
   with open(model_path, 'rb') as file:
       classifier = pickle.load(file)
   return classifier


def load_data_columns(columns_path):
   with open(columns_path, 'rb') as file:
       data_columns = pickle.load(file)
   return data_columns


def preprocess_smiles(smiles_list):
   df = pd.DataFrame({'SMILES': smiles_list})
   df['Standardized_SMILES'] = df['SMILES'].apply(standardize_jumpcp)
   return df


def generate_mordred_descriptors(df, data_columns):
   calc = Calculator(descriptors, ignore_3D=True)
   Ser_Mol = df['Standardized_SMILES'].apply(Chem.MolFromSmiles)
   Mordred_table = calc.pandas(Ser_Mol).astype('float')
   Mordred_table = Mordred_table[data_columns]
   return Mordred_table


def run_predictions(classifier, df, data_columns):
   threshold = 0.641338  # Fixed threshold
   Mordred_table = generate_mordred_descriptors(df, data_columns)
   X = np.array(Mordred_table)
   X[np.isnan(X)] = 0
   X[np.isinf(X)] = 0
   prob_test = classifier.predict_proba(X)[:, 1]
   predictions = (prob_test >= threshold).astype(int)
   return prob_test, predictions



