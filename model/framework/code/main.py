import os
import sys
import pandas as pd
import numpy as np
import csv
from functions import load_model, load_data_columns, preprocess_smiles, run_predictions


if __name__ == '__main__':
   input_file = sys.argv[1]
   output_file = sys.argv[2]


   root = os.path.dirname(os.path.abspath(__file__))
   checkpoints_dir = os.path.abspath(os.path.join(root, "..", "..", "checkpoints"))
   model_path = os.path.join(checkpoints_dir, 'FINAL_Physicochemical_model.sav')
   columns_path = os.path.join(checkpoints_dir, 'data_columns.pkl')


   classifier = load_model(model_path)
   data_columns = load_data_columns(columns_path)


   smiles_list = []
   with open(input_file, "r") as f:
       reader = csv.reader(f)
       next(reader)  # skip header
       for row in reader:
           smiles_list.append(row[0])
   # Error Checking
   if len(smiles_list) == 0:
       print("Error: No valid SMILES strings found in the input file.")
       sys.exit(1)
   # Run predictions
   df = preprocess_smiles(smiles_list)
   probabilities, predictions = run_predictions(classifier, preprocess_smiles(smiles_list), data_columns)
   input_len = len(smiles_list)
   output_len = len(probabilities)
   assert input_len == output_len, "Input and output lengths do not match"

    
   with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Probability", "Prediction"])  # Header
        for prob, pred in zip(probabilities, predictions):
            writer.writerow([prob, pred])

  