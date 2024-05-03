### PII-Detection
Kaggle PII Automated data detection and removal compettion

## Usage
Notebooks are written to be run on kaggle and will require modification to run on alternate platforms or local machine. W&B is used throughout but can be removed if report_to = 'none'

## Files
 1. Inference notebook- apply finetuned model to data and does post processing on predictions. Does not update model.

 2. Training notebook- Fine tune and optimize given model (e.g. deBERTa) using labeled data. Resource intensive, GPU

 3. Preparation notebook- Creates supporting datasets for PII detection training and validation

 4. EDA Notebook - Anlaysis of the competition training dataset. 

## Overview
The goal of this competition is to develop a model that detects personally identifiable information (PII) in student writing. Your efforts to automate the detection and removal of PII from educational data will lower the cost of releasing educational datasets. This will support learning science research and the development of educational tools.

Reliable automated techniques could allow researchers and industry to tap into the potential that large public educational datasets offer to support the development of effective tools and interventions for supporting teachers and students.

# Evaluation
Submissions are evaluated on micro F(Beta), which is a classification metric that assigns value to recall and precision. The value of Beta is set to 5, which means that recall is weighted 5 times more heavily than precision.

# Citations
1. Langdon Holmes, Scott Crossley, Perpetual Baffour, Jules King, Lauryn Burleigh, Maggie Demkin, Ryan Holbrook, Walter Reade, Addison Howard. (2024). The Learning Agency Lab - PII Data Detection. Kaggle. https://kaggle.com/competitions/pii-detection-removal-from-educational-data


# Reference Notebooks
 1. https://www.kaggle.com/code/nbroad/transformer-ner-baseline-lb-0-881 for baseline deBERTa

