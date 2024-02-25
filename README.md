### PII-Detection
Kaggle PII Automated data detection and removal compettion

## Files
 1. Inference notebook- apply model (finetuned or not) to data. Does not update model. Does not require GPU to run? 
    Dependancies- model.py (updated or not), test_data.json
    Input- Test data. 
    Output- predictions.

 2. Training notebook- Fine tune and optimize given model (e.g. deBERTa) using labeled data. Resource intensive, GPU
    Dependancies- model.py (base model)
    Input- Train_X_data, Train_label_data
    output- fine-tuned model.py

## Overview
The goal of this competition is to develop a model that detects personally identifiable information (PII) in student writing. Your efforts to automate the detection and removal of PII from educational data will lower the cost of releasing educational datasets. This will support learning science research and the development of educational tools.

Reliable automated techniques could allow researchers and industry to tap into the potential that large public educational datasets offer to support the development of effective tools and interventions for supporting teachers and students.

# Evaluation
Submissions are evaluated on micro F(Beta), which is a classification metric that assigns value to recall and precision. The value of Beta is set to 5, which means that recall is weighted 5 times more heavily than precision.

# Citations
1. Langdon Holmes, Scott Crossley, Perpetual Baffour, Jules King, Lauryn Burleigh, Maggie Demkin, Ryan Holbrook, Walter Reade, Addison Howard. (2024). The Learning Agency Lab - PII Data Detection. Kaggle. https://kaggle.com/competitions/pii-detection-removal-from-educational-data

2. Repo organization from https://drivendata.github.io/cookiecutter-data-science/#why-use-this-project-structure 

# Reference Notebooks
 1. https://www.kaggle.com/code/nbroad/transformer-ner-baseline-lb-0-881 for baseline deBERTa

 2. 
