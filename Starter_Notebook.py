# %% [markdown]
# <center><img src="https://keras.io/img/logo-small.png" alt="Keras logo" width="100"><br/>
# This starter notebook is provided by the Keras team.</center>

# %% [markdown]
# # PII Data Detection with [KerasNLP](https://github.com/keras-team/keras-nlp) and [Keras](https://github.com/keras-team/keras)
# 
# > The objective of this competition is to detect and remove personally identifiable information (PII) from student writing.
# 
# <div align="center">
#     <img src="https://i.ibb.co/3stPB0t/pii-data-detection.jpg" alt="PII Data Detection">
# </div>
# 
# The task of this competition falls under **Token Classification** (not Text Classification!), sometimes known as **Named Entity Recognition (NER)**. This notebook guides you through performing this task from scratch for the competition. Implementing from scratch is a unique feature of this notebook, as most public notebooks use **HuggingFace** to handle modeling and data processing, which performs many tasks under the hood. One may have to look deeper into the repository to understand what is happening inside. In contrast, this notebook goes step by step, showing you exactly how Token Classification works. A cherry on top: this notebook leverages **Mixed Precision** and **Distributed (multi-GPU)** Training/Inference to turbocharge performance!
# 
# <u>Fun fact</u>: This notebook is backend-agnostic, supporting TensorFlow, PyTorch, and JAX. Utilizing KerasNLP and Keras allows us to choose our preferred backend. Explore more details on [Keras](https://keras.io/keras_3/).
# 
# In this notebook, you will learn how to:
# 
# - Design a data pipeline for token classification.
# - Create a model for token classification with KerasNLP.
# - Load the data efficiently using [`tf.data`](https://www.tensorflow.org/guide/data).
# - Perform Mixed Precision and Distributed Training/Inference with Keras 3.
# - Make submission on test data.
# 
# **Note**: For a more in-depth understanding of KerasNLP, refer to the [KerasNLP guides](https://keras.io/keras_nlp/).
# 

# %% [markdown]
# ## 🛠 | Install Libraries  
# 
# Since internet access is **disabled** during inference, we cannot install libraries in the usual `!pip install <lib_name>` manner. Instead, we need to install libraries from local files. In the following cell, we will install libraries from our local files. The installation code stays very similar - we just use the `filepath` instead of the `filename` of the library. So now the code is `!pip install <local_filepath>`. 
# 
# > The `filepath` of these local libraries look quite complicated, but don't be intimidated! Also `--no-deps` argument ensures that we are not installing any additional libraries.

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-03-12T19:31:54.502881Z","iopub.execute_input":"2024-03-12T19:31:54.503893Z","iopub.status.idle":"2024-03-12T19:31:59.988584Z","shell.execute_reply.started":"2024-03-12T19:31:54.503861Z","shell.execute_reply":"2024-03-12T19:31:59.987097Z"}}
!pip install -q /kaggle/input/kerasv3-lib-ds/tensorflow-2.15.0.post1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps
!pip install -q /kaggle/input/kerasv3-lib-ds/keras-3.0.4-py3-none-any.whl --no-deps

# %% [markdown]
# # 📚 | Import Libraries 

# %% [code] {"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-03-12T19:32:07.465085Z","iopub.execute_input":"2024-03-12T19:32:07.466035Z","iopub.status.idle":"2024-03-12T19:32:07.472772Z","shell.execute_reply.started":"2024-03-12T19:32:07.465997Z","shell.execute_reply":"2024-03-12T19:32:07.471705Z"}}
from pathlib import Path
import os
os.environ["KERAS_BACKEND"] = "jax" # # you can also use tensorflow or torch

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import keras
import keras_nlp
from keras import ops
import tensorflow as tf

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
import evaluate
from datasets import Dataset, features

import json
import argparse
from itertools import chain
from functools import partial

import plotly.graph_objs as go
import plotly.express as px

# %% [markdown]
# ## Library Versions

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-03-12T19:32:21.885955Z","iopub.execute_input":"2024-03-12T19:32:21.886341Z","iopub.status.idle":"2024-03-12T19:32:21.891708Z","shell.execute_reply.started":"2024-03-12T19:32:21.886310Z","shell.execute_reply":"2024-03-12T19:32:21.890740Z"}}
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasNLP:", keras_nlp.__version__)
print("hi")

# %% [markdown]
# # ⚙️ | Configuration


##||Configuration||


##OUR BLOCK
TRAINING_DATA_PATH = "/kaggle/input/pii-detect-miniset-and-validation-ds/mini_no_overlap.json"
TRAINING_MODEL_PATH = "microsoft/deberta-v3-xsmall" #pretrained backbone model
TRAINING_MAX_LENGTH = 1024 # max size of input sequence for training

#Here or in body of notebook?
# TRAIN_BATCH_SIZE = 2 * 8 # size of the input batch in training, x 2 as two GPUs
# EPOCHS = 6 # number of epochs to train
# LR_MODE = "exp" # lr scheduler mode from one of "cos", "step", "exp"

#OUR BLOCK
FINE_TUNED_NAME = "deberta3_xsmall_pii2d_1024_mini_v1"
OUTPUT_DIR = "/kaggle/working/"

#OUR BLOCK
NOTEBOOK_SEED= 42

#OUR BLOCK
LABEL_SET = ["B-EMAIL", "B-ID_NUM", "B-NAME_STUDENT", "B-PHONE_NUM",
          "B-STREET_ADDRESS", "B-URL_PERSONAL", "B-USERNAME",
          "I-ID_NUM", "I-NAME_STUDENT", "I-PHONE_NUM",
          "I-STREET_ADDRESS","I-URL_PERSONAL","O"]

# Trainer API Configs
# if commented out then it is commented out in trainer API args

#OUR BLOCK
#LR = 2e-5  # Initial learning rate
GRADIENT_ACCUMULATION_STEPS = 2  # How many batches to accumulate gradient before optimization if batch size limited by GPU memory
REPORT_TO = "none"  # Where training report progress, "none" prevents wandb login
NUM_TRAIN_EPOCHS = 2  # Number of training epochs
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # Batch size based per GPU
DO_EVAL = False  # Whether or not to perform eval during training
EVALUATION_STRATEGY = "no"  # When to evaluate during training {no, steps or epoch}
# LOGGING_DIR = OUTPUT_DIR + "/logs"  # Directory to save training logs
LOGGING_STEPS = 100  # Log training progress every X steps
# LOAD_BEST_MODEL_AT_END = True  # Load the best model at the end of training
# METRIC_FOR_BEST_MODEL = "f5"  # Metric to determine the best model ("accuracy", f1...)
# GREATER_IS_BETTER = True  # If higher eval metric is better. True for f1 and acc
SAVE_TOTAL_LIMIT = 1  # How many checkpoints to keep at end (1 means most recent)
# WARMUP_RATIO = 0.1  # Steps to gradually increase learning rate. Can help stabilize training at beginning
# WEIGHT_DECAY = 0.01  # L2 regularization to prevent overfitting

##||Data Selection||


#OUR BLOCK
#data from orginal training json
data = json.load(open(TRAINING_DATA_PATH))
org_data_df = pd.DataFrame(data)
train_df = org_data_df
print("Training Data: ", len(data))


##||Tokenization||


#OUR BLOCK
#prep data for NER training by tokenize the text and align labels to tokens
def tokenize(example, tokenizer, label2id, max_length):
    """This function ensures that the text is correctly tokenized and the labels 
    are correctly aligned with the tokens for NER training.

    Args:
        example (dict): The example containing the text and labels.
        tokenizer (Tokenizer): The tokenizer used to tokenize the text.
        label2id (dict): A dictionary mapping labels to their corresponding ids.
        max_length (int): The maximum length of the tokenized text.

    Returns:
        dict: The tokenized example with aligned labels.

    Reference: credit to https://www.kaggle.com/code/valentinwerner/915-deberta3base-training/notebook
    """

    #OUR BLOCK
    # rebuild text from tokens
    text = []
    labels = []

    #OUR BLOCK
    #iterate through tokens, labels, and trailing whitespace using zip to create tuple from three lists
    for t, l, ws in zip(
        example["tokens"], example["labels"], example["trailing_whitespace"]
    ):
        text.append(t)

        #OUR BLOCK
        #extend so we can add multiple elements to end of list if ws
        labels.extend([l] * len(t))
        if ws:
            text.append(" ")
            labels.append("O")

    #OUR BLOCK
    #Tokenize text and return offsets for start and end character position. Limit length of tokenized text.
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length, truncation=True)

    #OUR BLOCK
    #convert to np array for indexing
    labels = np.array(labels)

    #OUR BLOCK
    # join text list into a single string 
    text = "".join(text)
    token_labels = []

    #OUR BLOCK
    #iterate through each tolken
    for start_idx, end_idx in tokenized.offset_mapping:
        #if special tolken (CLS token) then append O
        #CLS : classification token added to the start of each sequence
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        #OUR BLOCK
        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        #append orginal label to token_labels
        token_labels.append(label2id[labels[start_idx]])

    #OUR BLOCK
    length = len(tokenized.input_ids)

    #OUR BLOCK
    return {**tokenized, "labels": token_labels, "length": length}


#OUR BLOCK
#Set up labeling for NER with #Targets: B-Beginning entity, I-inside entity, O- outside entity

#OUR BLOCK
#Extract all unique labels w/ list comprehension. Use chain to flatten list of lists
all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))

#OUR BLOCK
#Create dictionary of label to id
label2id = {l: i for i,l in enumerate(all_labels)}

#OUR BLOCK
#Create dictionary of id to label
id2label = {v:k for k,v in label2id.items()}

#OUR BLOCK
#target labels identified in the training data- changed to all possible target labels
target = [
    'B-NAME_STUDENT', 'B-EMAIL','B-USERNAME', 'B-ID_NUM', 'B-PHONE_NUM',
    'B-URL_PERSONAL', 'B-STREET_ADDRESS',
    'I-NAME_STUDENT', 'I-EMAIL','B-USERNAME', 'I-ID_NUM', 'I-PHONE_NUM',
    'I-URL_PERSONAL', 'I-STREET_ADDRESS', 'O'
]

#OUR BLOCK
print(id2label)

#OUR BLOCK
#load tokenizer based on pretrained model
tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

#OUR BLOCK
#convert to hugging face Dataset object
ds = Dataset.from_pandas(train_df)

#OUR BLOCK
# Map the tokenize function to your dataset
ds = ds.map(
    tokenize,
    fn_kwargs={      # pass keyword args
        "tokenizer": tokenizer,
        "label2id": label2id,
        "max_length": TRAINING_MAX_LENGTH
    }, 
    num_proc=2   #apply in paralell using 3 processes
)



##||Metrics and Training|| [Entire Section is 'Our Block']


#TODO- Review and confirm works
from seqeval.metrics import recall_score, precision_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


def compute_metrics(p, all_labels):
    """Compute the F1, recall, precision metrics for a NER task.

    Args:
        p (Tuple[np.ndarray, np.ndarray]): The predictions and labels.
        all_labels (List[str]): The list of all possible labels.

    Returns:
        Dict[str, float]: The computed metrics (recall, precision, f1_score).
    Ref: https://www.kaggle.com/code/valentinwerner/915-deberta3base-training/notebook
    """
    #Note: seqeval framework for sequence labeling like NER
    
    # Unpack the predictions and labels
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f5_score = (1 + 5*5) * recall * precision / (5*5*precision + recall)
    
    results = {
        'recall': recall,
        'precision': precision,
        'f5': f5_score
    }
    return results


#load
model = AutoModelForTokenClassification.from_pretrained(
    TRAINING_MODEL_PATH,        #pretrained model
    num_labels=len(all_labels), #num of unique labels for finetuning
    id2label=id2label,          #dicts for converting in fine tuning
    label2id=label2id,
    ignore_mismatched_sizes=True #pretrained model might have been trained on different num of labels
)

#collate list of sample from dataset into batches. 16 might be benefical for GPU architecture
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

#Configure training process
#no validation set specified
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,  # Directory to save checkpoints and logs
    fp16=True,  # mix-precision training on 16 bit to reduce memory and speed up training
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    report_to=REPORT_TO,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    do_eval=DO_EVAL,
    evaluation_strategy=EVALUATION_STRATEGY,
    logging_steps=LOGGING_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    # Uncomment the following lines if you have defined these variables in your config script
    #learning_rate=LR,
    # save_steps=SAVE_STEPS,
    # logging_dir=LOGGING_DIR,
    # load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
    # metric_for_best_model=METRIC_FOR_BEST_MODEL,
    # greater_is_better=GREATER_IS_BETTER,
    # lr_scheduler_type=LR_SCHEDULER_TYPE,
    # warmup_ratio=WARMUP_RATIO,
    # weight_decay=WEIGHT_DECAY,
)

#inialize trainer for training and evaluation interface
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=ds,
    data_collator=collator, 
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, all_labels=all_labels), #partial to fix all_label argument
)

%%time

#train model 
trainer.train()

trainer.save_model(FINE_TUNED_NAME)
tokenizer.save_pretrained(FINE_TUNED_NAME)



##||Next Sections|| 



# %% [markdown]
# # 🔗 | Reference
# * [Detect Fake Text: KerasNLP [TF/Torch/JAX][Train]](https://www.kaggle.com/code/awsaf49/detect-fake-text-kerasnlp-tf-torch-jax-train)
# * [Token classification](https://huggingface.co/docs/transformers/en/tasks/token_classification)
# * [transformer ner baseline [lb 0.854]](https://www.kaggle.com/code/nbroad/transformer-ner-baseline-lb-0-854)
