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



##||Metrics and Training||








# %% [code] {"execution":{"iopub.status.busy":"2024-03-12T19:32:32.928899Z","iopub.execute_input":"2024-03-12T19:32:32.929669Z","iopub.status.idle":"2024-03-12T19:32:32.936772Z","shell.execute_reply.started":"2024-03-12T19:32:32.929639Z","shell.execute_reply":"2024-03-12T19:32:32.935709Z"}}
class CFG:
    seed = 42
    preset = "deberta_v3_small_en" # name of pretrained backbone
    train_seq_len = 1024 # max size of input sequence for training
    train_batch_size = 2 * 8 # size of the input batch in training, x 2 as two GPUs
    infer_seq_len = 2000 # max size of input sequence for inference
    infer_batch_size = 2 * 2 # size of the input batch in inference, x 2 as two GPUs
    epochs = 6 # number of epochs to train
    lr_mode = "exp" # lr scheduler mode from one of "cos", "step", "exp"
    
    labels = ["B-EMAIL", "B-ID_NUM", "B-NAME_STUDENT", "B-PHONE_NUM",
              "B-STREET_ADDRESS", "B-URL_PERSONAL", "B-USERNAME",
              "I-ID_NUM", "I-NAME_STUDENT", "I-PHONE_NUM",
              "I-STREET_ADDRESS","I-URL_PERSONAL","O"]
    id2label = dict(enumerate(labels)) # integer label to BIO format label mapping
    label2id = {v:k for k,v in id2label.items()} # BIO format label to integer label mapping
    num_labels = len(labels) # number of PII (NER) tags
    
    train = True # whether to train or use already trained ckpt

# %% [markdown]
# # ♻️ | Reproducibility 
# Sets value for random seed to produce similar result in each run.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-12T19:32:45.452877Z","iopub.execute_input":"2024-03-12T19:32:45.453250Z","iopub.status.idle":"2024-03-12T19:32:45.458397Z","shell.execute_reply.started":"2024-03-12T19:32:45.453220Z","shell.execute_reply":"2024-03-12T19:32:45.457462Z"}}
keras.utils.set_random_seed(CFG.seed)

# %% [markdown]
# # 🚀 | Distributed Training / Inference
# 
# In this notebook, we will also use the `Data Parallel` strategy for **Distributed Training/Inference**. This means that the model weights will be replicated across all devices, and each device will process a portion of the input data.
# 
# > **Note**: Currently, `DataParallel` is implemented on the JAX backend, so for TensorFlow and PyTorch backends, we can only use a single GPU.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-12T19:32:52.497202Z","iopub.execute_input":"2024-03-12T19:32:52.497542Z","iopub.status.idle":"2024-03-12T19:32:53.119773Z","shell.execute_reply.started":"2024-03-12T19:32:52.497514Z","shell.execute_reply":"2024-03-12T19:32:53.118906Z"}}
# Get devices default "gpu" or "tpu"
devices = keras.distribution.list_devices()
print("Device:", devices)

if len(devices) > 1:
    # Data parallelism
    data_parallel = keras.distribution.DataParallel(devices=devices)

    # Set the global distribution.
    keras.distribution.set_distribution(data_parallel)

# %% [markdown]
# # 🧮 | Mixed Precision
# 
# To enable larger batch sizes and faster training, we'll utilize `mixed_precision` in this notebook. In Keras, this can be achieved with just **one line of code**, as shown below.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-12T19:33:01.120529Z","iopub.execute_input":"2024-03-12T19:33:01.121257Z","iopub.status.idle":"2024-03-12T19:33:01.125763Z","shell.execute_reply.started":"2024-03-12T19:33:01.121220Z","shell.execute_reply":"2024-03-12T19:33:01.124787Z"}}
keras.mixed_precision.set_global_policy("mixed_float16") 

# %% [markdown]
# # 📁 | Dataset Path 

# %% [code] {"execution":{"iopub.status.busy":"2024-03-12T19:33:48.936890Z","iopub.execute_input":"2024-03-12T19:33:48.937651Z","iopub.status.idle":"2024-03-12T19:33:48.941879Z","shell.execute_reply.started":"2024-03-12T19:33:48.937618Z","shell.execute_reply":"2024-03-12T19:33:48.940866Z"}}
BASE_PATH = "/kaggle/input/pii-detection-removal-from-educational-data"

# %% [markdown]
# # 📖 | Meta Data
# 
# The competition dataset contains ~$22,000$ student essays where $70\%$ essays are reserved for **testing**, leaving $30\%$ for **training** and **validation**.
# 
# Sure, here's the modified markdown with an example of the BIO format label:
# 
# **Data Overview:**
# 
# * All essays were written in response to the **same prompt**, applying course material to a real-world problem.
# * The dataset includes **7 types of PII**: `NAME_STUDENT`, `EMAIL`, `USERNAME`, `ID_NUM`, `PHONE_NUM`, `URL_PERSONAL`, `STREET_ADDRESS`.
# * Labels are given in **BIO (Beginning, Inner, Outer)** format.
# 
# **Example of BIO format label:**
# 
# Let's consider a sentence: `"The email address of Michael jordan is mjordan@nba.com"`. In BIO format, the labels for the personally identifiable information (PII) would be annotated as follows:
# 
# | **Word** | The | email | address | of | Michael | Jordan | is | mjordan@nba.com |
# |----------|-----|-------|---------|----|---------|--------|----|----------------|
# | **Label** | O   | O     | O       | O  | B-NAME_STUDENT | I-NAME_STUDENT | O  | B-EMAIL        |
# 
# In the example above, `B-` indicates the beginning of an PII, `I-` indicates an inner part of a multi-token PII, and `O` indicates tokens that do not belong to any PII.
# 
# **Data Format:**
# 
# * The train/test data is stored in `{test|train}.json` files.
# * Each json file has:
#     * `document`: unique ID (integer)
#     * `full_text`: essay content (string)
#     * `tokens`: individual words in the essay (list of strings)
#     * `labels` (training data only): BIO labels for each token (list of strings)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-12T19:33:22.938348Z","iopub.execute_input":"2024-03-12T19:33:22.938990Z","iopub.status.idle":"2024-03-12T19:33:28.357124Z","shell.execute_reply.started":"2024-03-12T19:33:22.938956Z","shell.execute_reply":"2024-03-12T19:33:28.356167Z"}}
# Train-Valid data
data = json.load(open(f"{BASE_PATH}/train.json"))

# Initialize empty arrays
words = np.empty(len(data), dtype=object)
labels = np.empty(len(data), dtype=object)

# Fill the arrays
for i, x in tqdm(enumerate(data), total=len(data)):
    words[i] = np.array(x["tokens"])
    labels[i] = np.array([CFG.label2id[label] for label in x["labels"]])

# %% [markdown]
# # 📊 | Exploratory Data Analysis
# 
# From the following label distribution plot, it is evident that there is a significant **class imbalance** between PII tags. This could be a key area for improvement where **external datasets** and **augmentations** could play a pivotal role.

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-02-16T07:40:06.058598Z","iopub.execute_input":"2024-02-16T07:40:06.059001Z","iopub.status.idle":"2024-02-16T07:40:07.537372Z","shell.execute_reply.started":"2024-02-16T07:40:06.058974Z","shell.execute_reply":"2024-02-16T07:40:07.536506Z"}}
# Get unique labels and their frequency
all_labels = np.array([x for label in labels for x in label])
unique_labels, label_counts = np.unique(all_labels, return_counts=True)

# Plotting
fig = go.Figure(data=go.Bar(x=CFG.labels, y=label_counts))
fig.update_layout(
    title="Label Distribution",
    xaxis_title="Labels",
    yaxis_title="Count",
    yaxis_type="log",
)

fig.update_traces(text=label_counts, textposition="outside")
fig.show()


# %% [markdown]
# ## 🔪 | Data Split
# 
# In the following code snippet, we will split the dataset into training and testing subsets using an `80%-20%` ratio.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-16T07:40:07.538675Z","iopub.execute_input":"2024-02-16T07:40:07.538958Z","iopub.status.idle":"2024-02-16T07:40:07.545492Z","shell.execute_reply.started":"2024-02-16T07:40:07.538933Z","shell.execute_reply":"2024-02-16T07:40:07.544538Z"}}
# Splitting the data into training and testing sets
train_words, valid_words, train_labels, valid_labels = train_test_split(
    words, labels, test_size=0.2, random_state=CFG.seed
)

# %% [markdown]
# # 🍽️ | Pre-Processing
# 
# Initially, raw text data is quite complex and challenging for modeling due to its high dimensionality. We simplify this complexity by converting text into words then more manageable set of tokens using `tokenizers`. For example, transforming the sentence `"The quick brown fox"` into tokens like `["the", "qu", "##ick", "br", "##own", "fox"]` helps us break down the text effectively. Then, since models can't directly process strings, they are converted into integers, like `[10, 23, 40, 51, 90, 84]`. Additionally, many models require special tokens and additional tensors to understand input better. A `preprocessing` layer helps with this by adding these special tokens, which aid in separating input and identifying padding, among other tasks.
# 
# You can explore the following pages to access the available preprocessing and tokenizer layers in **KerasNLP**:
# - [Tokenizers](https://keras.io/api/keras_nlp/tokenizers/)
# - [Preprocessing](https://keras.io/api/keras_nlp/preprocessing_layers/)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-16T07:40:07.54673Z","iopub.execute_input":"2024-02-16T07:40:07.546997Z","iopub.status.idle":"2024-02-16T07:40:08.998705Z","shell.execute_reply.started":"2024-02-16T07:40:07.546975Z","shell.execute_reply":"2024-02-16T07:40:08.997878Z"}}
# To convert string input or list of strings input to numerical tokens
tokenizer = keras_nlp.models.DebertaV3Tokenizer.from_preset(
    CFG.preset,
)

# Preprocessing layer to add spetical tokens: [CLS], [SEP], [PAD]
packer = keras_nlp.layers.MultiSegmentPacker(
    start_value=tokenizer.cls_token_id,
    end_value=tokenizer.sep_token_id,
    sequence_length=10,
)

# %% [markdown]
# ## Tokenizer in Action
# 
# The following code shows the effects of `DebertaV3Tokenizer`. We can see that the word `["reflexion"]` has been divided into `["▁reflex", "ion"]` tokens. Therefore, for similar cases, it's necessary to align labels of tokens to labels of words.

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-02-16T07:40:08.999835Z","iopub.execute_input":"2024-02-16T07:40:09.000136Z","iopub.status.idle":"2024-02-16T07:40:10.175232Z","shell.execute_reply.started":"2024-02-16T07:40:09.00011Z","shell.execute_reply":"2024-02-16T07:40:10.17428Z"}}
sample_words = words[0][:5]
sample_tokens_int = [
    token.tolist() for word in sample_words for token in tokenizer(word)
]
sample_tokens_str = [tokenizer.id_to_token(token) for token in sample_tokens_int]

print("words        :", sample_words.tolist())
print("tokens (str) :", sample_tokens_str)
print("tokens (int) :", sample_tokens_int)


# %% [markdown]
# ## Preprocessor in Action
# 
# Even though we converted string inputs to integer tokens with Tokenizer, we are not done yet. We need to add special tokens like `[CLS]`, `[SEP]`, `[PAD]`. This is wehere `Preprocessing` layer comes into the picture. In this notebook, we will use `MultiSegmentPacker` layer. Let's see it action.

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-02-16T07:40:10.179401Z","iopub.execute_input":"2024-02-16T07:40:10.179715Z","iopub.status.idle":"2024-02-16T07:40:10.271891Z","shell.execute_reply.started":"2024-02-16T07:40:10.179688Z","shell.execute_reply":"2024-02-16T07:40:10.270913Z"}}
padded_sample_tokens_int = packer(np.array(sample_tokens_int))[0].tolist()
padded_sample_tokens_str = [
    tokenizer.id_to_token(token) for token in padded_sample_tokens_int
]

print("tokens (str)        :", sample_tokens_str)
print("padded tokens (str) :", padded_sample_tokens_str, "\n")

print("tokens (int)        :", sample_tokens_int)
print("padded tokens (int) :", padded_sample_tokens_int)


# %% [markdown]
# # 🥣 | Data Processing
# 
# One of the key factors that sets Token Classification apart from Text Classification is the data processing part. Unlike text classification, where we simply send our tokenized text to the model, in token classification, we have to apply more processing before sending it to the model. For example, when the `tokenizer` creates multiple tokens for single word or the `processing` layer adds special tokens `[CLS]`, `[SEP]`, and `[PAD]`, they create a mismatch between the input and labels. Thus, a single word corresponding to a single label may now be split into two tokens. We need to realign the tokens labels with word labels by:
# 
# - Mapping tokens label to their corresponding word label using `token_ids`.
# - Assigning the label `-100` to special tokens `[CLS]`, `[SEP]` and `[PAD]` to disregard them in the `CrossEntropy` loss calculation.
# - Labeling only the first token of each word and assigning `-100` to other tokens belonging to the same word.
# 
# Specifically, the following cell contains the following functions:
# - `process_data()` - prepares input, label, and token_ids
#     - `get_tokens()` - creates input tokens ans padding masks from string words
#     - `get_token_ids()` - generates token ids for aligning tokens and labels
#     - `get_token_labels()` - realigns token labels and adds padding (`-100`) to match input
#     - `process_token_ids()` - adds padding (`-1`) to token ids to match input

# %% [code] {"execution":{"iopub.status.busy":"2024-02-16T07:40:10.273112Z","iopub.execute_input":"2024-02-16T07:40:10.27347Z","iopub.status.idle":"2024-02-16T07:40:10.28844Z","shell.execute_reply.started":"2024-02-16T07:40:10.273438Z","shell.execute_reply":"2024-02-16T07:40:10.287574Z"}}
def get_tokens(words, seq_len, packer):
    # Tokenize input
    token_words = tf.expand_dims(
        tokenizer(words), axis=-1
    )  # ex: (words) ["It's", "a", "cat"] ->  (token_words) [[1, 2], [3], [4]]
    tokens = tf.reshape(
        token_words, [-1]
    )  # ex: (token_words) [[1, 2], [3], [4]] -> (tokens) [1, 2, 3, 4]
    # Pad tokens
    tokens = packer(tokens)[0][:seq_len]
    inputs = {"token_ids": tokens, "padding_mask": tokens != 0}
    return inputs, tokens, token_words


def get_token_ids(token_words):
    # Get word indices
    word_ids = tf.range(tf.shape(token_words)[0])
    # Get size of each word
    word_size = tf.reshape(tf.map_fn(lambda word: tf.shape(word)[0:1], token_words), [-1])
    # Repeat word_id with size of word to get token_id
    token_ids = tf.repeat(word_ids, word_size)
    return token_ids


def get_token_labels(word_labels, token_ids, seq_len):
    # Create token_labels from word_labels ->  alignment
    token_labels = tf.gather(word_labels, token_ids)
    # Only label the first token of a given word and assign -100 to others
    mask = tf.concat([[True], token_ids[1:] != token_ids[:-1]], axis=0)
    token_labels = tf.where(mask, token_labels, -100)
    # Truncate to max sequence length
    token_labels = token_labels[: seq_len - 2]  # -2 for special tokens ([CLS], [SEP])
    # Pad token_labels to align with tokens (use -100 to pad for loss/metric ignore)
    pad_start = 1  # for [CLS] token
    pad_end = seq_len - tf.shape(token_labels)[0] - 1  # for [SEP] and [PAD] tokens
    token_labels = tf.pad(token_labels, [[pad_start, pad_end]], constant_values=-100)
    return token_labels


def process_token_ids(token_ids, seq_len):
    # Truncate to max sequence length
    token_ids = token_ids[: seq_len - 2]  # -2 for special tokens ([CLS], [SEP])
    # Pad token_ids to align with tokens (use -1 to pad for later identification)
    pad_start = 1  # [CLS] token
    pad_end = seq_len - tf.shape(token_ids)[0] - 1  # [SEP] and [PAD] tokens
    token_ids = tf.pad(token_ids, [[pad_start, pad_end]], constant_values=-1)
    return token_ids


def process_data(seq_len=720, has_label=True, return_ids=False):
    # To add spetical tokens: [CLS], [SEP], [PAD]
    packer = keras_nlp.layers.MultiSegmentPacker(
        start_value=tokenizer.cls_token_id,
        end_value=tokenizer.sep_token_id,
        sequence_length=seq_len,
    )

    def process(x):
        # Generate inputs from tokens
        inputs, tokens, words_int = get_tokens(x["words"], seq_len, packer)
        # Generate token_ids for maping tokens to words
        token_ids = get_token_ids(words_int)
        if has_label:
            # Generate token_labels from word_labels
            token_labels = get_token_labels(x["labels"], token_ids, seq_len)
            return inputs, token_labels
        elif return_ids:
            # Pad token_ids to align with tokens
            token_ids = process_token_ids(token_ids, seq_len)
            return token_ids
        else:
            return inputs

    return process

# %% [markdown]
# # 🍚 | Dataloader
# 
# The code below sets up a data flow pipeline using `tf.data.Dataset` for data processing. Notable aspects of `tf.data` include its ability to simplify pipeline construction and represent components in sequences. To learn more about `tf.data`, refer to this [documentation](https://www.tensorflow.org/guide/data).
# 
# > **Note**: We have used `ragged` tensor as each row has text with different sizes.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-16T07:40:10.289907Z","iopub.execute_input":"2024-02-16T07:40:10.290301Z","iopub.status.idle":"2024-02-16T07:40:10.304393Z","shell.execute_reply.started":"2024-02-16T07:40:10.290267Z","shell.execute_reply":"2024-02-16T07:40:10.303588Z"}}
def build_dataset(words, labels=None, return_ids=False, batch_size=4,
                  seq_len=512, shuffle=False, cache=True, drop_remainder=True):
    AUTO = tf.data.AUTOTUNE 

    slices = {"words": tf.ragged.constant(words)}
    if labels is not None:
        slices.update({"labels": tf.ragged.constant(labels)})

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(process_data(seq_len=seq_len,
                             has_label=labels is not None, 
                             return_ids=return_ids), num_parallel_calls=AUTO) # apply processing
    ds = ds.cache() if cache else ds  # cache dataset
    if shuffle: # shuffle dataset
        ds = ds.shuffle(1024, seed=CFG.seed)  
        opt = tf.data.Options() 
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)  # batch dataset
    ds = ds.prefetch(AUTO)  # prefetch next batch
    return ds

# %% [markdown]
# ## Build Train & Valid Dataloader
# 
# In the following code, we'll create **train** and **valid** data loaders.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-16T07:40:10.305368Z","iopub.execute_input":"2024-02-16T07:40:10.305617Z","iopub.status.idle":"2024-02-16T07:40:42.695659Z","shell.execute_reply.started":"2024-02-16T07:40:10.305596Z","shell.execute_reply":"2024-02-16T07:40:42.694869Z"}}
train_ds = build_dataset(train_words, train_labels,  batch_size=CFG.train_batch_size,
                         seq_len=CFG.train_seq_len, shuffle=True)

valid_ds = build_dataset(valid_words, valid_labels, batch_size=CFG.train_batch_size, 
                         seq_len=CFG.train_seq_len, shuffle=False)

# %% [markdown]
# ## Dataset Check
# 
# Let's check a batch of samples and their associated labels from the dataset.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-16T07:40:42.696806Z","iopub.execute_input":"2024-02-16T07:40:42.697084Z","iopub.status.idle":"2024-02-16T07:40:43.304332Z","shell.execute_reply.started":"2024-02-16T07:40:42.69706Z","shell.execute_reply":"2024-02-16T07:40:43.303303Z"}}
inp, tar = next(iter(valid_ds))
print("# Input:\n",inp); print("\n# Labels:\n",tar)

# %% [markdown]
# # 🔍 | Loss & Metric

# %% [markdown]
# ## Loss: CrossEntropy
# 
# To optimize our model we will use `CrossEntropy` loss, also known as log loss. It is defined as:
# 
# $$
# \text{CrossEntropy} = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
# $$
# 
# Where:
# - $N$ is the number of samples.
# - $y_i$ is the true label of the $i^{th}$ sample.
# - $\hat{y}_i$ is the predicted probability of the $i^{th}$ sample being in the positive class.
# 
# > **Note**: We will not compute loss for `ignore_class` which indicates special tokens (`[CLS]`, `[SEP]`, `[PAD]`) or intermediate token of a word.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-16T07:40:43.305515Z","iopub.execute_input":"2024-02-16T07:40:43.305837Z","iopub.status.idle":"2024-02-16T07:40:43.313651Z","shell.execute_reply.started":"2024-02-16T07:40:43.305812Z","shell.execute_reply":"2024-02-16T07:40:43.31277Z"}}
class CrossEntropy(keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, ignore_class=-100, reduction=None, **args):
        super().__init__(reduction=reduction, **args)
        self.ignore_class = ignore_class

    def call(self, y_true, y_pred):
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1, CFG.num_labels])
        loss = super().call(y_true, y_pred)
        if self.ignore_class is not None:
            valid_mask = ops.not_equal(
                y_true, ops.cast(self.ignore_class, y_pred.dtype)
            )
            loss = ops.where(valid_mask, loss, 0.0)
            loss = ops.sum(loss)
            loss /= ops.maximum(ops.sum(ops.cast(valid_mask, loss.dtype)), 1)
        else:
            loss = ops.mean(loss)
        return loss


# %% [markdown]
# ## Metric: FBetaScore ($\beta = 5$)
# 
# The competition metric is $F^\beta$, which combines precision and recall, weighted by a parameter $\beta = 5$. It is defined as:
# 
# $$
# \text{FBetaScore} = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}
# $$
# 
# Where:
# - Precision $= \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$
# - Recall $= \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$
# - $\beta$ controls the weighting between precision and recall. As in this competition, $\beta = 5$, it means more weight is given to recall. In other words, **metric will penalize more, if a positive token is classified as negative**.
# 
# > **Note${}^1$**: The competition will use `micro` averaging for the `FBetaScore`, considering total counts across all classes, which is influenced by class imbalances. The `macro` averaging treats each class equally, regardless of frequency. Organizers may want models that perform well on predicting more common PII tags.
# 
# > **Note${}^2$**: We will not compute the metric for `ignore_classes`, which indicates special tokens (`[CLS]`, `[SEP]`, `[PAD]`) or non-start tokens of a word or `O` (Outer) labels.

# %% [code] {"execution":{"iopub.status.busy":"2024-02-16T07:40:43.314849Z","iopub.execute_input":"2024-02-16T07:40:43.315102Z","iopub.status.idle":"2024-02-16T07:40:43.331715Z","shell.execute_reply.started":"2024-02-16T07:40:43.31508Z","shell.execute_reply":"2024-02-16T07:40:43.330835Z"}}
class FBetaScore(keras.metrics.FBetaScore):
    def __init__(self, ignore_classes=[-100, 12], average="micro", beta=5.0,
                 name="f5_score", **args):
        super().__init__(beta=beta, average=average, name=name, **args)
        self.ignore_classes = ignore_classes or []

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = ops.convert_to_tensor(y_pred, dtype=self.dtype)
        
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1, CFG.num_labels])
            
        valid_mask = ops.ones_like(y_true, dtype=self.dtype)
        if self.ignore_classes:
            for ignore_class in self.ignore_classes:
                valid_mask &= ops.not_equal(y_true, ops.cast(ignore_class, y_pred.dtype))
        valid_mask = ops.expand_dims(valid_mask, axis=-1)
        
        y_true = ops.one_hot(y_true, CFG.num_labels)
        
        if not self._built:
            self._build(y_true.shape, y_pred.shape)

        threshold = ops.max(y_pred, axis=-1, keepdims=True)
        y_pred = ops.logical_and(
            y_pred >= threshold, ops.abs(y_pred) > 1e-9
        )

        y_pred = ops.cast(y_pred, dtype=self.dtype)
        y_true = ops.cast(y_true, dtype=self.dtype)
        
        tp = ops.sum(y_pred * y_true * valid_mask, self.axis)
        fp = ops.sum(y_pred * (1 - y_true) * valid_mask, self.axis)
        fn = ops.sum((1 - y_pred) * y_true * valid_mask, self.axis)
            
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

# %% [markdown]
# # 🤖 | Modeling
# 
# In this notebook, we will use the `DebertaV3` backbone from KerasNLP's pretrained models to extract features of tokens and employ `Dense` layers for token-level classification. Unlike Text Classification, transformer outputs are not pooled; instead, a `Dense` layer is applied to the outputs to obtain predictions.
# 
# To clarify, the output of the transformer model is a 3D tensor of shape $(batch\_size, seq\_len, feat\_dim)$, where only the $feat\_dim$ is changed, while the others remain the same. Subsequently, the `Dense` (or `Linear`) layer maps the `feat_dim` to `num_labels` and then applies a `softmax` activation to get the final prediction.
# 
# To explore other backbones, simply modify the `preset` in the `CFG` (config). A list of available pretrained backbones can be found on the [KerasNLP website](https://keras.io/api/keras_nlp/models/).
# 
# > **Note:** The output `dtype` of the final activation is manually set to `float32` to facilitate `mixed_precision`.
# 
# <u>Food for thought</u>: 
# 1. Some may wonder why the input to the `Dense` layer is 3D `(batch_size, d0, d1)` instead of the traditional 2D `(batch_size, d0)`. You can check [Hint: you can check this page](https://keras.io/api/layers/core_layers/dense/).
# 2. We are training our model with sequence of `1024` length, however we are doing inference with sequence of `2000` length. What is happening here?

# %% [code] {"execution":{"iopub.status.busy":"2024-02-16T07:40:43.332842Z","iopub.execute_input":"2024-02-16T07:40:43.333175Z","iopub.status.idle":"2024-02-16T07:40:54.593423Z","shell.execute_reply.started":"2024-02-16T07:40:43.333143Z","shell.execute_reply":"2024-02-16T07:40:54.592517Z"}}
# Build Token Classification model
backbone = keras_nlp.models.DebertaV3Backbone.from_preset(
    CFG.preset,
)
out = backbone.output
out = keras.layers.Dense(CFG.num_labels, name="logits")(out)
out = keras.layers.Activation("softmax", dtype="float32", name="prediction")(out)
model = keras.models.Model(backbone.input, out)

# Compile model for optimizer, loss and metric
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=CrossEntropy(),
    metrics=[FBetaScore()],
)

# Summary of the model architecture
model.summary()

# %% [markdown]
# # ⚓ | LR Schedule
# 
# A well-structured learning rate schedule is essential for efficient model training, ensuring optimal convergence and avoiding issues such as overshooting or stagnation.

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-02-16T07:40:54.594578Z","iopub.execute_input":"2024-02-16T07:40:54.594857Z","iopub.status.idle":"2024-02-16T07:40:54.605291Z","shell.execute_reply.started":"2024-02-16T07:40:54.594833Z","shell.execute_reply":"2024-02-16T07:40:54.60435Z"}}
import math

def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 6e-6, 2.5e-6 * batch_size, 1e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        fig = px.line(x=np.arange(epochs),
                      y=[lrfn(epoch) for epoch in np.arange(epochs)], 
                      title='LR Scheduler',
                      markers=True,
                      labels={'x': 'epoch', 'y': 'lr'})
        fig.update_layout(
            yaxis = dict(
                showexponent = 'all',
                exponentformat = 'e'
            )
        )
        fig.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-02-16T07:40:54.606471Z","iopub.execute_input":"2024-02-16T07:40:54.606765Z","iopub.status.idle":"2024-02-16T07:40:55.937966Z","shell.execute_reply.started":"2024-02-16T07:40:54.606742Z","shell.execute_reply":"2024-02-16T07:40:55.937016Z"}}
lr_cb = get_lr_callback(CFG.train_batch_size, mode=CFG.lr_mode, plot=True)

# %% [markdown]
# # 🚂 | Training

# %% [code] {"execution":{"iopub.status.busy":"2024-02-16T07:40:55.939256Z","iopub.execute_input":"2024-02-16T07:40:55.939628Z"}}
if CFG.train:
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=CFG.epochs,
        callbacks=[lr_cb],
        verbose=1,
    )
else:
    model.load_weights("/kaggle/input/pii-data-detection-ckpt-ds/model.weights.h5")

# %% [markdown]
# Even though this notebook does both training and inference, let's store the **weights** of the trained model on disk in case we may need it later. Also, if you need more time for inference, you can create separate notebooks for training and inference.
# 
# > **Note**: The filename of the weights should end in `.weights.h5`

# %% [code]
model.save_weights("model.weights.h5")

# %% [markdown]
# # 🔬 | Evaluation
# 
# We have trained and validated our model on a `1024` sequence length; however, we will be making inference using a `2000` sequence length. Thus, it is important to check how our model performs with `2000` sequence length inputs.

# %% [code]
# Build Validation dataloader with "infer_seq_len"
valid_ds = build_dataset(valid_words, valid_labels, return_ids=False, batch_size=CFG.infer_batch_size,
                        seq_len=CFG.infer_seq_len, shuffle=False, cache=False)

# %% [code]
# Evaluate
model.evaluate(valid_ds, return_dict=True, verbose=0)

# %% [markdown]
# # 🧪 | Prediction

# %% [markdown]
# ## Build Test Dataloader

# %% [code]
# Test data
test_data = json.load(open(f"{BASE_PATH}/test.json"))

# Ensure number of samples is divisble by number of devices
need_samples  = len(devices) - len(test_data) % len(devices)
for _ in range(need_samples):
    test_data.append(test_data[-1]) # repeat the last sample
    
# Initialize empty arrays
test_words = np.empty(len(test_data), dtype=object)
test_docs = np.empty(len(test_data), dtype=np.int32)

# Fill the arrays
for i, x in tqdm(enumerate(test_data), total=len(test_data)):
    test_words[i] = np.array(x["tokens"])
    test_docs[i] = x["document"]

# Get token ids
id_ds = build_dataset(test_words, return_ids=True, batch_size=len(test_words), 
                        seq_len=CFG.infer_seq_len, shuffle=False, cache=False, drop_remainder=False)
test_token_ids = ops.convert_to_numpy([ids for ids in iter(id_ds)][0])

# Build test dataloader
test_ds = build_dataset(test_words, return_ids=False, batch_size=CFG.infer_batch_size,
                        seq_len=CFG.infer_seq_len, shuffle=False, cache=False, drop_remainder=False)

# %% [markdown]
# ## Inference

# %% [code]
# Do inference
test_preds = model.predict(test_ds, verbose=1)

# Convert probabilities to class labels via max confidence
test_preds = np.argmax(test_preds, axis=-1)

# %% [markdown]
# ## Remove Extra Samples
# 
# We need to remove the extra samples we added to the test data to ensure the number of samples is divisible by the number of devices.

# %% [code]
test_docs = test_docs[:-need_samples]
test_token_ids = test_token_ids[:-need_samples]
test_preds = test_preds[:-need_samples]
test_words = test_words[:-need_samples]

# %% [markdown]
# ## 🧹 | Post-Processing
# 
# The following code processes the prediction to filter out unwanted parts. Specifically, it does the following:
# 
# 1. It filters out any tokens of a word that are not at the start (refer to the [🥣 | Data Processing](https://www.kaggle.com/code/awsaf49/pii-data-detection-kerasnlp-starter-notebook#%F0%9F%A5%A3-%7C-Data-Processing) section for more details).
# 2. It removes samples labeled as `O` (BIO format), as the submission file requires only non-`O` samples.
# 3. It ignores predictions for special tokens like `[CLS]`, `[SEP]`, and `[PAD]`.
# 
# > **Note**: A unique feature of following post-processing is that it uses numpy vectorized operations to filter out predictions, making it very fast and efficient.

# %% [code]
document_list = []
token_id_list = []
label_id_list = []
token_list = []

for doc, token_ids, preds, tokens in tqdm(
    zip(test_docs, test_token_ids, test_preds, test_words), total=len(test_words)
):
    # Create mask for filtering
    mask1 = np.concatenate(([True], token_ids[1:] != token_ids[:-1])) # ignore non-start tokens of a word
    mask2 = (preds != 12) # ignore `O` (BIO format) label -> 12 (integer format) label
    mask3 = (token_ids != -1)  # ignore [CLS], [SEP], and [PAD] tokens
    mask = (mask1 & mask2 & mask3) # merge filters
    
    # Apply filter
    token_ids = token_ids[mask]
    preds = preds[mask]

     # Store prediction if number of tokens is not zero
    if len(token_ids):
        token_list.extend(tokens[token_ids])
        document_list.extend([doc] * len(token_ids))
        token_id_list.extend(token_ids)
        label_id_list.extend(preds)

# %% [markdown]
# # 📩 | Submission
# 
# Let's build a dataframe from the predictions which will help us visually check if our model is predicting correctly or not. We also have to map **integer** labels to **string** BIO format labels.

# %% [code]
pred_df = pd.DataFrame(
    {
        "document": document_list,
        "token": token_id_list,
        "label_id": label_id_list,
        "token_string": token_list,
    }
)
pred_df = pred_df.rename_axis("row_id").reset_index() # add `row_id` column
pred_df["label"] = pred_df.label_id.map(CFG.id2label) # map integer label to BIO format label
pred_df.head(10)

# %% [markdown]
# In `submission.csv` we are excluding `token_string` and `label_id` from the columns as they are not part of submission file.

# %% [code]
sub_df = pred_df.drop(columns=["token_string", "label_id"]) # remove extra columns
sub_df.to_csv("submission.csv", index=False)

# %% [markdown]
# # 🔗 | Reference
# * [Detect Fake Text: KerasNLP [TF/Torch/JAX][Train]](https://www.kaggle.com/code/awsaf49/detect-fake-text-kerasnlp-tf-torch-jax-train)
# * [Token classification](https://huggingface.co/docs/transformers/en/tasks/token_classification)
# * [transformer ner baseline [lb 0.854]](https://www.kaggle.com/code/nbroad/transformer-ner-baseline-lb-0-854)
