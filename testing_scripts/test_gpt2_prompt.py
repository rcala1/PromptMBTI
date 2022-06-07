import os
import torch
import sys
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    set_seed,
)
from torch.nn import DataParallel
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from dataset import TRAITS
from dataset import prepare_prompt_mbti_splits
from statistics import get_prompt_true_pred

CURR_TRAIT = 0
FEW = True

PATH_DATASET = (
    "/home/rcala/PromptMBTI_Masters/filtered/bert_filtered_"
    + TRAITS[CURR_TRAIT]
    + "_prompt"
    + ".csv"
)
if not FEW:
    GPT2_LOAD_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/"
        + "gpt2"
        + "_"
        + TRAITS[CURR_TRAIT]
        + "_prompt"
    )
else:
    GPT2_LOAD_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/"
        + "gpt2"
        + "_"
        + TRAITS[CURR_TRAIT]
        + "_prompt_few"
    )

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

random_seed = 1

torch.manual_seed(random_seed)
set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

tokenizer = GPT2Tokenizer.from_pretrained(
    pretrained_model_name_or_path=GPT2_LOAD_PATH, do_lower_case=True
)
tokenizer.do_lower_case = True
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=GPT2_LOAD_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(
    pretrained_model_name_or_path=GPT2_LOAD_PATH, config=model_config
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.to(dev)

train_batch_size = 2
test_batch_size = 1

train_loader, val_loader, test_loader = prepare_prompt_mbti_splits(
    PATH_DATASET, train_batch_size, test_batch_size, tokenizer, "gpt2", CURR_TRAIT, FEW
)

model.eval()
all_pred = []
all_true = []

tqdm_test = tqdm(test_loader)

with torch.no_grad():

    for idx, (empty_prompts, labels, prompts, inputs) in enumerate(tqdm_test):

        inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

        loss, prediction = model(**inputs)[:2]

        loss = loss.mean()

        y_true, y_pred, unknown_present = get_prompt_true_pred(
            model,
            tokenizer,
            dev,
            empty_prompts,
            labels,
            prompts,
            CURR_TRAIT,
            "gpt2",
            is_training=False,
        )

        if unknown_present:
            print(f"Unknown found at {idx}")

        all_pred += y_pred
        all_true += y_true

test_acc = accuracy_score(all_true, all_pred)
test_f1 = f1_score(all_true, all_pred, average="macro")

print(f"Test_acc: {test_acc:.4f} Test_f1: {test_f1:.4f}")
print()
