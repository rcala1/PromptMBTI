import os
import torch
import sys
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    set_seed,
)
from torch.nn import DataParallel
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from dataset import TRAITS
from dataset import prepare_prompt_mbti_splits
from statistics import get_prompt_true_pred

CURR_TRAIT = 3
FEW = True

PATH_DATASET = (
    "/home/rcala/PromptMBTI_Masters/filtered/bert_filtered_"
    + TRAITS[CURR_TRAIT]
    + "_prompt"
    + ".csv"
)
BERT_MODEL_PATH = "bert-base-uncased"

if not FEW:
    BERT_LOAD_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/"
        + "bert"
        + "_"
        + TRAITS[CURR_TRAIT]
        + "_prompt"
    )
else:
    BERT_LOAD_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/"
        + "bert"
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

model = BertForMaskedLM.from_pretrained(BERT_LOAD_PATH)
tokenizer = BertTokenizer.from_pretrained(BERT_LOAD_PATH, do_lower_case=True)
model.to(dev)

train_batch_size = 2
test_batch_size = 1

train_loader, val_loader, test_loader = prepare_prompt_mbti_splits(
    PATH_DATASET, train_batch_size, test_batch_size, tokenizer, "bert", CURR_TRAIT, FEW
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
            "bert",
            is_training=False,
        )

        if unknown_present:
            print(f"Unknown found at {idx}")

        all_pred += y_pred
        all_true += y_true

        tqdm_test.set_description("Test batch_loss: {}".format(loss.item()))

test_acc = accuracy_score(all_true, all_pred)
test_f1 = f1_score(all_true, all_pred, average="macro")

print(f"Test_acc: {test_acc:.4f} Test_f1: {test_f1:.4f}")
