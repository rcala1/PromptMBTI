import os
import torch
import sys
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    set_seed,
)
from torch.nn import DataParallel
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dataset import TRAITS
from dataset import prepare_classic_mbti_splits

CURR_TRAIT = 0
FEW = False

PATH_DATASET = (
    "/home/rcala/PromptMBTI_Masters/filtered/bert_filtered_"
    + TRAITS[CURR_TRAIT]
    + "_discrete"
    + ".csv"
)
if not FEW:
    GPT2_LOAD_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/" + "gpt2" + "_" + TRAITS[CURR_TRAIT] + "_classic"
    )
else:
    GPT2_LOAD_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/" + "gpt2" + "_" + TRAITS[CURR_TRAIT] + "_classic_few"
    )

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

torch.manual_seed(123)
set_seed(123)
np.random.seed(123)
random.seed(123)

model_config = GPT2Config.from_pretrained(
    pretrained_model_name_or_path=GPT2_LOAD_PATH, num_labels=2
)
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=GPT2_LOAD_PATH)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = GPT2ForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=GPT2_LOAD_PATH, config=model_config
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.to(dev)

batch_size = 16

train_loader, val_loader, test_loader = prepare_classic_mbti_splits(
    PATH_DATASET, batch_size, tokenizer, FEW
)

model.eval()
all_pred = []
all_true = []

tqdm_test = tqdm(test_loader)

with torch.no_grad():
    for texts, inputs in tqdm_test:

        inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

        loss, prediction = model(**inputs)[:2]

        loss = loss.mean()
        all_pred += list(prediction.cpu().detach().numpy().argmax(axis=1))
        all_true += list(inputs["labels"].cpu().detach().numpy())

        tqdm_test.set_description("Test batch_loss: {}".format(loss.item()))

test_acc = accuracy_score(all_true, all_pred)
test_f1 = f1_score(all_true, all_pred, average="macro")

print(f"Test_acc: {test_acc:.4f} Test_f1: {test_f1:.4f}")
