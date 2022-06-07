import os
import torch
import sys
import numpy as np
import random
from random import randrange
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AdamW,
    BertForMaskedLM,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.nn import DataParallel
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from dataset import TRAITS
from dataset import prepare_prompt_mbti_splits
from dataset import CONT_PROMPT_LENGTH
from pytorchtools import EarlyStopping
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
    BERT_SAVE_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/"
        + "bert"
        + "_"
        + TRAITS[CURR_TRAIT]
        + "_prompt"
    )
else:
    BERT_SAVE_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/"
        + "bert"
        + "_"
        + TRAITS[CURR_TRAIT]
        + "_prompt_few"
    )

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

model = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, do_lower_case=True)
model.to(dev)

org_tokens_len = len(tokenizer)

special_tokens_list = [f"[CONT_{i}]" for i in range(CONT_PROMPT_LENGTH, 0, -1)]
special_tokens_dict = {"additional_special_tokens": special_tokens_list}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

sampled_idxs = []
random.seed(1)
for i in range(1, CONT_PROMPT_LENGTH + 1):
    sampled_org_tok_idx = randrange(org_tokens_len)
    sampled_idxs += [sampled_org_tok_idx]
    sampled_org_tok_embd = model.bert.embeddings.word_embeddings.weight[
        sampled_org_tok_idx, :
    ]
    model.bert.embeddings.word_embeddings.weight.data[-i, :] = sampled_org_tok_embd
random.seed(random_seed)

earlystopping = EarlyStopping(patience=5, path=BERT_SAVE_PATH)
optimizer = AdamW([model.bert.embeddings.word_embeddings.weight], lr=7.5e-5)
epochs = 100
train_batch_size = 2
test_batch_size = 1

train_loader, val_loader, test_loader = prepare_prompt_mbti_splits(
    PATH_DATASET, train_batch_size, test_batch_size, tokenizer, "bert", CURR_TRAIT, FEW
)

total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps,
)

for epoch in range(epochs):

    model.train()

    tqdm_train = tqdm(train_loader)

    for empty_prompts, labels, prompts, inputs in tqdm_train:

        optimizer.zero_grad()
        model.zero_grad()

        inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

        loss, prediction = model(**inputs)[:2]

        loss = loss.mean()
        loss.backward()

        model.bert.embeddings.word_embeddings.weight.grad[:-CONT_PROMPT_LENGTH, :] = 0

        optimizer.step()
        scheduler.step()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        tqdm_train.set_description(
            "Epoch {}, Train batch_loss: {}".format(epoch + 1, loss.item(),)
        )

    model.eval()
    all_pred = []
    all_true = []

    tqdm_val = tqdm(val_loader)

    with torch.no_grad():

        for empty_prompts, labels, prompts, inputs in tqdm_val:

            inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

            loss, prediction = model(**inputs)[:2]

            loss = loss.mean()

            y_true, y_pred, _ = get_prompt_true_pred(
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
            all_pred += y_pred
            all_true += y_true

            tqdm_val.set_description(
                "Epoch {}, Val batch_loss: {}".format(epoch + 1, loss.item())
            )

    val_acc = accuracy_score(all_true, all_pred)
    val_f1 = f1_score(all_true, all_pred, average="macro")

    print(f"Epoch {epoch+1}")
    print(f"Val_acc: {val_acc:.4f} Val_f1: {val_f1:.4f}")
    earlystopping(-val_f1, model, tokenizer)
    print()
    if earlystopping.early_stop == True:
        break
