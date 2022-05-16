import os
import torch
import numpy as np
import random
import sys
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AdamW,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.nn import DataParallel
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from dataset import TRAITS
from dataset import prepare_prompt_mbti_splits
from pytorchtools import EarlyStopping
from statistics import get_prompt_true_pred

CURR_TRAIT = 3
FEW = False

PATH_DATASET = (
    "/home/rcala/PromptMBTI_Masters/filtered/bert_filtered_"
    + TRAITS[CURR_TRAIT]
    + "_prompt"
    + ".csv"
)
GPT_MODEL_PATH = "gpt2"
if not FEW:
    GPT2_SAVE_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/"
        + "gpt2"
        + "_"
        + TRAITS[CURR_TRAIT]
        + "_prompt"
    )
else:
    GPT2_SAVE_PATH = (
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

random_seed = 123

torch.manual_seed(random_seed)
set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

tokenizer = GPT2Tokenizer.from_pretrained(
    pretrained_model_name_or_path=GPT_MODEL_PATH, do_lower_case=True
)
tokenizer.do_lower_case = True
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=GPT_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(
    pretrained_model_name_or_path=GPT_MODEL_PATH, config=model_config
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.to(dev)

optimizer = AdamW(model.parameters(), lr=2e-5)
earlystopping = EarlyStopping(patience=2, path=GPT2_SAVE_PATH)
epochs = 6
train_batch_size = 2
test_batch_size = 1

train_loader, val_loader, test_loader = prepare_prompt_mbti_splits(
    PATH_DATASET, train_batch_size, test_batch_size, tokenizer, "gpt2", CURR_TRAIT, FEW
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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

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
                "gpt2",
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
