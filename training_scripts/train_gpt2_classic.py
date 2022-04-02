import os
import torch
import sys
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AdamW,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.nn import DataParallel
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dataset import TRAITS
from dataset import prepare_classic_mbti_splits
from pytorchtools import EarlyStopping

CURR_TRAIT = 0
FEW = False

PATH_DATASET = (
    "/home/rcala/PsromptMBTI_Masters/filtered/bert_filtered_"
    + TRAITS[CURR_TRAIT]
    + "_discrete"
    + ".csv"
)
GPT2_MODEL_PATH = "gpt2"

if not FEW:
    GPT2_SAVE_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/" + "gpt2" + "_" + TRAITS[CURR_TRAIT] + "_classic_1e-5"
    )
else:
    GPT2_SAVE_PATH = (
        "/home/rcala/PromptMBTI_Masters/models/" + "gpt2" + "_" + TRAITS[CURR_TRAIT] + "_classic_few"
    )

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    pretrained_model_name_or_path=GPT2_MODEL_PATH, num_labels=2
)
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=GPT2_MODEL_PATH)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = GPT2ForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=GPT2_MODEL_PATH, config=model_config
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.to(dev)

earlystopping = EarlyStopping(patience=2, path=GPT2_SAVE_PATH)
optimizer = AdamW(model.parameters(), lr=1e-5)
epochs = 6
batch_size = 2

train_loader, val_loader, test_loader = prepare_classic_mbti_splits(
    PATH_DATASET, batch_size, tokenizer, FEW
)

total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps,
)

for epoch in range(epochs):

    model.train()
    all_pred = []
    all_true = []

    tqdm_train = tqdm(train_loader)

    for texts, inputs in tqdm_train:

        optimizer.zero_grad()
        model.zero_grad()

        inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

        loss, prediction = model(**inputs)[:2]

        loss = loss.mean()
        loss.backward()
        optimizer.step()

        all_pred += list(prediction.cpu().detach().numpy().argmax(axis=1))
        all_true += list(inputs["labels"].cpu().detach().numpy())
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        tqdm_train.set_description(
            "Epoch {}, Train batch_loss: {}".format(
                epoch + 1,
                loss.item(),
            )
        )
        
        scheduler.step()

    train_acc = accuracy_score(all_true, all_pred)
    train_f1 = f1_score(all_true, all_pred, average="macro")

    model.eval()
    all_pred = []
    all_true = []

    tqdm_val = tqdm(val_loader)

    with torch.no_grad():
        for texts, inputs in tqdm_val:

            inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

            loss, prediction = model(**inputs)[:2]

            loss = loss.mean()

            all_pred += list(prediction.cpu().detach().numpy().argmax(axis=1))
            all_true += list(inputs["labels"].cpu().detach().numpy())

            tqdm_val.set_description(
                "Epoch {}, Val batch_loss: {}".format(epoch + 1, loss.item())
            )

    val_acc = accuracy_score(all_true, all_pred)
    val_f1 = f1_score(all_true, all_pred, average="macro")

    print(f"Epoch {epoch+1}")
    print(f"Train_acc: {train_acc:.4f} Train_f1: {train_f1:.4f}")
    print(f"Val_acc: {val_acc:.4f} Val_f1: {val_f1:.4f}")
    earlystopping(-val_acc, model, tokenizer)
    print()
    if earlystopping.early_stop == True:
        break
