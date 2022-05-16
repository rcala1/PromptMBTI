import torch
import pandas as pd
from torch.utils.data import DataLoader
from collections import Counter

TRAITS = ["introverted", "intuitive", "thinking", "perceiving"]
OPPOSITE_TRAITS = ["extroverted", "sensing", "feeling", "judging"]

FEW_TRAIN_NUM_EXAMPLES = 48
FEW_VAL_NUM_EXAMPLES = 16

CONT_PROMPT_LENGTH = 30

true2prompt = {
    "introverted": "introverted",
    "extroverted": "extroverted",
    "intuitive": "intuitive",
    "sensing": "sensing",
    "thinking": "thinking",
    "feeling": "feeling",
    "perceiving": "perceiving",
    "judging": "judging",
}

prompt2true = {
    "introverted": "introverted",
    "extroverted": "extroverted",
    "intuitive": "intuitive",
    "sensing": "sensing",
    "thinking": "thinking",
    "feeling": "feeling",
    "perceiving": "perceiving",
    "judging": "judging",
}

PREFIX_PROMPT = ""
MID_PROMPT = ""


class MBTITraitDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        super().__init__()
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        item = {"text": self.texts[idx], "label": self.labels[idx]}

        return item


def collate_classic_mbti(samples, tokenizer):

    texts = [sample["text"] for sample in samples]
    labels = [sample["label"] for sample in samples]

    inputs = tokenizer(
        text=texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
    )

    inputs.update({"labels": torch.tensor(labels)})

    return texts, inputs


def prepare_filter_splits(path, idx_split, trait, batch_size, tokenizer):

    all_comments_with_mbti = pd.read_csv(path, usecols=["text", TRAITS[trait]])

    all_comments_with_mbti = all_comments_with_mbti.sample(frac=1)

    texts = list(all_comments_with_mbti["text"])

    labels = list(all_comments_with_mbti[TRAITS[trait]])

    collate_fun = lambda samples: collate_classic_mbti(samples, tokenizer=tokenizer)

    if idx_split == 1:

        training_texts = texts[: int(2 / 3 * len(texts))]
        test_texts = texts[int(2 / 3 * len(texts)) :]
        train_texts = training_texts[: int(0.8 * len(training_texts))]
        val_texts = training_texts[int(0.8 * len(training_texts)) :]

        training_labels = labels[: int(2 / 3 * len(labels))]
        test_labels = labels[int(2 / 3 * len(labels)) :]
        train_labels = training_labels[: int(0.8 * len(training_labels))]
        val_labels = training_labels[int(0.8 * len(training_labels)) :]

    if idx_split == 2:

        training_texts = (
            texts[: int(1 / 3 * len(texts))] + texts[int(2 / 3 * len(texts)) :]
        )
        test_texts = texts[int(1 / 3 * len(texts)) : int(2 / 3 * len(texts))]
        train_texts = training_texts[: int(0.8 * len(training_texts))]
        val_texts = training_texts[int(0.8 * len(training_texts)) :]

        training_labels = (
            labels[: int(1 / 3 * len(labels))] + labels[int(2 / 3 * len(labels)) :]
        )
        test_labels = labels[int(1 / 3 * len(labels)) : int(2 / 3 * len(labels))]
        train_labels = training_labels[: int(0.8 * len(training_labels))]
        val_labels = training_labels[int(0.8 * len(training_labels)) :]

    if idx_split == 3:

        training_texts = texts[int(1 / 3 * len(texts)) :]
        test_texts = texts[: int(1 / 3 * len(texts))]
        train_texts = training_texts[: int(0.8 * len(training_texts))]
        val_texts = training_texts[int(0.8 * len(training_texts)) :]

        training_labels = labels[int(1 / 3 * len(labels)) :]
        test_labels = labels[: int(1 / 3 * len(labels))]
        train_labels = training_labels[: int(0.8 * len(training_labels))]
        val_labels = training_labels[int(0.8 * len(training_labels)) :]

    train_dataset = MBTITraitDataset(train_texts, train_labels)
    val_dataset = MBTITraitDataset(val_texts, val_labels)
    test_dataset = MBTITraitDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate_fun,)
    val_loader = DataLoader(val_dataset, batch_size, collate_fn=collate_fun,)
    test_loader = DataLoader(test_dataset, batch_size, collate_fn=collate_fun,)

    return train_loader, val_loader, test_loader


def prepare_classic_mbti_splits(path, batch_size, tokenizer, few=False):

    filtered_texts_with_mbti = pd.read_csv(path)

    filtered_texts_with_mbti = filtered_texts_with_mbti.sample(frac=1)

    texts = list(filtered_texts_with_mbti["text"])
    trait_column = filtered_texts_with_mbti.columns[1]
    labels = list(filtered_texts_with_mbti[trait_column])

    collate_fun = lambda samples: collate_classic_mbti(samples, tokenizer=tokenizer)

    if not few:

        train_texts = texts[: int(0.7 * len(texts))]
        val_texts = texts[int(0.7 * len(texts)) : int(0.8 * len(texts))]
        test_texts = texts[int(0.8 * len(texts)) :]

        train_labels = labels[: int(0.7 * len(labels))]
        val_labels = labels[int(0.7 * len(labels)) : int(0.8 * len(labels))]
        test_labels = labels[int(0.8 * len(labels)) :]

    else:

        train_texts = texts[:FEW_TRAIN_NUM_EXAMPLES]
        val_texts = texts[
            FEW_TRAIN_NUM_EXAMPLES : FEW_TRAIN_NUM_EXAMPLES + FEW_VAL_NUM_EXAMPLES
        ]
        test_texts = texts[int(0.8 * len(texts)) :]

        train_labels = labels[:FEW_TRAIN_NUM_EXAMPLES]
        val_labels = labels[
            FEW_TRAIN_NUM_EXAMPLES : FEW_TRAIN_NUM_EXAMPLES + FEW_VAL_NUM_EXAMPLES
        ]
        test_labels = labels[int(0.8 * len(labels)) :]

    train_dataset = MBTITraitDataset(train_texts, train_labels)
    val_dataset = MBTITraitDataset(val_texts, val_labels)
    test_dataset = MBTITraitDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate_fun,)
    val_loader = DataLoader(val_dataset, batch_size, collate_fn=collate_fun,)
    test_loader = DataLoader(test_dataset, batch_size, collate_fn=collate_fun,)

    return train_loader, val_loader, test_loader


class MBTIPromptTraitDataset(torch.utils.data.Dataset):
    def __init__(self, empty_prompts, labels):
        super().__init__()
        self.empty_prompts = empty_prompts
        self.labels = labels

    def __len__(self):
        return len(self.empty_prompts)

    def __getitem__(self, idx):

        item = {"empty_prompt": self.empty_prompts[idx], "label": self.labels[idx]}

        return item


def collate_gpt2_prompt_mbti(samples, tokenizer):

    empty_prompts = [sample["empty_prompt"] for sample in samples]
    labels = [true2prompt[sample["label"]] for sample in samples]
    prompts = [value[0] + value[1] for value in zip(empty_prompts, labels)]

    prompts_inputs = tokenizer(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    labels_inputs = torch.clone(prompts_inputs["input_ids"])

    for idx, empty_prompt in enumerate(empty_prompts):
        empty_prompt_inputs = tokenizer(
            text=empty_prompt, return_tensors="pt", truncation=True, max_length=512
        )
        labels_inputs[idx, : empty_prompt_inputs["input_ids"].shape[-1] - 1] = -100
        prompt_eos_idxs = (
            prompts_inputs["input_ids"][idx, :] == tokenizer.eos_token_id
        ).nonzero(as_tuple=True)[0]
        if len(prompt_eos_idxs):
            labels_inputs[idx, prompt_eos_idxs[0] + 1 :] = -100

    prompts_inputs.update({"labels": labels_inputs})

    return empty_prompts, labels, prompts, prompts_inputs


def collate_bert_prompt_mbti(samples, tokenizer, trait):

    empty_prompts = [sample["empty_prompt"] for sample in samples]
    labels = [true2prompt[sample["label"]] for sample in samples]

    prompt_labels_dict = {}
    prompt_labels_dict[true2prompt[TRAITS[trait]]] = len(
        tokenizer.encode(true2prompt[TRAITS[trait]], add_special_tokens=False)
    )
    prompt_labels_dict[true2prompt[OPPOSITE_TRAITS[trait]]] = len(
        tokenizer.encode(true2prompt[OPPOSITE_TRAITS[trait]], add_special_tokens=False)
    )

    prompts = []
    for value in zip(empty_prompts, labels):
        prompt = value[0] + value[1]
        unk_add = max(prompt_labels_dict.values()) - prompt_labels_dict[value[1]]
        for _ in range(unk_add):
            prompt += " [UNK]"
        prompts += [prompt]

    prompts_inputs = tokenizer(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    labels_inputs = torch.clone(prompts_inputs["input_ids"])

    for idx, empty_prompt in enumerate(empty_prompts):
        empty_prompt_inputs = tokenizer(
            text=empty_prompt, return_tensors="pt", truncation=True, max_length=512
        )
        prompt_sep_idx = (
            prompts_inputs["input_ids"][idx, :] == tokenizer.sep_token_id
        ).nonzero(as_tuple=True)[0]
        labels_inputs[idx, : empty_prompt_inputs["input_ids"].shape[-1] - 1] = -100
        labels_inputs[idx, prompt_sep_idx + 1 :] = -100
        prompts_inputs["input_ids"][
            idx, empty_prompt_inputs["input_ids"].shape[-1] - 1 : prompt_sep_idx
        ] = tokenizer.mask_token_id

    prompts_inputs.update({"labels": labels_inputs})

    return empty_prompts, labels, prompts, prompts_inputs


def prepare_prompt_mbti_splits(
    path, train_batch_size, test_batch_size, tokenizer, model_name, trait, few=False
):

    filtered_texts_with_mbti = pd.read_csv(path)

    filtered_texts_with_mbti = filtered_texts_with_mbti.sample(frac=1)

    texts = list(filtered_texts_with_mbti["text"])
    trait_column = filtered_texts_with_mbti.columns[1]
    labels = list(filtered_texts_with_mbti[trait_column])

    # Text : [text] MBTI : [trait_label]
    # empty_prompts = [
    #    PREFIX_PROMPT + " " + text + " " + MID_PROMPT + " " for text in texts
    # ]

    # [text] Based on the previous text , the person is of MBTI personality trait [trait_label]
    # [text] Question : What is the MBTI personality trait of the person in the previous text ? Answer: [trait_label]
    # empty_prompts = [text + " " + MID_PROMPT + " " for text in texts]

    # [text] [trait_label]
    empty_prompts = [text + " " for text in texts]

    # [text] [two_personalities] ? [trait_label]
    # empty_prompts = [
    #    text
    #    + " "
    #    + f"{true2prompt[TRAITS[trait]]} or {true2prompt[OPPOSITE_TRAITS[trait]]}?"
    #    + " "
    #    for text in texts
    # ]

    # continuous prompt
    # empty_prompts = []
    # for text in texts:
    #    for i in range(CONT_PROMPT_LENGTH, 0, -1):
    #        text = f"[CONT_{i}] " + text
    #    empty_prompts += [text]

    if model_name == "bert":
        collate_fun = lambda samples: collate_bert_prompt_mbti(
            samples, tokenizer, trait
        )
    else:
        collate_fun = lambda samples: collate_gpt2_prompt_mbti(samples, tokenizer)

    if not few:

        train_empty_prompts = empty_prompts[: int(0.7 * len(empty_prompts))]
        val_empty_prompts = empty_prompts[
            int(0.7 * len(empty_prompts)) : int(0.8 * len(empty_prompts))
        ]
        test_empty_prompts = empty_prompts[int(0.8 * len(empty_prompts)) :]

        train_labels = labels[: int(0.7 * len(labels))]
        val_labels = labels[int(0.7 * len(labels)) : int(0.8 * len(labels))]
        test_labels = labels[int(0.8 * len(labels)) :]

    else:

        train_empty_prompts = empty_prompts[:FEW_TRAIN_NUM_EXAMPLES]
        val_empty_prompts = empty_prompts[
            FEW_TRAIN_NUM_EXAMPLES : FEW_TRAIN_NUM_EXAMPLES + FEW_VAL_NUM_EXAMPLES
        ]
        test_empty_prompts = empty_prompts[int(0.8 * len(empty_prompts)) :]

        train_labels = labels[:FEW_TRAIN_NUM_EXAMPLES]
        val_labels = labels[
            FEW_TRAIN_NUM_EXAMPLES : FEW_TRAIN_NUM_EXAMPLES + FEW_VAL_NUM_EXAMPLES
        ]

        test_labels = labels[int(0.8 * len(labels)) :]

    train_dataset = MBTIPromptTraitDataset(train_empty_prompts, train_labels)
    val_dataset = MBTIPromptTraitDataset(val_empty_prompts, val_labels)
    test_dataset = MBTIPromptTraitDataset(test_empty_prompts, test_labels)

    train_loader = DataLoader(train_dataset, train_batch_size, collate_fn=collate_fun,)
    val_loader = DataLoader(val_dataset, test_batch_size, collate_fn=collate_fun,)
    test_loader = DataLoader(test_dataset, test_batch_size, collate_fn=collate_fun,)

    return train_loader, val_loader, test_loader

