import torch
import random
import math
from dataset import MID_PROMPT
from dataset import prompt2true, true2prompt, TRAITS, OPPOSITE_TRAITS


traits2class_dicts = [
    {"extroverted": 0, "introverted": 1},
    {"sensing": 0, "intuitive": 1},
    {"feeling": 0, "thinking": 1},
    {"judging": 0, "perceiving": 1},
]


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def get_prompt_true_pred(
    model, tokenizer, dev, prompts, trait, model_name, is_training=False
):
    if model_name == "gpt2":
        y_true, y_pred, unknown_present = get_labels_prompts_gpt2(
            model, tokenizer, dev, prompts, trait, is_training
        )
    else:
        y_true, y_pred, unknown_present = get_labels_prompts_bert(
            model, tokenizer, dev, prompts, trait, is_training
        )

    return y_true, y_pred, unknown_present


def get_labels_prompts_gpt2(model, tokenizer, dev, prompts, trait, is_training=False):

    if is_training:
        model.eval()

    true_labels = [
        prompt[prompt.find(MID_PROMPT) + len(MID_PROMPT) :].split()[0]
        for prompt in prompts
    ]

    if true2prompt.get(true_labels[0]) not in traits2class_dicts[trait]:
        return [], [], True

    prompts_without_labels = [
        rreplace(item[0], item[1], "", 1) for item in zip(prompts, true_labels)
    ]

    prompts_without_labels[0] = prompts_without_labels[0][:-1]

    inputs = tokenizer(
        text=prompts_without_labels,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    inputs = {k: v.type(torch.long).to(dev) for k, v in inputs.items()}

    trait_indices = tokenizer.encode(
        " " + true2prompt[TRAITS[trait]], add_special_tokens=False
    )
    opposite_trait_indices = tokenizer.encode(
        " " + true2prompt[OPPOSITE_TRAITS[trait]], add_special_tokens=False
    )

    if len(trait_indices) > len(opposite_trait_indices):
        diff_len = len(trait_indices) - len(opposite_trait_indices)
        for _ in range(diff_len):
            opposite_trait_indices += [tokenizer.eos_token_id]

    if len(opposite_trait_indices) > len(trait_indices):
        diff_len = len(opposite_trait_indices) - len(trait_indices)
        for _ in range(diff_len):
            trait_indices += [tokenizer.eos_token_id]

    new_generate_len = len(trait_indices)

    added_token_id = 50256

    for idx in range(new_generate_len):

        tokens_logits = model(**inputs).logits
        token_logits = tokens_logits[0, [-1], :]

        decoding_indices = []
        if idx == 0:
            decoding_indices = [trait_indices[idx]] + [opposite_trait_indices[idx]]
        if idx > 0:
            if added_token_id == trait_indices[idx - 1]:
                decoding_indices += [trait_indices[idx]]
            if added_token_id == opposite_trait_indices[idx - 1]:
                decoding_indices += [opposite_trait_indices[idx]]

        if len(decoding_indices) == 0:
            return [], [], True

        decoding_indices = torch.tensor(decoding_indices)
        decoding_indices_oh = torch.nn.functional.one_hot(
            decoding_indices, tokenizer.vocab_size
        )

        mask_unimportant = torch.sum(decoding_indices_oh, 0) == 0
        mask_unimportant = mask_unimportant.repeat(token_logits.shape[0], 1)
        token_logits[mask_unimportant] = -math.inf
        added_token_id = torch.topk(token_logits, 1, dim=1).indices.tolist()[0][0]
        inputs["input_ids"] = torch.cat(
            (inputs["input_ids"], torch.tensor([[added_token_id]]).to(dev)), dim=1
        )
        inputs["attention_mask"] = torch.cat(
            (inputs["attention_mask"], torch.tensor([[1]]).to(dev)), dim=1
        )

    pred_labels = []
    real_output = []

    predicted_prompt = tokenizer.decode(
        inputs["input_ids"][0], skip_special_tokens=True
    )
    real_output += [predicted_prompt]
    splitted = predicted_prompt[
        predicted_prompt.find(MID_PROMPT.lower()) + len(MID_PROMPT.lower()) :
    ].split()
    if len(splitted) > 0:
        pred_label = splitted[0].lower()
    else:
        pred_label = "unknown"
        return [], [], True

    pred_labels += [pred_label]

    true_labels_discrete = []
    pred_labels_discrete = []

    for pred_label, true_label in zip(pred_labels, true_labels):

        pred_trait = prompt2true.get(pred_label)
        true_trait = prompt2true.get(true_label)

        if true_trait and traits2class_dicts[trait].get(true_trait) is not None:
            true_labels_discrete += [traits2class_dicts[trait][true_trait]]
            if pred_trait and traits2class_dicts[trait].get(pred_trait) is not None:
                pred_labels_discrete += [traits2class_dicts[trait].get(pred_trait)]
            else:
                return [], [], True

    if is_training:
        model.train()

    return true_labels_discrete, pred_labels_discrete, False


def get_labels_prompts_bert(model, tokenizer, dev, prompts, trait, is_training=False):

    if is_training:
        model.eval()

    true_labels = [
        prompt[prompt.find(MID_PROMPT) + len(MID_PROMPT) :].split()[0]
        for prompt in prompts
    ]

    if true2prompt.get(true_labels[0]) not in traits2class_dicts[trait]:
        return [], [], True

    prompts_without_labels = [
        rreplace(item[0], item[1], "", 1) for item in zip(prompts, true_labels)
    ]

    prompts_without_labels_wo_unk = []
    for prompt in prompts_without_labels:
        while "[UNK]" in prompt:
            prompt = rreplace(prompt, "[UNK]", "", 1)
        prompts_without_labels_wo_unk += [prompt]
    prompts_without_labels = prompts_without_labels_wo_unk

    masked_inputs = tokenizer(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    for idx, empty_prompt in enumerate(prompts_without_labels):
        empty_prompt_inputs = tokenizer(
            text=empty_prompt, return_tensors="pt", truncation=True, max_length=512
        )
        prompt_sep_idx = (
            masked_inputs["input_ids"][idx, :] == tokenizer.sep_token_id
        ).nonzero(as_tuple=True)[0]
        masked_inputs["input_ids"][
            idx, empty_prompt_inputs["input_ids"].shape[-1] - 1 : prompt_sep_idx
        ] = tokenizer.mask_token_id

    masked_inputs = {k: v.type(torch.long).to(dev) for k, v in masked_inputs.items()}

    mask_token_indexes = torch.where(
        masked_inputs["input_ids"] == tokenizer.mask_token_id
    )[1]

    trait_indices = tokenizer.encode(
        true2prompt[TRAITS[trait]], add_special_tokens=False
    )
    opposite_trait_indices = tokenizer.encode(
        true2prompt[OPPOSITE_TRAITS[trait]], add_special_tokens=False
    )

    if len(trait_indices) > len(opposite_trait_indices):
        diff_len = len(trait_indices) - len(opposite_trait_indices)
        for _ in range(diff_len):
            opposite_trait_indices += [tokenizer.unk_token_id]

    if len(opposite_trait_indices) > len(trait_indices):
        diff_len = len(opposite_trait_indices) - len(trait_indices)
        for _ in range(diff_len):
            trait_indices += [tokenizer.unk_token_id]

    added_token_id = 100

    for idx, mask_token_idx in enumerate(mask_token_indexes):

        tokens_logits = model(**masked_inputs).logits
        mask_token_logits = tokens_logits[0, [mask_token_idx], :]

        decoding_indices = []
        if idx == 0:
            decoding_indices = [trait_indices[idx]] + [opposite_trait_indices[idx]]
        if idx > 0:
            if added_token_id == trait_indices[idx - 1]:
                decoding_indices += [trait_indices[idx]]
            if added_token_id == opposite_trait_indices[idx - 1]:
                decoding_indices += [opposite_trait_indices[idx]]

        if len(decoding_indices) == 0:
            return [], [], True

        decoding_indices = torch.tensor(decoding_indices)
        decoding_indices_oh = torch.nn.functional.one_hot(
            decoding_indices, len(tokenizer.vocab)
        )

        mask_unimportant = torch.sum(decoding_indices_oh, 0) == 0
        mask_unimportant = mask_unimportant.repeat(mask_token_logits.shape[0], 1)
        mask_token_logits[mask_unimportant] = -math.inf
        added_token_id = torch.topk(mask_token_logits, 1, dim=1).indices.tolist()[0][0]
        masked_inputs["input_ids"][0, mask_token_idx] = added_token_id

    pred_labels = []

    predicted_prompt = tokenizer.decode(masked_inputs["input_ids"][0])
    splitted = predicted_prompt[
        predicted_prompt.find(MID_PROMPT.lower()) + len(MID_PROMPT.lower()) :
    ].split()
    if len(splitted) > 0:
        pred_label = splitted[0].lower()
    else:
        return [], [], True

    pred_labels += [pred_label]

    true_labels_discrete = []
    pred_labels_discrete = []

    for pred_label, true_label in zip(pred_labels, true_labels):

        pred_trait = prompt2true.get(pred_label)
        true_trait = prompt2true.get(true_label)

        if true_trait and traits2class_dicts[trait].get(true_trait) is not None:
            true_labels_discrete += [traits2class_dicts[trait][true_trait]]
            if pred_trait and traits2class_dicts[trait].get(pred_trait) is not None:
                pred_labels_discrete += [traits2class_dicts[trait].get(pred_trait)]
            else:
                return [], [], True

    if is_training:
        model.train()

    return true_labels_discrete, pred_labels_discrete, False
