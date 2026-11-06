"""Add one token per code to a tokenizer/model, initialised from the code's description."""
import torch


def add_code_tokens(tokenizer, model, vocab):
    """Register `vocab` (from mimic.code_vocabulary) as tokens and resize the model.

    New embedding rows start at the mean embedding of the code's title tokens, so a code is
    text-aware at step 0 and then drifts toward how it is used. Embeddings are tied to the MLM
    head, so the same rows score the code as an output.
    """
    titles = [tokenizer(v["title"] or v["code"], add_special_tokens=False)["input_ids"] for v in vocab]
    tokenizer.add_tokens([v["token"] for v in vocab])
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for v, t in zip(vocab, titles):
            emb[tokenizer.convert_tokens_to_ids(v["token"])] = emb[t].mean(dim=0)


def description_examples(vocab, with_text):
    """One row per code. with_text=True puts the title in the note slot (title -> code);
    False leaves it empty so the code section's own title is the only text (code -> title)."""
    return [{"note_id": "", "subject_id": "0", "hadm_id": "0",
             "text": v["title"] if with_text else "",
             "codes": [{k: v[k] for k in ("kind", "icd_version", "code", "title")}]}
            for v in vocab]
