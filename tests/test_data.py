import torch
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM

from trialcode.data.collator import CodeAwareMLMCollator
from trialcode.data.mimic import VARIANT_KINDS, code_ancestors, code_token, code_vocabulary, load_mimic
from trialcode.data.tokenize import make_tokenize_fn
from trialcode.data.vocab import add_code_tokens


def test_ancestors_and_tokens():
    assert code_ancestors("diagnosis", "10", "E1165") == ["E116", "E11"]
    assert code_ancestors("diagnosis", "9", "0389") == ["038"]
    assert code_ancestors("drug", "0", "metformin") == []
    assert code_token("diagnosis", "10", "E119") == "[DX10:E119]"
    assert code_token("drug", "0", "insulin regular") == "[RX:insulin_regular]"


def test_leading_zeros_preserved(mimic_root):
    codes = load_mimic(mimic_root, "codes")[0]["codes"]
    assert "0BH17EZ" in [c["code"] for c in codes]


def test_vocab_prunes_chains_and_links_parents(mimic_root):
    vocab = {v["token"]: v for v in code_vocabulary(mimic_root, 1, VARIANT_KINDS["codes"])}
    assert vocab["[DX10:E1165]"]["parent"] == "[DX10:E11]"  # E116 is a one-child chain link -> pruned
    assert "[DX10:E116]" not in vocab
    assert all(v["parent"] in vocab for v in vocab.values() if v["parent"])


def test_rare_code_backs_off_to_ancestor(mimic_root):
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    vocab = code_vocabulary(mimic_root, 11, VARIANT_KINDS["codes"])  # E1165 (10 admissions) is below the cutoff
    tok.add_tokens([v["token"] for v in vocab])
    row = {"text": [""], "codes": [[{"kind": "diagnosis", "icd_version": "10", "code": "E1165", "title": "x"}]]}
    ids = make_tokenize_fn(tok, 64, code_tokens=True, include_titles=False)(row)["input_ids"][0]
    assert "[DX10:E11]" in tok.convert_ids_to_tokens(ids)


def test_collator_masks_whole_codes():
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    tok.add_tokens(["[DX10:A]", "[DX10:B]"])
    a, b = tok.convert_tokens_to_ids(["[DX10:A]", "[DX10:B]"])
    f = {"input_ids": [tok.cls_token_id, 5, 6, tok.sep_token_id, a, b, tok.sep_token_id],
         "code_group": [0, 0, 0, 0, 1, 2, 0], "code_part": [0, 0, 0, 0, 1, 1, 0]}
    out = CodeAwareMLMCollator(tok, mlm_probability=0.0, code_prob=1.0)([f])
    assert (out["labels"][0][4:6] != -100).all() and (out["labels"][0][:4] == -100).all()
    assert out["input_ids"].shape[1] % 8 == 0


def test_hierarchical_embedding_starts_at_description_mean(mimic_root):
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    cfg = ModernBertConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, pad_token_id=tok.pad_token_id)
    model = ModernBertForMaskedLM(cfg)
    base = model.get_input_embeddings().weight.detach().clone()
    vocab = code_vocabulary(mimic_root, 1, VARIANT_KINDS["codes"])
    add_code_tokens(tok, model, vocab)
    v = next(v for v in vocab if v["code"] == "E1165")
    got = model.get_input_embeddings()(torch.tensor([tok.convert_tokens_to_ids(v["token"])]))[0]
    want = base[tok(v["title"], add_special_tokens=False)["input_ids"]].mean(0)
    assert torch.allclose(got, want, atol=1e-5)
    assert model.decoder.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr()
