# Tests written by Claude Code

"""
Tests for Stage 1.1 — char-level tokenizer.

Run with:  pytest test_llm.py -v
(from /Users/julien/Desktop/llm4fun/)

These tests import `llm` as a module, which will execute everything at the
top level of llm.py (including any print statements). That's fine for now;
pytest will just show the prints. We may clean that up later with an
`if __name__ == "__main__":` guard, but it isn't required for correctness.
"""

import llm


def test_chars_is_sorted_and_unique():
    """
    The vocab must be deterministic across runs and have no duplicates.
    Every downstream piece of the model — the embedding table's rows, the
    final logit indices, the softmax distribution — is indexed by position
    in `chars`. If `chars` changes order between runs, a saved model
    becomes meaningless. If it has duplicates, two ids point to the same
    char and decode becomes ambiguous.
    """
    assert llm.chars == sorted(llm.chars)
    assert len(llm.chars) == len(set(llm.chars))


def test_vocab_size_matches_chars():
    """
    `vocab_size` is the name we'll reuse everywhere:
      - the embedding table will be shape (vocab_size, embedding_dim)
      - the final projection layer will output `vocab_size` logits per token
      - the cross-entropy loss will be computed over `vocab_size` classes
    It must equal len(chars). Getting these out of sync is a classic source
    of off-by-one bugs that show up much later as an IndexError.
    """
    assert llm.vocab_size == len(llm.chars)
    assert llm.vocab_size > 0


def test_stoi_itos_are_inverses():
    """
    stoi and itos must be exact inverses on the full vocab in both
    directions. If they're not, encode/decode round-trips will silently
    corrupt your training data — the model will train on one sequence
    and you'll think it's a different sequence.

    This is the kind of bug you cannot detect by looking at the loss
    going down; loss can drop perfectly while the model learns nonsense.
    """
    for c in llm.chars:
        assert llm.itos[llm.stoi[c]] == c
    for i in range(llm.vocab_size):
        assert llm.stoi[llm.itos[i]] == i


def test_encode_returns_ints_in_range():
    """
    Every encoded id must be a valid index into the embedding table:
    0 <= id < vocab_size. An out-of-range id will crash the model on
    the first forward pass with a cryptic CUDA/MPS error — and on MPS
    in particular, the error messages are often unhelpful, so we'd
    rather catch this here.
    """
    sample = "".join(llm.chars)  # a string containing one of every vocab char
    ids = llm.encode(sample)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < llm.vocab_size for i in ids)


def test_encode_is_length_preserving():
    """
    Char-level tokenization: one character in => one id out. Always.

    This is a *defining* property of char-level. When you eventually try
    BPE (byte-pair encoding) in Pass 2, this property goes away — one
    word might collapse to a single token. So this test is also a
    marker reminding future-you which tokenization scheme is in play.
    """
    s = "".join(llm.chars)
    assert len(llm.encode(s)) == len(s)


def test_encode_decode_roundtrip():
    """
    The point of the tokenizer in one line:
        decode(encode(s)) == s    for any s drawn from the vocab.

    A failure here means your model's training targets are not what you
    think they are. The model will still learn *something*, but it will
    be learning to predict a corrupted version of your corpus. This is
    one of the worst categories of bugs because everything *looks* fine.
    """
    s = "".join(llm.chars) * 3  # repeat the alphabet 3x for a non-trivial test
    assert llm.decode(llm.encode(s)) == s


def test_decode_encode_roundtrip_on_ids():
    """
    The other direction:
        encode(decode(ids)) == ids    for any list of valid ids.

    Together with the previous test, this confirms encode and decode
    are mutual inverses on their full domains.
    """
    ids = list(range(llm.vocab_size))
    assert llm.encode(llm.decode(ids)) == ids


def test_encode_empty_string():
    """
    Edge case: empty input. Easy to get wrong with naive code that
    assumes the input has at least one character. Important because
    during generation we sometimes feed very small or empty contexts.
    """
    assert llm.encode("") == []


def test_decode_empty_list():
    """
    Edge case: empty ids. Same reasoning as above.
    """
    assert llm.decode([]) == ""


def test_encode_single_chars_match_stoi():
    """
    Encoding a single character must agree with looking it up in stoi.
    This catches the common bug where encode does something extra
    (e.g., prepending a special token like <BOS>) that the round-trip
    tests above would still pass — because a symmetric extra step in
    decode could cancel it out.
    """
    for c in llm.chars:
        assert llm.encode(c) == [llm.stoi[c]]
