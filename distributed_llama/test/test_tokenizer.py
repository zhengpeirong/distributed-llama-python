"""Basic tests for tokenizer module."""

import os
import struct
import tempfile


def _write_minimal_tokenizer(path):
    """Write a minimal valid tokenizer file for testing."""
    vocab = [b"hello", b"world", b"<s>", b"</s>"]
    scores = [0.1, 0.2, 0.0, 0.0]
    bos_id = 2
    eos_tokens = [3]
    chat_template = "<|start_header_id|>{role}<|end_header_id|>\n\n{message}</s>"

    with open(path, "wb") as f:
        # Magic
        f.write(struct.pack("<i", 0x567124))
        # Header
        params = {
            0: 1,  # version
            1: len(vocab),  # vocab_size
            2: 10,  # max_token_length
            3: bos_id,  # bos_id
            7: len(chat_template),  # chat_template
            9: 1,  # n_eos_tokens
            10: 1,  # add_bos
        }
        header = b""
        for k, v in params.items():
            header += struct.pack("<ii", k, v)
        header_size = 8 + len(header)
        f.write(struct.pack("<i", header_size))
        f.write(header)
        f.write(chat_template.encode("utf-8"))
        for e in eos_tokens:
            f.write(struct.pack("<i", e))
        for i in range(len(vocab)):
            f.write(struct.pack("<fI", scores[i], len(vocab[i])))
            f.write(vocab[i])


def test_tokenizer_load():
    from distributed_llama.tokenizer import Tokenizer

    with tempfile.NamedTemporaryFile(suffix=".t", delete=False) as tmp:
        _write_minimal_tokenizer(tmp.name)
        tmp_path = tmp.name

    try:
        tok = Tokenizer(tmp_path)
        assert tok.vocab_size == 4
        assert tok.bos_id == 2
        assert tok.eos_token_ids == [3]
        assert tok.add_bos is True
        assert tok.chat_template is not None
        assert "<|start_header_id|>" in tok.chat_template

        # Test encode (use text matching exact vocab entries)
        tokens = tok.encode("helloworld", is_start=True, add_special_tokens=True)
        assert len(tokens) >= 2, f"Expected >= 2 tokens, got {tokens}"
        assert tokens[0] == 2, f"Expected BOS(2), got {tokens[0]}"  # BOS

        # Test decode
        tok.reset_decoder()
        for t in tokens[1:]:
            piece = tok.decode(t)
        print(f"  Tokenizer loaded and encoded: {tokens}")
    finally:
        os.unlink(tmp_path)


def test_sampler():
    from distributed_llama.tokenizer import Sampler

    sampler = Sampler(vocab_size=10, temperature=0.0)
    logits = [0.1, 0.2, 0.9, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    token = sampler.sample(logits)
    assert token == 2  # argmax

    sampler.set_temp(1.0)
    token = sampler.sample(logits)
    assert 0 <= token < 10
    print(f"  Sampler: argmax=2, sampled={token}")


def test_chat_template():
    from distributed_llama.tokenizer import (
        ChatTemplateGenerator, ChatTemplateType, ChatItem,
    )

    gen = ChatTemplateGenerator(ChatTemplateType.LLAMA3, None, "</s>")
    items = [
        ChatItem("system", "You are helpful."),
        ChatItem("user", "Hello!"),
    ]
    content, public_prompt = gen.generate(items, True)
    assert "<|start_header_id|>system" in content
    assert "<|start_header_id|>user" in content
    assert "<|start_header_id|>assistant" in content
    print(f"  Chat template: OK")


if __name__ == "__main__":
    test_tokenizer_load()
    test_sampler()
    test_chat_template()
    print("All tokenizer tests passed!")
