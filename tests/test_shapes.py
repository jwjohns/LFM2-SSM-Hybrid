import pytest
import torch


@pytest.mark.parametrize("vocab,d,layers,attn_every", [(128, 64, 2, 2), (256, 128, 3, 1)])
def test_forward_shapes(vocab, d, layers, attn_every):
    from lfm2_hybrid.model import LFM2_SSM_Small

    B, T = 2, 8
    model = LFM2_SSM_Small(vocab, d, layers, attn_every=attn_every)
    tokens = torch.randint(0, vocab, (B, T))
    logits = model(tokens)
    assert logits.shape == (B, T, vocab)
