import torch

from liger_kernel.chunked_loss import LigerFusedLinearPPOLoss

# @pytest.mark.parametrize("use_ref_model", [False, True])
def test_ppo_loss_forward_backward(use_ref_model):
    # Hyperparameters for the test
    batch_size = 16
    seq_len = 5
    hidden_size = 8
    vocab_size = 10

    # Create random inputs
    # _input -> shape: (batch_size * seq_len, hidden_size)
    _input = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    lin_weight = torch.randn(vocab_size, hidden_size, requires_grad=True)
    bias = torch.randn(vocab_size, requires_grad=True)

    # Create random targets (with values in [0, vocab_size-1]) 
    # shape: (batch_size * seq_len,)
    target = torch.randint(
        low=0, 
        high=vocab_size, 
        size=(batch_size, seq_len)
    )

    # Create random preference labels
    # shape: (batch_size,) -> booleans
    preference_labels = torch.randint(0, 2, (batch_size,)).bool()

    # Optionally create reference model inputs, weights, biases
    if use_ref_model:
        ref_input = _input.detach()
        ref_weight = torch.randn(vocab_size, hidden_size, requires_grad=True)
        ref_bias = torch.randn(vocab_size, requires_grad=True)
    else:
        ref_input = None
        ref_weight = None
        ref_bias = None

    # Instantiate the PPO loss module
    ppo_loss_fn = LigerFusedLinearPPOLoss(
        ignore_index=-100,
        # beta=0.1,
        compiled=True,       
        use_ref_model=use_ref_model,
        chunk_size=4,
    )

    # Forward pass
    loss, _ = ppo_loss_fn(
        _input=_input,
        lin_weight=lin_weight,
        bias=bias,
        target=target,
        preference_labels=preference_labels,
        ref_input=ref_input,
        ref_weight=ref_weight,
        ref_bias=ref_bias,
    )

    # Check the loss is a scalar
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"

    # Backward pass
    loss.backward()

    # Check gradients
    assert _input.grad is not None, "Input gradient should not be None."
    assert lin_weight.grad is not None, "Linear weight gradient should not be None."
    assert bias.grad is not None, "Bias gradient should not be None."

    # print("Gradient on lin_weight:", lin_weight.grad)
    # print("Gradient on _input:", _input.grad)
    # print("Gradient on ref_weight:", ref_weight.grad)
    # print("Gradient on ref_input:", ref_input.grad)

    print(f"Test passed for use_ref_model={use_ref_model} with loss={loss.item():.4f}.")


test_ppo_loss_forward_backward(use_ref_model=True)

