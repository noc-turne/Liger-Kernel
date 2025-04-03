import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# Define the PPO loss class
class PolicyLoss(nn.Module):
    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # print("advantages", advantages)
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        # print("ratio:", ratio)
        # print("surr1", surr1)
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        # print("loss", loss.shape)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss


# Define the masked mean function
def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


# The model class
class PPOModel(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, clip_eps: float = 0.2, lin_weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None, dtype = torch.float32):
        super().__init__()
        # Linear layer to map hidden_size to vocab_size (for logits)

        self.linear = nn.Linear(hidden_size, vocab_size, dtype=dtype)
        if lin_weight is not None and bias is not None:
            # Manually set the weight and bias
            self.linear.weight = nn.Parameter(lin_weight, requires_grad=True)
            self.linear.bias = nn.Parameter(bias, requires_grad=True)
        # PPO Loss function
        self.actor_loss_fn = PolicyLoss(clip_eps)

    def forward(self, input_tensor: torch.Tensor, labels: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass that computes the PPO loss.
        
        Arguments:
        - input_tensor: [batch_size, seq_len, hidden_size] input sequence.
        - labels: [batch_size, seq_len] target labels.
        - old_log_probs: [batch_size, seq_len] log probabilities from the previous policy.
        - advantages: [batch_size, seq_len] advantages for the current state-action pair.
        - action_mask: Optional [batch_size, seq_len] mask for valid actions.
        
        Returns:
        - loss: PPO loss computed from the log probabilities and the advantages.
        """
        # Compute logits from the input_tensor through the linear layer
        logits = self.linear(input_tensor)  # [batch_size, seq_len, vocab_size]

        # Compute the log probabilities from logits for all tokens in the sequence
        log_probs = log_probs_from_logits(logits, labels)

        # print("log_probs", log_probs.shape)
        
        # Compute PPO loss for each time step in the sequence
        loss = self.actor_loss_fn(
            log_probs,
            old_log_probs,
            advantages,
            action_mask=action_mask
        )
        
        return loss


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits.div_(temperature)
    batch_dim = logits.shape[:-1]
    last_dim = logits.shape[-1]
    logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
    logsumexp_values = logsumexp_values.view(*batch_dim)
    log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)

    # print("log_probs_labels", log_probs_labels)
    return log_probs_labels


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)

    return logsumexp_values



if "__name__" == "__main__":

    # Example usage:
    batch_size = 8
    seq_len = 10
    hidden_size = 128
    vocab_size = 50

    # Example inputs
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    old_log_probs = torch.randn(batch_size, seq_len)  # Old log probabilities
    advantages = torch.randn(batch_size, seq_len)  # Advantage values
    # action_mask = torch.ones(batch_size, seq_len)  # Action mask (optional)
    action_mask = None

    # Instantiate the model and compute the loss

    lin_weight = torch.randn(vocab_size, hidden_size, requires_grad=True)
    bias = torch.randn(vocab_size, requires_grad=True)
    model = PPOModel(hidden_size, vocab_size, lin_weight=lin_weight, bias=bias)
    loss = model(input_tensor, labels, old_log_probs, advantages, action_mask)


    loss.backward()

    print("PPO Loss:", loss.item())
