import torch
from liger_kernel.chunked_loss.fused_linear_unpaired_preference import LigerFusedLinearUnpairedPreferenceBase


class LigerFusedLinearPPOFunction(LigerFusedLinearUnpairedPreferenceBase):
    @staticmethod
    def preference_loss_fn(
        log_prob_chunk,
        preference_labels_chunk,
        full_target,
        ref_log_prob_chunk=None,
        beta=0.2,
    ):
        """
        Implements a simplified PPO clipped loss driven by preference labels.

        For each batch element:
        - ratio = exp(new_log_prob - old_log_prob)
          (where old_log_prob can come from a reference model or the policy 'before update')
        - advantage A is defined by preference_labels_chunk:
            +1 if the label is "chosen" (True)
            -1 if the label is "rejected" (False)
        - The clipped PPO objective is:
            L_PPO = - E[ min(ratio * A, clip(ratio, 1 - clip_range, 1 + clip_range) * A) ]

        Args:
            log_prob_chunk: Log probabilities for the chunk (average the seq_len) (batch_size, seq_len)
            preference_labels_chunk: Boolean preferences for each sample(average) (batch_size, seq_len)
            full_target: Non-chunked full target tensor (batch_size, )
            ref_log_prob_chunk: Old (or reference) model log probs (average the seq_len) (batch_size,)
            clip_range: Clipping range for PPO
        Returns:
            - loss: The PPO loss value (scalar)
            - chosen_rewards_sum: Sum of (ratio * advantage) for chosen samples (not returned)
            - rejected_rewards_sum: Sum of (ratio * advantage) for rejected samples (not returned)
        """

        # print(average_log_prob_chunk.shape)
        # print(average_log_prob_chunk)
        # ratio = exp(new_log_prob - old_log_prob)
        # print("log_prob_chunk:", log_prob_chunk.shape)
        # print("log_prob_chunk_value:", log_prob_chunk)
        # print("ref_prob_chunk:", ref_log_prob_chunk)
        if ref_log_prob_chunk is not None:
            ratio_chunk = torch.exp(log_prob_chunk - ref_log_prob_chunk)
        else:
            # If no reference is provided, treat the ratio as exp(log_prob_chunk)
            ratio_chunk = torch.exp(log_prob_chunk)

        # print("ratio_chunk:", ratio_chunk)

        # Advantage is +1 if chosen, -1 if rejected
        advantage_chunk = torch.where(preference_labels_chunk, 1.0, -0.01).unsqueeze(-1)

        # print("advantage_chunk", advantage_chunk)

        # Unclipped objective
        obj_unclipped = ratio_chunk * advantage_chunk
        # print("obj_unclipped", obj_unclipped)

        # Clipped objective
        ratio_clipped = torch.clamp(ratio_chunk, 1.0 - beta, 1.0 + beta)
        # print("ratio_clipped", ratio_clipped)
        obj_clipped = ratio_clipped * advantage_chunk
        # print("obj_clipped", obj_clipped)

        # PPO's clipped loss: - min(obj_unclipped, obj_clipped)
        # negative sign because we usually maximize PPO objective
        losses = -torch.minimum(obj_unclipped, obj_clipped)

        # print("losses.shape", losses.shape)
        # print("losses", losses)

        # if full_target is not None:
        # print(full_target.shape)


        # For logging: sum of ratio * advantage for chosen/rejected
        # chosen_rewards_sum = (obj_unclipped * preference_labels_chunk).sum()
        # rejected_rewards_sum = (obj_unclipped * (~preference_labels_chunk)).sum()

        # Return mean loss over the batch
        # return losses.sum() / (full_target.shape[0]), chosen_rewards_sum, rejected_rewards_sum

        # print("losses", losses.sum() / (full_target.shape[0]))
        # print("sum_loss", (losses.sum()).shape)
        # print("losses.shape", losses.shape)
        return losses.mean()


    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        target,
        preference_labels,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        ignore_index=-100,
        beta=0.2,
        compiled=True,
        use_ref_model=True,
        chunk_size=1,
    ):
        """
        Fused linear layer with a PPO-style preference loss.

        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, hidden_size)
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
            target (torch.LongTensor): Target tensor. Shape: (batch_size, seq_len,)
            preference_labels (torch.Tensor): Boolean preference labels (batch_size,)
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
            ref_input (torch.Tensor, optional): Reference model input tensor.
            ref_weight (torch.Tensor, optional): Reference model weight tensor.
            ref_bias (torch.Tensor, optional): Reference model bias tensor.
            ignore_index (int): Index to ignore in loss computation
            beta (float): Clipping parameter for PPO
            compiled (bool): Whether to use torch.compile or not
            use_ref_model (bool): Whether to use the reference model
            average_log_prob (bool): Average log prob per unmasked token or not
            chunk_size (int): Size of chunks for processing
        Returns:
            torch.Tensor: Computed PPO loss
        """
        return LigerFusedLinearUnpairedPreferenceBase.forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            target=target,
            preference_labels=preference_labels,
            bias=bias,
            ignore_index=ignore_index,
            beta=beta, 
            compiled=compiled,
            use_ref_model=use_ref_model,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            chunk_size=chunk_size,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Pass gradients back through the linear layer,
        ignoring the gradients for the additional PPO-specific arguments.
        """
        grads = LigerFusedLinearUnpairedPreferenceBase.backward(ctx, grad_output)[:5]
        return (
            *grads,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LigerFusedLinearPPOLoss(torch.nn.Module):
    """
    Fused linear layer with a PPO-style preference loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        clip_range: float = 0.2,
        compiled: bool = True,
        use_ref_model: bool = False,
        chunk_size: int = 1,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss calculation
            clip_range (float): PPO clipping parameter
            compiled (bool): Whether to use compiled operations
            use_ref_model (bool): Whether to use a reference model
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.clip_range = clip_range
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.chunk_size = chunk_size

    def forward(
        self,
        _input,
        lin_weight,
        target,
        bias=None,
        preference_labels=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        """
        Forward pass computing the fused linear + PPO loss.

        Args:
            _input (torch.Tensor): Input tensor. (batch_size*seq_len, hidden_dim)
            lin_weight (torch.Tensor): Linear layer weights. (vocab_size, hidden_dim)
            target (torch.LongTensor): Target tokens. (batch_size*seq_len,)
            bias (torch.Tensor, optional): Linear layer bias. (vocab_size,)
            preference_labels (torch.Tensor, optional): Boolean preference labels. (batch_size,)
            ref_input/ref_weight/ref_bias (torch.Tensor, optional): For reference model

        Returns:
            torch.Tensor: A scalar loss value
        """
        return LigerFusedLinearPPOFunction.apply(
            _input,
            lin_weight,
            target,
            preference_labels,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            self.ignore_index,
            self.clip_range,
            self.compiled,
            self.use_ref_model,
            self.chunk_size,
        )
