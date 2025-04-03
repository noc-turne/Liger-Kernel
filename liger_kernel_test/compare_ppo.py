import torch
import time
import matplotlib.pyplot as plt
from datetime import datetime
from openrlhf_ppo_loss import PPOModel, log_probs_from_logits
from liger_kernel.chunked_loss import LigerFusedLinearPPOLoss

def benchmark_ppo_losses():
    hidden_size = 4
    vocab_size = 5
    batch_size = 1
    seq_lens = [128 * (2**i) for i in range(5)]  # 128, 256, 512, 1024, 2048
    dtype = torch.float32

    num_runs = 5  # number of times to repeat (excluding warmup)

    openrlhf_avg_times = []
    liger_avg_times = []

    for seq_len in seq_lens:
        openrlhf_times = []
        liger_times = []

        for run in range(num_runs + 1):  # extra run for warmup
            input = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
            lin_weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
            bias = torch.randn(vocab_size, device="cuda", dtype=dtype, requires_grad=True)
            target = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
            preference_labels = torch.randint(0, 2, (batch_size,), device="cuda").bool()

            ref_input = input.detach()
            ref_weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype, requires_grad=False)
            ref_bias = torch.randn(vocab_size, device="cuda", dtype=dtype, requires_grad=False)

            # === OpenRLHF PPOModel ===
            model = PPOModel(hidden_size, vocab_size, lin_weight=lin_weight, bias=bias, dtype=dtype).to("cuda")

            torch.cuda.synchronize()
            start_time = time.time()
            ref_logits = torch.matmul(ref_input, ref_weight.t()) + ref_bias
            ref_log_probs = log_probs_from_logits(ref_logits, target)
            advantages = preference_labels.unsqueeze(1).expand(-1, seq_len)
            advantages = torch.where(advantages, torch.tensor(1.0, device="cuda", dtype=dtype),
                                     torch.tensor(-0.01, device="cuda", dtype=dtype))
            action_mask = None

            openrlhf_loss = model(input, target, ref_log_probs, advantages, action_mask)
            openrlhf_loss.backward()
            torch.cuda.synchronize()
            if run > 0:
                openrlhf_times.append(time.time() - start_time)
                openrlhf_lin_grad = model.linear.weight.detach().clone()
                # openrlhf_grads.append(openrlhf_lin_grad)

            # === Liger PPO Loss ===
            ppo_loss_fn = LigerFusedLinearPPOLoss(
                ignore_index=-100,
                compiled=True,
                use_ref_model=True,
                # chunk_size=128,
            )

            torch.cuda.synchronize()
            start_time = time.time()
            liger_loss, _ = ppo_loss_fn(
                _input=input,
                lin_weight=lin_weight,
                bias=bias,
                target=target,
                preference_labels=preference_labels,
                ref_input=ref_input,
                ref_weight=ref_weight,
                ref_bias=ref_bias,
            )
            liger_loss.backward()
            torch.cuda.synchronize()
            if run > 0:
                liger_times.append(time.time() - start_time)
                liger_lin_grad = lin_weight.grad.detach().clone()

                print("openrlhf_grad", openrlhf_lin_grad)
                print("liger_grad", liger_lin_grad)
                print("openrlhf_loss", openrlhf_loss)
                print("liger_loss", liger_loss)
                assert torch.allclose(openrlhf_loss, liger_loss, rtol=1e-2, atol=1e-3), "loss differs"
                assert torch.allclose(openrlhf_lin_grad, liger_lin_grad, rtol=1e-2, atol=1e-3), "grad differs"

            # Free memory
            del input, lin_weight, bias, target, preference_labels
            del ref_input, ref_weight, ref_bias, model, openrlhf_loss, liger_loss, ref_logits, ref_log_probs, advantages
            
            torch.cuda.empty_cache()

        openrlhf_avg_times.append(sum(openrlhf_times) / num_runs)
        liger_avg_times.append(sum(liger_times) / num_runs)

    return seq_lens, openrlhf_avg_times, liger_avg_times

# Run the benchmark
seq_lens, openrlhf_times, liger_times = benchmark_ppo_losses()

# Plotting the results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"/fs-computility/ai-shen/marunmin/liger_kernel_test/res/ppo_loss_{timestamp}.png"

plt.figure(figsize=(10, 6))
plt.plot(seq_lens, openrlhf_times, label="OpenRLHF PPOModel", marker='o')
plt.plot(seq_lens, liger_times, label="Liger PPO Loss", marker='s')
plt.xlabel("Sequence Length")
plt.ylabel("Average Time (seconds)")
plt.title("PPO Loss Computation Time (GPU, Averaged, bfloat16, chunk_size=1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path)
