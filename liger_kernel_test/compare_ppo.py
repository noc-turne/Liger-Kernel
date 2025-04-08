import torch
import time
import matplotlib.pyplot as plt
from datetime import datetime
from openrlhf_ppo_loss import PPOModel, log_probs_from_logits
from liger_kernel.chunked_loss import LigerFusedLinearPPOLoss

hidden_size = 768
vocab_size = 10000
batch_size = 8
seq_lens = [128 * (2**i) for i in range(5)]  # 128, 256, 512, 1024, 2048
dtype = torch.float32
chunk_size = 1


def benchmark_ppo_losses():
    num_runs = 5  # number of times to repeat (excluding warmup)

    openrlhf_avg_times = []
    liger_avg_times = []

    openrlhf_avg_peak = []
    liger_avg_peak = []

    for seq_len in seq_lens:
        openrlhf_times = []
        liger_times = []

        openrlhf_peak_usages = []
        liger_peak_usages = []

        for run in range(num_runs + 1):  # extra run for warmup
            input = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
            lin_weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
            bias = torch.randn(vocab_size, device="cuda", dtype=dtype, requires_grad=True)
            target = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
            preference_labels = torch.randint(0, 2, (batch_size,), device="cuda").bool()

            ref_input = input.detach()

            noise_scale = 0.01  # 控制扰动大小
            ref_weight = lin_weight.detach() + noise_scale * torch.randn_like(lin_weight)
            ref_bias = bias.detach() + noise_scale * torch.randn_like(bias)

            # === OpenRLHF PPOModel ===
            model = PPOModel(hidden_size, vocab_size, lin_weight=lin_weight, bias=bias, dtype=dtype).to("cuda")

            
            start_time = time.time()
            ref_logits = torch.matmul(ref_input, ref_weight.t()) + ref_bias
            ref_log_probs = log_probs_from_logits(ref_logits, target)
            advantages = preference_labels.unsqueeze(1).expand(-1, seq_len)
            advantages = torch.where(advantages, torch.tensor(1.0 + 1e-8, device="cuda", dtype=dtype, requires_grad=False),
                                     torch.tensor(-0.01 + 1e-8, device="cuda", dtype=dtype, requires_grad=False))
            action_mask = None

            torch.cuda.synchronize()

            openrlhf_loss = model(input, target, ref_log_probs, advantages, action_mask)
            openrlhf_loss.backward()
            torch.cuda.synchronize()


            if run > 0:
                openrlhf_times.append(time.time() - start_time)
                openrlhf_lin_grad = model.linear.weight.grad
                peak_mem = torch.cuda.max_memory_allocated()
                openrlhf_peak_usages.append(peak_mem)

            torch.cuda.reset_max_memory_allocated()


            # === Liger PPO Loss ===
            ppo_loss_fn = LigerFusedLinearPPOLoss(
                ignore_index=-100,
                compiled=True,
                use_ref_model=True,
                chunk_size=chunk_size,
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
                peak_mem = torch.cuda.max_memory_allocated()
                liger_peak_usages.append(peak_mem)

                assert torch.allclose(openrlhf_loss, liger_loss, rtol=1e-2, atol=1e-3), "loss differs"
                diff = torch.abs(openrlhf_lin_grad - liger_lin_grad)
                max_diff = diff.max()
                if torch.isnan(openrlhf_lin_grad).any():
                    print("openrlhf value error")
                if torch.isnan(liger_lin_grad).any():
                    print("liger value error")

                assert torch.allclose(openrlhf_lin_grad, liger_lin_grad, rtol=1e-1, atol=1e-2), \
                    f"grad differs: max diff = {max_diff.item()}"

            torch.cuda.reset_max_memory_allocated()

            # Free memory
            del input, lin_weight, bias, target, preference_labels
            del ref_input, ref_weight, ref_bias, model, openrlhf_loss, liger_loss, ref_logits, ref_log_probs, advantages
            
            torch.cuda.empty_cache()

        openrlhf_avg_times.append(sum(openrlhf_times) / num_runs)
        liger_avg_times.append(sum(liger_times) / num_runs)

        openrlhf_avg_peak.append(sum(openrlhf_peak_usages) / num_runs)
        liger_avg_peak.append(sum(liger_peak_usages) / num_runs)

    print("Test completed")

    return seq_lens, openrlhf_avg_times, liger_avg_times, openrlhf_avg_peak, liger_avg_peak

# Run the benchmark
seq_lens, openrlhf_times, liger_times, openrlhf_peak, liger_peak = benchmark_ppo_losses()

# Plotting the results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Time
time_save_path = f"/fs-computility/ai-shen/marunmin/liger_kernel_test/res/ppo_loss_time_{timestamp}.png"

plt.figure(figsize=(10, 6))
plt.plot(seq_lens, openrlhf_times, label="OpenRLHF PPOModel", marker='o')
plt.plot(seq_lens, liger_times, label="Liger PPO Loss", marker='s')
plt.xlabel("Sequence Length")
plt.ylabel("Average Time (seconds)")
plt.title(f"PPO Loss Computation Time ({dtype} hidden_size:{hidden_size} vocab_size:{vocab_size} batch_size:{batch_size} chunk_size:{chunk_size})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(time_save_path)

# Memory
memory_save_path = f"/fs-computility/ai-shen/marunmin/liger_kernel_test/res/ppo_loss_memory_{timestamp}.png"

print("open", openrlhf_peak)
print("liger", liger_peak)
plt.figure(figsize=(10, 6))
plt.plot(seq_lens, openrlhf_peak, label="OpenRLHF PPOModel Peak Mem", marker='o')
plt.plot(seq_lens, liger_peak, label="Liger PPO Loss Peak Mem", marker='s')
plt.xlabel("Sequence Length")
plt.ylabel("Peak Memory Allocated (bytes)")
plt.title(f"PPO Loss Peak Memory ({dtype} hidden_size:{hidden_size} vocab_size:{vocab_size} batch_size:{batch_size} chunk_size:{chunk_size})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(memory_save_path)

