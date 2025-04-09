# Liger PPO Loss 测试与对比分析

本目录旨在测试与对比 [Liger-Kernel](https://github.com/noc-turne/Liger-Kernel/tree/main/) 中实现的 PPO Loss，并对比 OpenRLHF 中的实现。该测试与分析有助于理解不同实现之间的性能差异，包括运行效率和显存占用。

## 内容概览

- **FusedLinearPPOLoss**  
  来自 `liger-kernel` 的 PPO loss 实现，采用 FusedLinear 的方式进行优化，加速前向与反向传播过程, 具体实现可参考https://aicarrier.feishu.cn/wiki/C3Uhwc2uDiCT6DkcgJsczatGn9i

- **liger_ppo_test.py**  
  用于测试 `liger-kernel` 中的 `FusedLinearPPOLoss` 是否功能正确。适合用作基础单元测试验证。

- **openrlhf_ppo_loss.py**  
  提取自 [OpenRLHF](https://github.com/OpenLMLab/OpenRLHF) 中的 PPO loss 实现，移植到本目录下进行本地测试。

- **compare_ppo.py**  
  用于对比 `liger-kernel` 与 `openrlhf` 两个 PPO loss 实现的运行时间与显存使用情况。支持调节以下参数：
  - `hidden_size`
  - `vocab_size`
  - `batch_size`
  - `seq_len`
  - `chunk_size`

- **res/**  
  存放 `compare_ppo.py` 中生成的分析图，包括显存占用与执行时间等可视化结果。

## 快速开始

```bash
git clone https://github.com/noc-turne/Liger-Kernel/tree/main/ 
cd Liger-Kernel
pip install -e .

# 对比两种实现
cd liger_kernel_test
python compare_ppo.py 
```

---

