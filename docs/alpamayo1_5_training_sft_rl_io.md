# Alpamayo 1.5 训练数据流：SFT 与 RL

本文专门讲训练阶段的数据输入输出，分为两部分：

1. SFT：监督微调，包括 stage1 离散轨迹 token 路径和 stage2 Alpamayo 1.5 action expert 路径。
2. RL：基于 Cosmos-RL/GRPO 的后训练路径，包括 rollout、reward、policy batch 和 loss。

阅读目标：把你当作刚入学的新生，不要求你先懂分布式训练或强化学习，但希望读完后能知道“一个 batch 里有哪些字段、形状是什么、被哪个函数消费、loss 从哪里来”。

## 1. 共同的数据基础

训练和评估都从 PAI 数据集读取样本。

核心文件：

- `src/alpamayo1_5/data/pai.py`
- `src/alpamayo1_5/load_physical_aiavdataset.py`
- `src/alpamayo1_5/processor/qwen_processor.py`
- `src/alpamayo1_5/chat_template/conversation.py`

基础样本字段：

| 字段 | 单样本形状 | batch 后形状 | 说明 |
|---|---:|---:|---|
| `image_frames` | `[C_cam,F_img,3,H,W]` | list 或特殊拼接 | 多相机多帧图像 |
| `camera_indices` | `[C_cam]` | `[B,C_cam]` 或 RL reshape 后 `[B,N_img]` | 相机 id |
| `relative_timestamps` | 与图像对应 | 与图像对应 | 时间戳 |
| `ego_history_xyz` | `[G,16,3]` | `[B,G,16,3]` | 历史 1.6s，自车位置 |
| `ego_history_rot` | `[G,16,3,3]` | `[B,G,16,3,3]` | 历史朝向 |
| `ego_future_xyz` | `[G,64,3]` | `[B,G,64,3]` | 未来 6.4s GT |
| `ego_future_rot` | `[G,64,3,3]` | `[B,G,64,3,3]` | 未来朝向 GT |
| `tokenized_data.input_ids` | `[1,L]` 或 `[L]` | `[B,L_pad]` | 语言模型 token ids |
| `tokenized_data.attention_mask` | `[1,L]` | `[B,L_pad]` | padding mask |
| `labels_mask` | `[1,L]` | `[B,L_pad]` | 哪些 token 参与 loss |

符号说明：

- `B`: batch size。
- `G`: trajectory group，目前推理通常要求 `G=1`。
- `L`: token 序列长度，取决于图片展开 token 数和文本长度。
- `L_pad`: batch 内 padding 后的最大长度。
- `T_h=16`: 历史点数。
- `T_f=64`: 未来点数。
- `D`: VLM/expert hidden size，运行时从 checkpoint config 读取。
- `V`: tokenizer 扩展后的词表大小。

## 2. SFT 总览

SFT 入口文件：

```text
finetune/sft/train_hf.py
```

主要流程：

1. Hydra 读取配置。
2. instantiate 模型。
3. instantiate train/eval dataset。
4. instantiate collate function。
5. 构造 `ReasoningVLA_Trainer`。
6. 调用 `trainer.train()`。

代码入口：

```python
@hydra.main(...)
def train(cfg):
    training_args = TrainingArguments(...)
    model = hydra.utils.instantiate(cfg.model)
    train_dataset = hydra.utils.instantiate(cfg.data.train_dataset, model_config=model.config)
    eval_dataset = hydra.utils.instantiate(cfg.data.val_dataset, model_config=model.config)
    collate_fn = hydra.utils.instantiate(cfg.data.collate_fn, model_config=model.config)
    trainer = ReasoningVLA_Trainer(...)
    trainer.train()
```

配置文件：

- `finetune/sft/configs/sft_base.yaml`
- `finetune/sft/configs/sft_stage1.yaml`
- `finetune/sft/configs/sft_stage2.yaml`
- `finetune/sft/configs/vla_processor.yaml`
- `finetune/sft/configs/models/ar1_base.yaml`
- `finetune/sft/configs/models/ar1_expert.yaml`

## 3. SFT 数据预处理

配置 `finetune/sft/configs/vla_processor.yaml`：

```yaml
components_order: ["image", "traj_history", "prompt", "traj_future"]
components_prompt: ["traj_future"]
label_components: ["traj_future"]
```

含义：

- 输入里有图像、历史轨迹和 prompt。
- assistant 部分要求模型输出未来轨迹。
- loss 只算 `traj_future` 组件。

### 3.1 QwenProcessor._preprocess_data

文件：`src/alpamayo1_5/processor/qwen_processor.py`

输入：

| 变量 | 形状 |
|---|---|
| `image_frames` | `[C_cam,F_img,3,H,W]` |
| `camera_indices` | `[C_cam]` |
| `ego_history_xyz` | `[G,16,3]` |
| `ego_future_xyz` | `[G,64,3]` |

输出 `tokenized_data`：

| 字段 | 形状 |
|---|---|
| `text` | string |
| `pixel_values` | Qwen image processor 输出 |
| `image_grid_thw` | `[N_img,3]` |

注意：此时通常还没有 `input_ids`，因为真正的文本 tokenizer padding 在 collate 时做。

### 3.2 QwenProcessor.collate_fn

输入：

- `data`: list of sample dict，长度为 `B`。

输出：

| 字段 | 形状 |
|---|---|
| `batch["ego_history_xyz"]` | `[B,G,16,3]` |
| `batch["ego_history_rot"]` | `[B,G,16,3,3]` |
| `batch["ego_future_xyz"]` | `[B,G,64,3]` |
| `batch["ego_future_rot"]` | `[B,G,64,3,3]` |
| `batch["tokenized_data"]["input_ids"]` | `[B,L_pad]` |
| `batch["tokenized_data"]["attention_mask"]` | `[B,L_pad]` |
| `batch["tokenized_data"]["pixel_values"]` | batch 内图像 processor 输出拼接 |
| `batch["tokenized_data"]["image_grid_thw"]` | `[sum_B N_img,3]` |
| `batch["labels_mask"]` | `[B,L_pad]` |

`labels_mask` 来自：

- `src/alpamayo1_5/utils/get_label_mask.py::get_label_mask()`
- `get_role_eos_mask()` 会额外把 assistant 的 `<|im_end|>` 也纳入 loss。

## 4. SFT stage1：离散轨迹 token 路径

模型类：

```text
finetune/sft/models/sft_base_model.py::TrainableReasoningVLA
```

配置：

```text
finetune/sft/configs/models/ar1_base.yaml
```

stage1 的目标：让 VLM 学会把未来轨迹作为离散 token 自回归生成。

### 4.1 forward 输入

函数：

```python
TrainableReasoningVLA.forward(
    tokenized_data,
    ego_history_xyz,
    ego_history_rot,
    ego_future_xyz,
    ego_future_rot,
    labels_mask,
)
```

输入形状：

| 参数 | 形状 |
|---|---|
| `tokenized_data["input_ids"]` | `[B,L]` |
| `ego_history_xyz` | `[B,G,16,3]` |
| `ego_history_rot` | `[B,G,16,3,3]` |
| `ego_future_xyz` | `[B,G,64,3]` |
| `ego_future_rot` | `[B,G,64,3,3]` |
| `labels_mask` | `[B,L]` |

### 4.2 历史和未来轨迹 token 注入

`TrainableReasoningVLA` 使用 `TrajectoryFusionWithFutureMixin.fuse_traj_tokens()`。

它会做两次替换：

1. 历史轨迹：
   - `tokenize_history_trajectory()` 输出 `[B, G*tokens_per_history_traj]`
   - 替换 `<|traj_history|>` 占位符
2. 未来轨迹：
   - `tokenize_future_trajectory()` 输出 `[B, G*tokens_per_future_traj]`
   - 替换 `<|traj_future|>` 占位符

输出：

- `input_ids [B,L]`，形状不变，但占位符被真实离散轨迹 token 替换。

### 4.3 VLM forward 与 logits

代码：

```python
labels = input_ids.clone()
labels = torch.where(labels_mask, labels, IGNORE_INDEX)
outputs = self.vlm(input_ids=input_ids, labels=labels, **tokenized_data)
```

输出：

| 变量 | 形状 |
|---|---|
| `outputs.logits` | `[B,L,V]` |
| `labels` | `[B,L]` |

### 4.4 loss 划分

代码：

```python
traj_mask = (
    (labels >= future_token_start_idx)
    & (labels < future_token_start_idx + traj_vocab_size)
) | future_start/end special token

losses["future_traj"] = CE(outputs.logits, labels, traj_mask)
labels[traj_mask] = IGNORE_INDEX
losses["others"] = CE(outputs.logits, labels, labels != IGNORE_INDEX)
outputs.loss = sum(losses.values())
```

形状：

| 变量 | 形状 | 含义 |
|---|---|---|
| `traj_mask` | `[B,L]` | 哪些位置属于 future trajectory token |
| `shift_logits` | `[num_valid_tokens,V]` | 参与 CE 的 logits |
| `shift_labels` | `[num_valid_tokens]` | 目标 token |
| `loss` | scalar | 总 loss |

stage1 本质是语言模型 next-token prediction，只是 future trajectory token 被单独识别出来，可以单独加权。

## 5. SFT stage2：Alpamayo 1.5 action expert 路径

模型类：

```text
finetune/sft/models/sft_alpamayo1_5.py::TrainableAlpamayo1_5
```

配置：

```text
finetune/sft/configs/models/ar1_expert.yaml
```

stage2 的目标：训练 Alpamayo 1.5 的 continuous action expert。它不再让 VLM 直接生成离散 future token，而是让 expert diffusion head 学会根据 VLM 上下文预测 flow matching vector field。

### 5.1 forward 输入

函数：

```python
TrainableAlpamayo1_5.forward(...)
```

输入形状：

| 参数 | 形状 |
|---|---|
| `tokenized_data["input_ids"]` | `[B,L]` |
| `ego_history_xyz` | `[B,G,16,3]` |
| `ego_history_rot` | `[B,G,16,3,3]` |
| `ego_future_xyz` | `[B,G,64,3]` |
| `ego_future_rot` | `[B,G,64,3,3]` |
| `labels_mask` | `[B,L]` |

当前代码里 `action = action.reshape(-1, *action_dims)`，所以如果 `G=1`，action batch 是 `B`；如果 `G>1`，会变成 `B*G`，但 VLM cache batch 仍是 `B`，因此实际训练应保持 `G=1`。

### 5.2 VLM 前向

代码：

```python
input_ids = self.fuse_traj_tokens(input_ids, traj_data)
labels = input_ids.clone()
labels = torch.where(labels_mask, labels, IGNORE_INDEX)
vlm_outputs = self.vlm(input_ids=input_ids, labels=labels, use_cache=True, **tokenized_data)
```

输出：

| 变量 | 形状 |
|---|---|
| `vlm_outputs.logits` | `[B,L,V]` |
| `vlm_outputs.past_key_values` | VLM KV cache，seq len 为 `L` |
| `vlm_outputs.loss` | scalar，如果 `cotrain_vlm=True` 会加进总 loss |

如果 `cotrain_vlm=False`，VLM 参数冻结，并且 VLM forward 放在 `torch.no_grad()` 里。

### 5.3 未来轨迹变成连续 action

函数：

```python
_process_traj_future_training()
```

内部调用：

```python
self.action_space.traj_to_action(
    traj_history_xyz,
    traj_history_rot,
    traj_future_xyz,
    traj_future_rot,
)
```

文件：

```text
src/alpamayo1_5/action_space/unicycle_accel_curvature.py
```

输入：

| 变量 | 形状 |
|---|---|
| `traj_history_xyz` | `[B,G,16,3]` |
| `traj_history_rot` | `[B,G,16,3,3]` |
| `traj_future_xyz` | `[B,G,64,3]` |
| `traj_future_rot` | `[B,G,64,3,3]` |

输出：

| 变量 | 形状 | 含义 |
|---|---|---|
| `action` | `[B,G,64,2]` | normalized acceleration + curvature |
| reshape 后 | `[B*G,64,2]` | diffusion 训练样本 |

### 5.4 FlowMatching 构造训练噪声

文件：

```text
src/alpamayo1_5/diffusion/flow_matching.py
```

函数：

```python
FlowMatching.construct_training_data(action)
```

输入：

- `x = action`: `[B,64,2]`

输出：

| 字段 | 形状 | 含义 |
|---|---|---|
| `x` | `[B,64,2]` | 真实 action |
| `noise` | `[B,64,2]` | 标准高斯噪声 |
| `timesteps` | `[B,1,1]` | 训练时间 t |
| `noisy_x` | `[B,64,2]` | `t*x + (1-t)*noise` |

Flow Matching 的目标向量：

```python
target = x - noise
```

也就是让网络学会从噪声流向真实 action 的 vector field。

### 5.5 action_in_proj + expert + action_out_proj

代码：

```python
action_embeds = self.action_in_proj(noisy_x, timesteps)
kv_cache = vlm_outputs.past_key_values
kv_cache.crop(last_traj_future_start_idx)
expert_outputs = self.expert(
    inputs_embeds=action_embeds,
    position_ids=position_ids,
    past_key_values=kv_cache,
    attention_mask=None,
    use_cache=True,
)
diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1]:]
pred = self.action_out_proj(diffusion_out).view(-1,64,2)
```

形状：

| 变量 | 形状 |
|---|---|
| `action_embeds` | `[B,64,D]` |
| `position_ids` | `[3,B,64]` |
| `expert_outputs.last_hidden_state` | `[B,64,D]` |
| `pred` | `[B,64,2]` |

这里 `kv_cache.crop(last_traj_future_start_idx)` 的含义是：只保留 `<|traj_future_start|>` 前的 VLM 上下文作为 expert 条件，不让 GT future token 泄漏给 expert。

### 5.6 stage2 loss

代码：

```python
future_traj_loss = self.diffusion.compute_loss_from_pred(training_data, pred)
loss = future_traj_loss
if self.cotrain_vlm:
    loss += vlm_outputs.loss
```

`compute_loss_from_pred()`：

```python
target = x - noise
MSE(target, pred)
```

形状：

| 变量 | 形状 |
|---|---|
| `target` | `[B,64,2]` |
| `pred` | `[B,64,2]` |
| `future_traj_loss` | scalar |
| `loss` | scalar |

stage2 的本质是 diffusion/flow matching 的监督训练。

## 6. SFT 评估

入口：

```text
finetune/sft/evaluate_hf.py
```

流程：

1. 初始化分布式。
2. instantiate model/dataset/collate。
3. 用 `ReasoningVLA_Trainer.get_eval_dataloader()` 取验证集。
4. `metric_runner.run(model, data, output_batch)`。

默认 metric 配置在 `sft_base.yaml`：

```yaml
metric_runner:
  metrics:
    - ReasoningSampler
    - DistanceMetrics
```

`ReasoningSampler.evaluate()` 调用：

```python
model.sample_trajectories_from_data(...)
```

输出：

| 字段 | 形状 |
|---|---|
| `pred_xyz` | `[B,N_set,N_s,64,3]` |
| `pred_rot` | `[B,N_set,N_s,64,3,3]` |

`DistanceMetrics.evaluate()` 读取：

| 字段 | 形状 |
|---|---|
| `pred_xyz` | `[B,N_set,N_s,64,3]` |
| `ego_future_xyz[:, -1]` | `[B,64,3]` |

输出 per-sample 指标：

- `min_ade [B]`
- `ade [B]`
- `ade/by_t=3.0 [B]`
- corner distance 相关指标

## 7. RL 总览

RL 路径使用 Cosmos-RL + GRPO。

核心入口：

```text
finetune/rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py
```

配置：

```text
finetune/rl/toml/alpamayo_rvla_rl_local_test.toml
finetune/rl/hydra_configs/alpamayo1_5_rvla_rl_pai.yaml
```

重要说明：当前 RL README 明确说明，这条 RL 路径训练的是 VLM backbone 的离散生成路径，不训练 Alpamayo 1.5 的 continuous action expert diffusion head。reward 当前主要看解码出的离散轨迹 token 对应的轨迹。

## 8. RL 的三个角色

| 角色 | 文件/类 | 作用 |
|---|---|---|
| Entry/Spec | `alpamayo_cosmos_rl_post_training_entry.py` | 注册 model wrapper、data packer、rollout、trainer、reward |
| Rollout | `rollout.py::ReasoningVlaVllmRollout` | 用 vLLM 生成 completion |
| Policy trainer | `trainer.py::ReasoningVLAGRPOTrainer` | 用 GRPO 更新 policy |
| Data packer | `data_packer.py::RVLADataPacker` | 在 rollout/policy 两侧把数据整理成需要的形式 |
| Reward | `rewards/aggregated_reward.py::compute_reward` | 对 completion 打分 |
| Dataset adapter | `base_dataset.py::AlpamayoCosmosDataset` | 给 Cosmos 返回轻量 idx，真实样本由 packer 读取 |

## 9. RL 初始化数据流

入口：

```python
REASONING_VLA_SPEC.launch()
```

最终调用：

```python
finetune/rl/launcher.py::launch_alpamayo_model()
```

关键步骤：

1. 从 TOML 读取 `[policy].model_name_or_path`。
2. `rl.state.init_once(...)` 初始化全局状态。
3. 注册 Cosmos model。
4. `launch_worker(...)` 启动 policy/rollout worker。

`rl.state.init_once()` 会保存：

| 全局对象 | 来源 | 用途 |
|---|---|---|
| `ckpt_cfg` | `AutoConfig.from_pretrained(ckpt_path)` | 读取 token 数、traj vocab、VLM 路径 |
| `dataloaders` | Hydra instantiate RL data config | 训练/验证数据 |
| `tokenizer` | `build_processor(...).tokenizer` | rollout 文本编码/解码 |
| `traj_tokenizer` | `hydra.instantiate(ckpt_cfg.traj_tokenizer_cfg)` | 轨迹 token 解码 |
| `traj_fuser` | `_RolloutTrajectoryFusion` | rollout 前注入历史轨迹 token |

## 10. RL rollout 侧：从样本到 completion

### 10.1 Cosmos dataset 只返回 idx

文件：

```text
finetune/rl/base_dataset.py
```

`AlpamayoCosmosDataset.__getitem__(idx)` 返回：

```python
{"idx": str(idx), "split": "train"}
```

这样做是为了让真正的大样本加载发生在 data packer / prefetch 里，而不是 Cosmos dispatcher 直接携带大 tensor。

### 10.2 RVLADataPacker.get_rollout_input

文件：

```text
finetune/rl/models/reasoning_vla/data_packer.py
```

输入：

```python
item = {"idx": "123", "split": "train"}
```

输出：

```python
TokensPrompt = {
    "prompt_token_ids": list[int],
    "multi_modal_data": {"image": list[Tensor]},
    "hf_processor_mm_kwargs": {...},
    "mm_processor_kwargs": {...},
}
```

关键函数：

```python
_sample_to_rollout_prompt(sample)
```

输入样本字段：

| 字段 | 形状 |
|---|---|
| `tokenized_data.input_ids` | `[1,L]` 或 `[L]` |
| `ego_history_xyz` | `[G,16,3]`，内部会 unsqueeze 成 `[1,G,16,3]` |
| `ego_history_rot` | `[G,16,3,3]` |
| `image_frames` | RL reshape 后常为 `[N_img,1,3,H,W]` |

处理：

1. 如果没有 `input_ids`，用 tokenizer 编码 `tokenized_data["text"]`。
2. `traj_fuser.fuse_traj_tokens()` 注入历史轨迹。
3. 把连续重复的 `<|image_pad|>` 压缩成一个占位，交给 vLLM 的 multimodal processor 展开。
4. 将 `image_frames.flatten(0,1)` 变为图片列表。

输出形状：

| 字段 | 形状/类型 |
|---|---|
| `prompt_token_ids` | Python `list[int]`，长度约为 `L_prompt` |
| `multi_modal_data["image"]` | list，长度 `N_img`，每个 tensor `[3,H,W]` |

### 10.3 vLLM rollout_generation

文件：

```text
finetune/rl/models/reasoning_vla/rollout.py
```

函数：

```python
ReasoningVlaVllmRollout.rollout_generation()
```

输入：

| 变量 | 形状/类型 |
|---|---|
| `payloads` | list of `RLPayload` |
| 每个 payload.prompt | `{"idx": "...", "split": "train"}` |
| `prompts` | list of `TokensPrompt` |

vLLM sampling 配置来自 TOML：

```toml
[rollout]
n_generation = 12
batch_size = 2

[rollout.sampling_config]
temperature = 0.6
top_p = 0.98
max_new_tokens = 256
```

代码会设置：

```python
sampling_params.stop_token_ids = [traj_future_end_token_id]
sampling_params.logprobs = 1
sampling_params.skip_special_tokens = False
```

输出：

| 变量 | 形状/类型 |
|---|---|
| `response` | list[list[str]]，外层 prompt，内层 `n_generation` 个 completion |
| `completion_logprobs` | list[list[float]] |
| `RolloutResult.completions` | list[str] |

每条 completion 是一段文本，里面保留特殊 token，例如：

```text
<|cot_start|> ... <|cot_end|><|traj_future_start|><i123><i98>...<|traj_future_end|>
```

## 11. RL reward：completion 如何打分

入口：

```text
finetune/rl/rewards/aggregated_reward.py::compute_reward()
```

输入：

| 参数 | 类型/形状 |
|---|---|
| `to_be_evaluated` | 单条 completion string |
| `reference["ego_future_xyz"]` | `[G,64,3]` |
| `reference["ego_history_xyz"]` | `[G,16,3]` |
| `reference["ego_history_rot"]` | `[G,16,3,3]` |
| `tokenizer` | Alpamayo tokenizer |
| `traj_tokenizer` | trajectory tokenizer |

### 11.1 解码 rollout 轨迹

文件：

```text
finetune/rl/utils/trajectory_decode.py
```

函数：

```python
decode_rollout_trajectory()
```

处理：

1. 从 completion 中切出 `<|traj_future_start|>` 和 `<|traj_future_end|>` 之间的文本。
2. tokenizer 编码这段文本。
3. `extract_traj_tokens()` 提取离散轨迹 token id。
4. `traj_tokenizer.decode(...)` 解码为连续轨迹。

输出：

| 变量 | 形状 |
|---|---|
| `predicted_fut_xyz` | `[G,64,3]`，通常 `G=1` |
| `predicted_fut_rot` | `[G,64,3,3]` |

如果模型没有生成轨迹 token，代码会 fallback 到全 0 token 序列，避免 reward 崩溃。

### 11.2 ADE reward

文件：

```text
finetune/rl/rewards/traj_reward.py
```

函数：

```python
calculate_ade(pred_trajectory, gt_trajectory)
```

输入：

| 变量 | 形状 |
|---|---|
| `pred_trajectory` | `[64,3]` |
| `gt_trajectory` | `[64,3]` |

输出：

- scalar float，XY 平均 L2 距离。

### 11.3 Comfort reward

文件：

```text
finetune/rl/rewards/comfort_reward.py
```

函数：

```python
compute_comfort(pred_xyz, pred_rot)
```

输入需要扩维成：

| 变量 | 形状 |
|---|---|
| `pred_xyz` | `[B,N,K,T,3]` |
| `pred_rot` | `[B,N,K,T,3,3]` |

在当前 reward 中调用：

```python
compute_comfort(predicted_fut_xyz[:, None, None, ...],
                predicted_fut_rot[:, None, None, ...])
```

如果 `predicted_fut_xyz` 是 `[1,64,3]`，扩维后是 `[1,1,1,64,3]`。

输出：

- dict，每个 comfort 指标是 `[B,N]` 或聚合后可转 float 的 tensor。

### 11.4 aggregated reward

TOML：

```toml
[custom.alpamayo.reward]
traj_l2_weight = 0.4
comfort_weight = 0.1
```

公式：

```python
if l2_dist < 3.0:
    reward = -traj_l2_weight * (l2_dist / 3.0) + comfort_weight * (comfort_score - 1.0)
else:
    reward = -1.0
```

输出：

```python
(reward, {
    "traj_L2": float,
    "comfort_reward": float,
    "reward": float,
})
```

直觉：

- 轨迹越接近 GT，ADE 越小，reward 越高。
- 驾驶越舒适，comfort penalty 越小。
- ADE 超过 3m 直接给 -1，避免很差轨迹靠 comfort 拿到虚假高分。

## 12. RL policy 侧：从 completion 到训练 batch

### 12.1 RVLADataPacker.get_policy_input

文件：

```text
finetune/rl/models/reasoning_vla/data_packer.py
```

函数：

```python
get_policy_input(sample, rollout_output, n_ignore_prefix_tokens=0)
```

输入：

| 参数 | 类型 |
|---|---|
| `sample` | `{"idx": "...", "split": "train"}` |
| `rollout_output` | completion string |

处理：

1. 重新读取原始样本。
2. 取原 prompt 的 `tokenized_data.input_ids [1,L_prompt]`。
3. tokenizer 编码 completion，得到 `gen_ids [L_gen]`。
4. 拼接：

```python
new_ids_row = concat(ids_row, gen_ids, traj_future_end)
```

5. 构造 `labels_mask`：只让 completion 和最后的 end token 参与 logprob/loss。
6. 解码 rollout 轨迹，存入：
   - `ego_rollout_xyz`
   - `ego_rollout_rot`

输出样本字段：

| 字段 | 形状 |
|---|---|
| `tokenized_data.input_ids` | `[1,L_prompt+L_gen+1]` |
| `tokenized_data.labels_mask` | `[1,L_prompt+L_gen+1]` |
| `ego_rollout_xyz` | `[G,64,3]` |
| `ego_rollout_rot` | `[G,64,3,3]` |

### 12.2 policy_collate_fn

输入：

- `processed_samples`: list，长度是 mini-batch size。

输出：

| 字段 | 形状 |
|---|---|
| `tokenized_data.input_ids` | `[B_rl,L_rl]` |
| `input_ids` | `[B_rl,L_rl]`，提升到 top-level 给 Cosmos trainer |
| `labels_mask` | `[B_rl,L_rl]` |
| `logprob_masks` | `[B_rl,L_rl]`，通常等于 `labels_mask` |
| `attention_mask` | 如果存在则 `[B_rl,L_rl]` |
| `ego_history_xyz` | `[B_rl,G,16,3]` |
| `ego_future_xyz` | `[B_rl,G,64,3]` |

`B_rl` 是 policy mini-batch size，来自 Cosmos GRPO trainer 中的切分，不一定等于 rollout batch size。

## 13. RL 模型 forward

模型类：

```text
finetune/rl/models/reasoning_vla/base_model.py::RLWrapperReasoningVLA
```

Cosmos wrapper：

```text
finetune/rl/models/reasoning_vla/cosmos_wrapper.py::RVLACosmos.forward()
```

### 13.1 RVLACosmos.forward

作用：把 Cosmos trainer 传进来的 top-level batch 字段桥接到 `RLWrapperReasoningVLA.forward()`。

输入：

| 字段 | 形状 |
|---|---|
| `input_ids` | `[B_rl,L_rl]` |
| `labels_mask` | `[B_rl,L_rl]` |
| `tokenized_data` | dict |
| `ego_history_xyz` | `[B_rl,G,16,3]` |
| `ego_future_xyz` | `[B_rl,G,64,3]` |

输出：

- `ReasoningVLAOutput(loss, logits)`

### 13.2 RLWrapperReasoningVLA.forward

处理流程与 SFT stage1 类似：

1. 从 `tokenized_data` 取 `input_ids`。
2. `fuse_traj_tokens()` 注入历史轨迹 token。
3. 根据 `labels_mask` 构造 labels。
4. VLM forward 得到 logits。
5. 计算 future trajectory token loss 和 other token loss。

输出：

| 字段 | 形状 |
|---|---|
| `logits` | `[B_rl,L_rl,V]` |
| `loss` | scalar |

不过在 GRPO 里，真正用于 policy optimization 的不是这个 supervised CE loss，而是 Cosmos trainer 后续从 logits 里抽取 completion token 的 logprob，再计算 GRPO loss。

## 14. GRPO trainer：一次 policy update 怎么发生

文件：

```text
finetune/rl/models/reasoning_vla/trainer.py
```

函数：

```python
ReasoningVLAGRPOTrainer.step_training()
```

输入：

| 变量 | 类型/形状 |
|---|---|
| `rollouts` | list of completed rollout |
| `rollout.prompt` | `{"idx": "...", "split": "train"}` |
| `rollout.completion` | string |
| `rollout.advantage` | scalar |
| `rollout.reward` | scalar |

流程：

1. 提取 `payloads_list`、`completions_list`、`advantages_list`。
2. 对每条 rollout 调用 `data_packer.get_policy_input()`。
3. 按 `mini_batch` 切分。
4. `policy_collate_fn()` 得到 mini-batch。
5. model forward 得到 logits `[B_rl,L_rl,V]`。
6. `compute_logprobs()` 取 completion token 的 log probability。
7. `compute_loss()` 计算 GRPO loss。
8. `loss.backward()`。
9. all-reduce、optimizer step、checkpoint、logging。

### 14.1 advantages

Cosmos/RL controller 对同一个 prompt 的 `n_generation` 条 completion 做组内比较，得到 advantage。

在 trainer 中：

```python
advantages_t = torch.tensor(advantages_list).to(self.device)
minibatched_advantages = advantages_t[i:end].unsqueeze(1).expand(-1, computed_max_len)
current_advantages = logprob_masks * minibatched_advantages
```

形状：

| 变量 | 形状 |
|---|---|
| `advantages_t` | `[B_rl]` |
| `minibatched_advantages` | `[B_rl,L_rl_pad]` |
| `logprob_masks` | `[B_rl,L_rl_pad]` |
| `current_advantages` | `[B_rl,L_rl_pad]` |

只有 completion token 的 mask 为 True，因此只有 completion 会被 GRPO 更新。

### 14.2 logprob

模型输出：

```python
raw_logits = model_out.logits  # [B_rl,L_rl,V]
```

然后：

```python
current_per_token_logprobs, cu_seqlens, metrics = self.compute_logprobs(...)
```

输出：

| 变量 | 形状/类型 |
|---|---|
| `current_per_token_logprobs` | 通常是 flatten 后的有效 token logprob |
| `cu_seqlens` | packed sequence 边界 |
| `metrics` | entropy 等 |

### 14.3 GRPO loss

调用 Cosmos-RL：

```python
compute_loss(
    current_per_token_logprobs,
    old_per_token_logps,
    ref_per_token_logps,
    current_advantages,
    cu_seqlens,
    config,
    logprob_masks,
)
```

直觉公式：

```text
ratio = exp(log p_current - log p_old)
loss = -min(ratio * advantage, clipped_ratio * advantage) + KL penalty
```

TOML 中相关参数：

```toml
[train.train_policy]
epsilon_low = 0.2
epsilon_high = 0.28
kl_beta = 0.0
mu_iterations = 1
mini_batch = 1
temperature = 1.0
```

输出：

| 变量 | 含义 |
|---|---|
| `loss` | 用于 backward 的 scalar |
| `per_token_loss` | 日志用 |
| `kl_loss` | KL 惩罚 |

## 15. RL 配置里的 batch 数量关系

来自 `finetune/rl/toml/alpamayo_rvla_rl_local_test.toml`：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `rollout.batch_size` | 2 | rollout replica 一次给 vLLM 的 prompt 数 |
| `rollout.n_generation` | 12 | 每个 prompt 生成多少条 completion |
| `train.train_batch_per_replica` | 48 | 每个 policy replica 每步消费多少 rollout 样本 |
| `policy.parallelism.dp_shard_size` | 4 | policy FSDP shard 使用 GPU 数 |
| `rollout.parallelism.dp_shard_size` | 1 | rollout replica 使用 GPU 数 |
| `policy.model_max_length` | 4096 | policy 最大序列长度 |

一次 rollout batch 最大产生：

```text
2 prompts * 12 completions = 24 rollout samples
```

一次 policy step 需要：

```text
train_batch_per_replica = 48 rollout samples
```

所以大约两个 rollout batch 可以供一个 policy step 使用，实际还受到异步 buffer、过滤空 completion、weight sync 的影响。

## 16. RL 完整数据流总表

| 阶段 | 文件/函数 | 输入 | 输出 |
|---|---|---|---|
| 初始化 | `launcher.py::launch_alpamayo_model` | TOML config | dataloaders/tokenizer/traj_tokenizer 全局状态 |
| dataset stub | `AlpamayoCosmosDataset.__getitem__` | idx | `{"idx","split"}` |
| rollout pack | `RVLADataPacker.get_rollout_input` | idx stub | `TokensPrompt` |
| vLLM generate | `ReasoningVlaVllmRollout.rollout_generation` | `TokensPrompt` list | completion string list |
| reward reference | `AlpamayoCosmosDataset.get_reference_answer` | idx | GT history/future |
| reward decode | `decode_rollout_trajectory` | completion string | `pred_xyz [G,64,3]` |
| reward compute | `compute_reward` | pred + GT | scalar reward |
| controller | Cosmos-RL | same prompt group rewards | advantages |
| policy pack | `RVLADataPacker.get_policy_input` | idx + completion | `input_ids [1,L]`, `labels_mask [1,L]` |
| policy collate | `policy_collate_fn` | processed samples | batch `[B_rl,L]` |
| model forward | `RVLACosmos.forward` -> `RLWrapperReasoningVLA.forward` | batch | logits `[B_rl,L,V]` |
| logprob | `compute_logprobs` | logits + masks | token logprobs |
| GRPO loss | Cosmos `compute_loss` | logprobs + advantages | scalar loss |
| update | `loss.backward()` + optimizer | gradients | updated policy weights |
| sync | Cosmos controller | policy weights | rollout vLLM weights |

## 17. SFT 与 RL 的关键区别

| 维度 | SFT stage1 | SFT stage2 | RL |
|---|---|---|---|
| 训练对象 | VLM 离散轨迹 token 生成 | Alpamayo 1.5 action expert，可选共训 VLM | VLM backbone 离散生成路径 |
| 输入 | GT prompt + GT future token 占位替换 | GT prompt + GT future continuous trajectory | prompt + model sampled completion |
| 输出 | `logits [B,L,V]` | `pred [B,64,2]` vector field | `logits [B,L,V]` |
| loss | next-token CE | flow matching MSE | GRPO policy loss |
| 是否需要 reward | 否 | 否 | 是 |
| 是否在线采样 | 否 | 否 | 是，rollout 生成 completion |
| trajectory 表示 | 离散 token | continuous action `(64,2)` | 离散 token 解码为轨迹用于 reward |

## 18. 后续做科研改进时应注意

当前仓库非常适合做以下几类改进：

1. Reward 改进：在 `finetune/rl/rewards/aggregated_reward.py` 中加入更多驾驶质量项，比如碰撞、车道保持、道路边界、TTC、终点误差、reasoning consistency。
2. Reasoning reward：completion 中 `<|cot_start|>...<|cot_end|>` 可以解析出来，加入规则评分或 LLM judge。
3. GRPO 稳定性：调 `kl_beta`、reference reset、positive NLL、advantage normalization、completion 过滤。
4. 数据效率：改善 prefetch、避免重复样本、做 hard case mining。
5. Alpamayo 1.5 expert RL：这是更大的研究点，需要把 continuous action expert 的 logprob/trajectory reward 训练路径真正接进 RL，而不是只训练离散 VLM token 路径。

最重要的工程原则：先确保每个 batch 的字段和维度稳定，再改算法。这个项目里很多 bug 都会表现为 shape 对不上、special token 被跳过、completion 没有 `<|traj_future_end|>`、或者 reward 解码拿不到轨迹。
