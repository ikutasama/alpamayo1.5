# Alpamayo 1.5 模型结构与推理数据流

本文面向刚入门的同学，目标是把 Alpamayo 1.5 的模型结构和一次推理的数据流讲清楚：数据从哪里来，经过哪个文件里的哪个函数，张量形状如何变化，最后如何得到未来 6.4 秒的轨迹。

本文依据当前仓库 `ikutasama/alpamayo1.5` 中的代码，以及官方 Alpamayo 1.5 release、官方 finetune 代码和模型卡。你本地的核心代码入口主要在：

- `src/alpamayo1_5/models/base_model.py`
- `src/alpamayo1_5/models/alpamayo1_5.py`
- `src/alpamayo1_5/processor/qwen_processor.py`
- `src/alpamayo1_5/chat_template/conversation.py`
- `src/alpamayo1_5/action_space/unicycle_accel_curvature.py`
- `src/alpamayo1_5/diffusion/flow_matching.py`

## 1. 先建立整体直觉

Alpamayo 1.5 是一个视觉-语言-动作模型，也就是 VLA：Vision-Language-Action。

它做的事情可以用一句话概括：

> 输入多相机图像、历史自车轨迹和文字 prompt，先让 VLM 生成驾驶推理，再让 action expert 根据 VLM 的隐状态生成连续未来轨迹。

它有两条相关但不完全一样的轨迹输出路径：

1. 离散轨迹 token 路径：VLM 自回归生成 `<i0>` 到 `<iN>` 这样的离散 token，再由 trajectory tokenizer 解码为轨迹。这个路径主要用于 Alpamayo 1 形式的 SFT/RL。
2. Alpamayo 1.5 action expert 路径：VLM 只生成 CoC/推理直到 `<|traj_future_start|>`，然后 diffusion expert 在连续 action space 里采样，最后由 unicycle 模型解码为未来轨迹。当前 `src/alpamayo1_5/models/alpamayo1_5.py` 的正式推理使用这条路径。

本文重点讲 Alpamayo 1.5 action expert 路径。

## 2. 常用形状符号

后文用这些符号描述维度：

| 符号 | 含义 | 常见值 |
|---|---|---|
| `B` | batch size | 推理常为 1，训练可多卡累积 |
| `G` | trajectory group 数 | 当前推理要求 `G=1` |
| `T_h` | 历史轨迹点数 | 16 |
| `T_f` | 未来轨迹点数 | 64 |
| `C_cam` | 相机数 | 可变，常见 4 |
| `F_img` | 每个相机帧数 | 可变，配置里常见 4 |
| `N_img` | 输入图片总数 | `C_cam * F_img` |
| `H,W` | 原始图片高宽 | 数据集决定 |
| `P_img` | Qwen processor 展开后的视觉 token 数 | 由图片分辨率和 patch merge 决定 |
| `L` | 文本 token 序列长度，包含图像占位 token | 运行时变量 |
| `V` | tokenizer 词表大小 | 原始 VLM 词表 + 特殊 token + 轨迹 token |
| `D` | hidden size | 由 Qwen/Cosmos-Reason backbone 决定，代码运行时从 config 读取 |
| `N_s` | 每个输入采样的轨迹条数 | `num_traj_samples` |
| `N_set` | trajectory set 数 | `num_traj_sets` |
| `B*` | 扩展 batch | `B * N_s * N_set` |

注意：模型卡称 Alpamayo 1.5 是 10B 级模型；仓库默认 VLM backbone 配置名称可以是 `Qwen3-VL-8B-Instruct` 或服务器本地转换后的 Cosmos-Reason/Qwen 目录。精确 `D`、层数、head 数由 checkpoint 的 HF config 决定，代码不会硬编码。

## 3. 输入数据长什么样

数据由 `src/alpamayo1_5/load_physical_aiavdataset.py` 读取，再由 `src/alpamayo1_5/data/pai.py` 包装为 `PAIDataset`。

原始样本核心字段如下：

| 字段 | 形状 | 含义 |
|---|---|---|
| `image_frames` | 通常 `[C_cam, F_img, 3, H, W]` | 多相机、多帧 RGB 图像 |
| `camera_indices` | `[C_cam]` 或 reshape 后 `[N_img]` | 每路相机 id |
| `relative_timestamps` | 与图像对应 | 每张图相对当前时刻的时间 |
| `ego_history_xyz` | `[G, T_h, 3]`，collate 后 `[B, G, T_h, 3]` | 自车历史位置 |
| `ego_history_rot` | `[G, T_h, 3, 3]`，collate 后 `[B, G, T_h, 3, 3]` | 自车历史朝向 |
| `ego_future_xyz` | `[G, T_f, 3]`，训练/评估时使用 | GT 未来位置 |
| `ego_future_rot` | `[G, T_f, 3, 3]`，训练/评估时使用 | GT 未来朝向 |
| `tokenized_data` | dict | processor 生成的 VLM 输入 |

`PAIDataset.__getitem__()` 做两件重要事情：

1. 调用 `load_physical_aiavdataset(...)` 读取 PAI 数据。
2. 如果配置提供 `vla_preprocess_args`，调用 `QwenProcessor._preprocess_data()` 生成 `tokenized_data`。

## 4. Processor：把图像、文字、轨迹占位符变成 VLM 输入

入口：

- `src/alpamayo1_5/processor/qwen_processor.py`
- `QwenProcessor._preprocess_data()`
- `QwenProcessor.collate_fn()`

### 4.1 排序图像

函数：`sort_images_by_camera_ids()`

输入：

- `image_frames`: `[C_cam, F_img, 3, H, W]`
- `camera_indices`: `[C_cam]`
- `relative_timestamps`: 与相机/帧对应

输出：

- `sorted_image_frames`: `[C_cam, F_img, 3, H, W]`
- `sorted_camera_ids`: `[C_cam]`
- `sorted_ts`: 与排序后的图像对应

这一步保证不同样本的相机顺序稳定，否则模型看到的“第 1 组图像”语义会混乱。

### 4.2 构建对话模板

函数：`src/alpamayo1_5/chat_template/conversation.py::build_conversation()`

它会把数据组织成 Qwen chat template：

```text
system: You are a driving assistant...

user:
  image tokens
  <|traj_history_start|><|traj_history|>...<|traj_history|><|traj_history_end|>
  prompt text

assistant:
  <|cot_start|> ... <|cot_end|>
  <|traj_future_start|> ...
```

关键组件：

| 组件 | 由哪个函数生成 | 作用 |
|---|---|---|
| `image` | `construct_image()` | 插入多相机图像 |
| `traj_history` | `construct_traj_history()` | 插入历史轨迹占位 token |
| `prompt` | `construct_user_prompt()` | 告诉模型要输出什么 |
| `cot` | `construct_cot()` | 训练时可放 GT 推理，生成时只放起始标记 |
| `traj_future` | `construct_traj_future()` | 训练离散轨迹时放 future 占位，1.5 expert 推理时作为停止边界 |

### 4.3 图像进入 Qwen image processor

函数：`QwenProcessor._preprocess_data()`

输入图像：

- `images`: `[C_cam, F_img, 3, H, W]`

处理：

```python
images_for_vlm = images.float() / 255.0  # 如果原始是 uint8
image_inputs = processor.image_processor(images=images_for_vlm.flatten(0, 1), do_rescale=False)
```

形状变化：

| 变量 | 形状 |
|---|---|
| `images.flatten(0, 1)` | `[N_img, 3, H, W]` |
| `image_inputs["pixel_values"]` | Qwen image processor 输出，通常是 patch/feature 格式，具体形状由 transformers Qwen3-VL processor 决定 |
| `image_inputs["image_grid_thw"]` | `[N_img, 3]`，每张图的 temporal/height/width grid |

然后代码会根据 `image_grid_thw` 把文本中的每个 image placeholder 展开成多个 `<|image_pad|>` token。这样 VLM 的文本 token 长度 `L` 与视觉 patch 数对齐。

### 4.4 collate 后的 tokenized_data

函数：`QwenProcessor.collate_fn()`

对于一个 batch，输出：

| 字段 | 形状 |
|---|---|
| `tokenized_data["input_ids"]` | `[B, L]` |
| `tokenized_data["attention_mask"]` | `[B, L]` |
| `tokenized_data["pixel_values"]` | 拼接后的视觉输入，第一维约等于 batch 内所有图像/patch 的总量 |
| `tokenized_data["image_grid_thw"]` | `[sum_B N_img, 3]` |
| `labels_mask` | `[B, L]`，训练时标记哪些 token 参与 loss |

在推理时，`labels_mask` 不重要；在 SFT/RL 训练时，它决定哪些 token 算交叉熵。

## 5. 特殊 token 和轨迹 token

定义位置：`src/alpamayo1_5/models/base_model.py`

核心 token：

| token | 用途 |
|---|---|
| `<|traj_history_start|>` / `<|traj_history_end|>` | 包住历史轨迹 token |
| `<|traj_history|>` | 历史轨迹占位符，会被真实离散历史轨迹 token 替换 |
| `<|traj_future_start|>` / `<|traj_future_end|>` | 包住未来轨迹生成区域 |
| `<|traj_future|>` | 离散 future token 占位符，SFT 离散路径会替换 |
| `<|cot_start|>` / `<|cot_end|>` | 包住推理文本 |
| `<|image_pad|>` | Qwen 视觉 patch 占位 |
| `<i0>` 到 `<i{traj_vocab_size-1}>` | 离散轨迹词表 token |

配置类：`ReasoningVLAConfig`

默认重要字段：

```python
traj_vocab_size = 768
tokens_per_history_traj = 16
tokens_per_future_traj = 64
```

RL 配置里可能覆盖为：

```yaml
traj_vocab_size: 4000
tokens_per_history_traj: 48
tokens_per_future_traj: 128
```

所以写代码时不要假设 token 数固定，要读 `model.config`。

## 6. 模型初始化结构

入口：

- `src/alpamayo1_5/models/base_model.py::ReasoningVLA`
- `src/alpamayo1_5/models/alpamayo1_5.py::Alpamayo1_5`

### 6.1 ReasoningVLA 基类

`ReasoningVLA.__init__()` 初始化三类东西：

1. VLM backbone：`self.vlm`
2. trajectory tokenizer：`self.traj_tokenizer` / `self.hist_traj_tokenizer`
3. tokenizer 和特殊 token id：`self.tokenizer` / `self.special_token_ids`

VLM 初始化在：

```python
ReasoningVLA._initialize_qwenvl3_vlm()
```

形状概念：

| 模块 | 输入 | 输出 |
|---|---|---|
| `self.vlm` | `input_ids [B,L]`、`pixel_values`、`image_grid_thw` | `logits [B,L,V]`、`past_key_values`、hidden states/cache |
| `self.tokenizer` | 文本 | `input_ids [B,L]` |
| `self.traj_tokenizer` | 连续轨迹 | 离散 token `[B, tokens_per_traj]` |

### 6.2 Alpamayo1_5 子类

`Alpamayo1_5.__init__()` 在 VLM 外额外初始化 action expert：

```python
expert_config = copy.deepcopy(self.vlm.config.text_config)
self.expert = AutoModel.from_config(expert_config)
del self.expert.embed_tokens

self.action_space = instantiate(config.action_space_cfg)
self.diffusion = instantiate(config.diffusion_cfg, x_dims=self.action_space.get_action_space_dims())
self.action_in_proj = instantiate(..., in_dims=(64,2), out_dim=D)
self.action_out_proj = instantiate(..., in_features=D, out_features=2)
```

各模块作用：

| 模块 | 文件 | 作用 |
|---|---|---|
| `self.vlm` | `base_model.py` | 读图像和文字，自回归生成推理文本 |
| `self.expert` | `alpamayo1_5.py` | 只接收 action token embedding，不再自己嵌入 token id |
| `self.diffusion` | `diffusion/flow_matching.py` | 从噪声 action 逐步采样到真实 action |
| `self.action_in_proj` | `models/action_in_proj.py` | 把 noisy action + timestep 投影成 expert 的 token embedding |
| `self.action_out_proj` | `alpamayo1_5.py` | 把 expert hidden state 投影回 action vector field |
| `self.action_space` | `action_space/unicycle_accel_curvature.py` | 在连续 action `(accel, curvature)` 和轨迹 `(xyz, rot)` 间转换 |

## 7. 历史轨迹 token 如何注入 input_ids

函数：

- `TrajectoryFusionMixin.fuse_traj_tokens()`
- `tokenize_history_trajectory()`
- `replace_pad_token()`

输入：

| 变量 | 形状 |
|---|---|
| `input_ids` | `[B, L]`，里面有 `<|traj_history|>` 占位符 |
| `ego_history_xyz` | `[B, G, T_h, 3]` |
| `ego_history_rot` | `[B, G, T_h, 3, 3]` |

处理：

1. `ego_history_xyz.flatten(start_dim=0, end_dim=1)`：`[B*G, T_h, 3]`
2. `hist_traj_tokenizer.encode(...)`：`[B*G, tokens_per_history_traj]`
3. `einops.rearrange(..., "(b g) n -> b (g n)")`：`[B, G*tokens_per_history_traj]`
4. `replace_pad_token()` 把 `input_ids == traj_token_ids["history"]` 的占位 token 替换成真实历史轨迹 token。

输出：

- `input_ids`: `[B, L]`，形状不变，但其中的历史轨迹占位符变成了真实离散轨迹 token id。

这一步很关键：模型不是直接吃连续历史轨迹坐标，而是把历史轨迹离散成 token，塞进语言模型序列里。

## 8. 推理主入口：sample_trajectories_from_data_with_vlm_rollout

文件：`src/alpamayo1_5/models/alpamayo1_5.py`

函数：

```python
Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout()
```

输入 `data`：

| 字段 | 形状 |
|---|---|
| `data["ego_history_xyz"]` | `[B, G, T_h, 3]` |
| `data["ego_history_rot"]` | `[B, G, T_h, 3, 3]` |
| `data["tokenized_data"]["input_ids"]` | `[B, L]` |
| `data["tokenized_data"]["attention_mask"]` | `[B, L]` |
| `data["tokenized_data"]["pixel_values"]` | processor 输出 |
| `data["tokenized_data"]["image_grid_thw"]` | `[sum_B N_img, 3]` |

当前代码要求：

```python
B, n_traj_group, _, _ = ego_history_xyz.shape
assert n_traj_group == 1
```

也就是 `G=1`。

## 9. 推理第 1 步：VLM 自回归生成 CoC

代码段：

```python
input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
vlm_outputs = self.vlm.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    stopping_criteria=StopAfterEOS(eos_token_id=<|traj_future_start|>),
    logits_processor=ExpertLogitsProcessor(...),
    **tokenized_data,
)
```

关键配置：

| 参数 | 含义 |
|---|---|
| `num_return_sequences = num_traj_samples` | 每个输入生成多少条候选推理 |
| `max_new_tokens = max_generation_length` | 最多生成多少新 token |
| `eos_token_id = <|traj_future_start|>` | 生成到 future start 后停止 |
| `ExpertLogitsProcessor` | 屏蔽离散轨迹 token，避免 VLM 直接生成 `<i*>` |

形状变化：

| 变量 | 形状 |
|---|---|
| 输入 `input_ids` | `[B, L]` |
| `vlm_outputs.sequences` | `[B*, L + L_gen]` |
| `B*` | `B * num_traj_samples * num_traj_sets` |
| `vlm_outputs.past_key_values` | KV cache，batch 维是 `B*`，seq len 是生成后的 prefill 长度 |

这里的 `L_gen` 是 VLM 实际生成的推理 token 数，直到 `<|traj_future_start|>`。

## 10. 推理第 2 步：为 expert 构造位置和 attention mask

函数：

- `_find_eos_offset()`
- `_build_expert_pos_ids_and_attn_mask()`

### 10.1 找到 expert 接入位置

`_find_eos_offset()` 找到每条序列里第一个 `<|traj_future_start|>` 的位置，返回 `offset = pos + 1`。

输入：

- `sequences`: `[B*, L_total]`

输出：

- `offset`: `[B*]`

含义：expert diffusion token 应该接在 `<|traj_future_start|>` 之后。

### 10.2 构造 Qwen RoPE position_ids

`_build_expert_pos_ids_and_attn_mask()` 输出：

| 变量 | 形状 | 含义 |
|---|---|---|
| `position_ids` | `[3, B*, T_f]` | Qwen VL 的三分量 RoPE position ids |
| `attention_mask` | `[B*, 1, T_f, KV + T_f]` | expert token 对 VLM cache 和自身 token 的注意力 mask |

其中 `T_f = self.action_space.get_action_space_dims()[0]`，当前 unicycle action space 是 64。

attention mask 会屏蔽：

1. prompt padding 区域。
2. `<|traj_future_start|>` 之后到 diffusion token 之前的 gap。

这样 expert 看到的是：图像、历史轨迹、生成推理文本，以及 `<|traj_future_start|>` 前后的有效上下文。

## 11. 推理第 3 步：Flow Matching 在 action space 里采样

文件：`src/alpamayo1_5/diffusion/flow_matching.py`

函数：

```python
self.diffusion.sample(batch_size=total_batch, step_fn=step_fn, ...)
```

action space 维度来自：

```python
self.action_space.get_action_space_dims()  # (64, 2)
```

所以 diffusion 采样的变量是：

```text
x: [B*, 64, 2]
```

这里最后一维 2 表示：

1. normalized acceleration
2. normalized curvature

Flow Matching 采样流程：

1. 初始化噪声：`x = torch.randn(B*, 64, 2)`
2. 构造时间：`time_steps = linspace(0, 1, num_inference_steps+1)`
3. 每一步调用 `step_fn(x, t)` 得到 vector field `v`
4. Euler 更新：`x = x + dt * v`

形状：

| 变量 | 形状 |
|---|---|
| `x` | `[B*, 64, 2]` |
| `t` | `[B*, 1, 1]`，可 broadcast |
| `v = step_fn(x,t)` | `[B*, 64, 2]` |
| `sampled_action` | `[B*, 64, 2]` |

## 12. step_fn：action expert 如何预测 vector field

`sample_trajectories_from_data_with_vlm_rollout()` 内部定义了 `step_fn()`。

### 12.1 action_in_proj

文件：`src/alpamayo1_5/models/action_in_proj.py`

类：`PerWaypointActionInProjV2`

输入：

| 变量 | 形状 |
|---|---|
| `x` | `[B*, 64, 2]` |
| `t` | `[B*, 1, 1]` |

处理：

1. 对 acceleration 和 curvature 分别做 Fourier encoding。
2. 对 timestep 做 Fourier encoding。
3. 拼接后过 MLP。
4. 输出 LayerNorm 后的 expert token embedding。

输出：

| 变量 | 形状 |
|---|---|
| `future_token_embeds` | `[B*, 64, D]` |

### 12.2 expert transformer

代码：

```python
expert_out_base = self.expert(
    inputs_embeds=future_token_embeds,
    position_ids=position_ids,
    past_key_values=prompt_cache,
    attention_mask=attention_mask,
    use_cache=True,
    is_causal=False,  # 如果 config.expert_non_causal_attention=True
)
```

输入：

| 变量 | 形状 |
|---|---|
| `inputs_embeds` | `[B*, 64, D]` |
| `position_ids` | `[3, B*, 64]` |
| `past_key_values` | VLM 生成后的 cache |
| `attention_mask` | `[B*, 1, 64, KV+64]` |

输出：

| 变量 | 形状 |
|---|---|
| `expert_out_base.last_hidden_state` | `[B*, 64, D]` |

注意：expert 删除了自己的 `embed_tokens`，所以它不是用 token id 作为输入，而是直接吃 `action_in_proj` 产生的 embedding。

### 12.3 action_out_proj

代码：

```python
pred = self.action_out_proj(last_hidden).view(-1, 64, 2)
```

形状：

| 变量 | 形状 |
|---|---|
| `last_hidden` | `[B*, 64, D]` |
| `action_out_proj(last_hidden)` | `[B*, 64, 2]` |
| `pred` | `[B*, 64, 2]` |

`pred` 是 flow matching 的 vector field，不是最终坐标。

## 13. 推理第 4 步：action_to_traj 解码成轨迹

文件：`src/alpamayo1_5/action_space/unicycle_accel_curvature.py`

函数：

```python
UnicycleAccelCurvatureActionSpace.action_to_traj()
```

输入：

| 变量 | 形状 |
|---|---|
| `sampled_action` | `[B*, 64, 2]` |
| `hist_xyz_rep` | `[B*, T_h, 3]` |
| `hist_rot_rep` | `[B*, T_h, 3, 3]` |

处理逻辑：

1. 将 normalized acceleration/curvature 反归一化。
2. 从历史轨迹估计当前速度 `v0`。
3. 用 unicycle kinematics 积分：
   - acceleration -> velocity
   - curvature + velocity -> yaw
   - velocity + yaw -> x/y
4. 生成 z 和旋转矩阵。

输出：

| 变量 | 形状 |
|---|---|
| `pred_xyz` | `[B*, 64, 3]` |
| `pred_rot` | `[B*, 64, 3, 3]` |

最后 reshape：

```python
pred_xyz = rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=N_set, nj=N_s)
pred_rot = rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...", ns=N_set, nj=N_s)
```

最终输出：

| 变量 | 形状 |
|---|---|
| `pred_xyz` | `[B, N_set, N_s, 64, 3]` |
| `pred_rot` | `[B, N_set, N_s, 64, 3, 3]` |

## 14. 一次完整推理的数据流总表

| 步骤 | 文件/函数 | 输入形状 | 输出形状 |
|---|---|---|---|
| 读取 PAI | `load_physical_aiavdataset.py` | clip id | `ego_history_xyz [1,1,16,3]`, `ego_future_xyz [1,1,64,3]`, images |
| Dataset 包装 | `data/pai.py::PAIDataset.__getitem__` | 原始样本 | 去掉第一维后的样本，batch 后再变 `[B,...]` |
| 图文预处理 | `processor/qwen_processor.py::_preprocess_data` | images `[C,F,3,H,W]` | `text`, `pixel_values`, `image_grid_thw` |
| batch tokenization | `QwenProcessor.collate_fn` | list of samples | `input_ids [B,L]`, `attention_mask [B,L]` |
| 历史轨迹注入 | `TrajectoryFusionMixin.fuse_traj_tokens` | `input_ids [B,L]`, hist traj | `input_ids [B,L]`，内容替换 |
| VLM 生成推理 | `self.vlm.generate` | `[B,L]` + images | `sequences [B*,L+L_gen]`, KV cache |
| 找 future start | `_find_eos_offset` | `sequences [B*,L_total]` | `offset [B*]` |
| expert position/mask | `_build_expert_pos_ids_and_attn_mask` | offsets + cache len | `position_ids [3,B*,64]`, mask `[B*,1,64,KV+64]` |
| diffusion sample | `FlowMatching.sample` | noise `[B*,64,2]` | action `[B*,64,2]` |
| action embedding | `PerWaypointActionInProjV2.forward` | `[B*,64,2]`, `t` | `[B*,64,D]` |
| expert transformer | `self.expert(...)` | action embeds + VLM cache | hidden `[B*,64,D]` |
| vector field | `action_out_proj` | hidden `[B*,64,D]` | pred `[B*,64,2]` |
| action 解码 | `action_to_traj` | action `[B*,64,2]` + history | xyz `[B*,64,3]`, rot `[B*,64,3,3]` |
| 输出 reshape | `einops.rearrange` | `[B*,...]` | `[B,N_set,N_s,64,...]` |

## 15. CFG navigation 版本

文件：`src/alpamayo1_5/models/alpamayo1_5.py`

函数：

```python
sample_trajectories_from_data_with_vlm_rollout_cfg_nav()
```

它与普通推理的主要区别：

1. 先用带 navigation 的 prompt 生成 guided VLM cache。
2. 再去掉 `<|route_start|>...<|route_end|>` 导航文本，构造 unguided VLM cache。
3. diffusion 采样时同时调用：
   - `step_fn`：guided vector field
   - `unguided_step_fn`：unguided vector field
4. 在 `FlowMatching._guided_v()` 中组合：

```python
guided_v = (1 - w) * unguided_v + w * guided_v
```

输出形状仍然是：

- `pred_xyz [B, N_set, N_s, 64, 3]`
- `pred_rot [B, N_set, N_s, 64, 3, 3]`

## 16. 你读代码时最该抓住的主线

如果只记一条主线，就是：

```text
PAI sample
  -> QwenProcessor: images/text/trajectory placeholders
  -> input_ids + pixel_values
  -> fuse_traj_tokens: 把历史轨迹塞进 token 序列
  -> VLM generate: 生成 CoC 到 <|traj_future_start|>
  -> expert 使用 VLM KV cache 作为条件
  -> FlowMatching 在 [64,2] action space 采样
  -> Unicycle action_to_traj
  -> pred_xyz [B,N_set,N_s,64,3], pred_rot [B,N_set,N_s,64,3,3]
```

从研究角度看，Alpamayo 1.5 的关键设计是：VLM 负责“看懂场景并生成可解释推理”，action expert 负责“把推理上下文变成连续、平滑、动力学可解释的驾驶轨迹”。这也是后续做强化学习改进时可以下手的两个方向：奖励推理质量，或者奖励轨迹质量与舒适性；当前仓库的 RL 路径主要训练 VLM 离散生成路径，action expert 路径的 RL 还没有作为默认训练目标接入。
