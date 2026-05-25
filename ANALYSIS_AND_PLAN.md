# Alpamayo 1.5 RL Reward 深度分析与改进方案

## 一、现状诊断：为什么CoT套模板 & Reward不增加

### 1.1 根本原因链

```
Regex关键词奖励 → 模板是最优策略 → 组内奖励方差坍塌 → advantage≈0 → 梯度信号消失 → 模型无法学习
```

详细分析:

1. **coc_reward.py 的关键词匹配本质**:
   - `score_factual_accuracy`: 检查"vehicle"/"lane"/"intersection"等关键词 → 模型学会堆砌关键词
   - `score_causal_coherence`: 检查"because"/"therefore"等连接词 → 模型学会模板句式
   - `score_safety_awareness`: 检查"safety"/"hazard"等安全词 → 每条输出都加"safe distance"
   - `score_completeness`: 检查6个维度覆盖率 → 模型学会每个维度用一个词覆盖

2. **GRPO组内方差坍塌**:
   - 同一prompt的N个生成(当前N=12)来自同一模型
   - 它们大概率都包含相似的关键词集合
   - 奖励函数对这些输出的打分几乎相同
   - advantage = reward - mean(rewards) ≈ 0
   - 梯度消失,模型无法区分好/坏的推理

3. **hcc_reward.py 的 _compute_grounded_fact_score 有改进但仍不够**:
   - 用GT轨迹验证CoC中的动作描述(好方向)
   - 但仍是regex匹配, "slow down" vs "no need to slow down" 无法区分
   - 没有利用障碍物检测数据验证CoC中提到的具体目标

### 1.2 RAA奖励的循环逻辑问题

RAA检查CoC文本 vs **预测轨迹**的一致性,但预测轨迹是模型自己生成的:
- 模型生成错误轨迹 + 匹配的错误CoC → 高RAA分数
- 这不是在鼓励正确行为,只是在鼓励一致的错误

### 1.3 obstacle.offline 数据完全未使用

PAI数据集包含 `obstacle.offline` 数据,含有:
- 障碍物的3D bounding box
- 障碍物类型(vehicle/pedestrian/cyclist)
- 障碍物速度和航向
- 这些是验证CoC是否"看到了正确目标"的关键真值

当前 `load_physical_aiavdataset` 完全没有加载这个feature。

## 二、对现有修改的评价

### 值得保留的:
1. ✅ HCC-RM的分层结构概念(4层递进)是合理的
2. ✅ Token-level advantage routing (CoC token用CoC奖励, traj token用traj奖励)
3. ✅ Grounded fact score用GT轨迹验证CoC动作描述
4. ✅ PGMO的Pareto加权思路

### 需要重大修改的:
1. ❌ 关键词匹配奖励 → 需要换成基于障碍物数据的语义验证
2. ❌ RAA奖励循环逻辑 → 需要改为CoC vs GT决策一致性
3. ❌ 缺乏奖励方差保护 → 需要加variance floor和rank-based advantage
4. ❌ 缺乏反模板机制 → 需要加diversity penalty

## 三、改进方案

### 改进1: 基于障碍物数据的Grounded CoC Reward
- 加载obstacle.offline数据到dataset
- 用障碍物位置/类型验证CoC中提到的"高威胁目标"是否正确
- 奖励正确识别真实障碍物,惩罚幻觉出不存在的障碍物

### 改进2: Decision-GT Consistency Reward (替代RAA)
- 从GT轨迹提取决策类别(stop/yield/nudge/proceed)
- 从CoC文本提取决策意图
- 只有两者一致才给奖励

### 改进3: 奖励方差保护
- 组内rank-based advantage替代raw normalization
- 最小方差阈值: 如果方差太低,跳过更新或使用perturbation
- 奖励去相关: 减少关键词奖励的权重,增加grounded奖励的权重

### 改进4: 反模板多样性奖励
- 检测CoC文本的n-gram重复率
- 对过于模板化的输出施加惩罚
- 用CoC文本的unique content ratio作为信号
