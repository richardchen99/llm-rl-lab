# LLM Reinforcement Learning Lab

An interactive research lab by **Richard Chen · 中国人民大学 / Renmin University of China**.

[Live lab](https://richardchen99.github.io/llm-rl-lab/) · [Personal research](https://richardchen99.github.io)

用一个能算清楚的策略说明 LLM RL：采样回答 → 获得奖励 → 计算优势 → clipped objective 与 KL → 改变回答概率 → 检查结果。

## Experiments

- **Training chain**：串联预训练、SFT、rollout、reward、policy update 和 evaluation。
- **Live training trace**：可复现的随机采样、重复回答、逐阶段显示的奖励 / 优势和完整 6 次策略更新。
- **PPO / GRPO comparison**：切换 value baseline 与组内相对优势，观察同一采样组如何产生不同学习信号。
- **Optimizer controls**：调整 group size、clip epsilon、KL 系数和学习率，重放实验。
- **Clipping explorer**：手动改变概率比与优势正负，观察 clipping 截断哪一侧的收益。
- **Learning trajectory**：每次更新后显示新策略概率、参考策略 KL、完整候选空间的期望奖励。

## Examples

| Example | Prompt | Reward |
| --- | --- | --- |
| Math verifier | `Solve 3x + 2 = 11. Give x.` | 预定义四个候选回答，正确答案 1，错误答案 0 |
| JSON verifier | `What is 17 + 25? Return only a JSON object with key "answer".` | 预定义候选中，仅合法 JSON 且 answer 为数值 42 得 1 |
| Preference reward | 向初学者简短解释 KV Cache | 公开给定的准确性与清晰度示例分，不是训练过的 reward model |

评分表对应有限候选回答，实验不调用 LLM，也没有任意文本评分器。

## Try this

1. 选择 Math verifier 与 GRPO，逐步执行。先有采样，再有 reward 与 advantage，之后才更新策略。
2. 到达 Evaluate 时，对比新旧回答概率和 E0–E6 的期望奖励轨迹。
3. 保持 seed、group size 不变，切换 PPO，对比 baseline 的作用。
4. 修改 KL 系数或学习率。所有影响训练的参数都会重置轨迹。
5. 点击 Resample group，观察有限采样噪声。若组内奖励完全相同，GRPO 组内优势为零。
6. 在 clipping explorer 中切换 Negative advantage，并将 ratio 降到下界以下。

## Mathematical scope

共享教学目标：

$$
J=\frac1G\sum_i\min\left(
\rho_i\hat A_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat A_i
\right)-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}}).
$$

$$
\rho_i=\frac{\pi_\theta(a_i)}{\pi_{\mathrm{old}}(a_i)},\qquad
\hat A_i^{\mathrm{GRPO}}=\frac{r_i-\bar r}{\sigma_r+10^{-8}}.
$$

这是 **四种完整回答上的单步 categorical softmax 策略**。采样、概率比、clipped surrogate、解析 KL、logit 梯度和参数更新均实际计算；reward 表公开给定。PPO 分支使用旧策略的精确期望奖励作单步 value baseline，GRPO 分支使用组内 population standard deviation。

真实 LLM 的 PPO / GRPO 涉及 token 序列、不同长度归一化与 KL 估计方式；PPO 通常另训 critic，并使用 GAE。本实验不模拟 critic 训练、分布式 rollout、token 级 credit assignment 或完整模型训练，不能用作这些算法实现的性能比较。

期望奖励按全部四个候选精确计算，**不是独立测试集得分**。有限采样与裁剪不保证每步单调改进。RLHF / RLVR 指反馈来源，PPO / GRPO 指优化方法；DPO 的典型偏好优化流程不需要同样的在线 rollout。

## Run locally

Use Node.js **24** (supported minimum: 22.12).

```bash
npm ci
npm run dev -- --host 127.0.0.1
npm test
npm run build
npm run preview -- --host 127.0.0.1
```

## Implementation

- `src/model.ts`：奖励任务、seeded sampling、baseline、clipped objective、KL、解析 logit 梯度与更新轨迹。
- `src/App.tsx`：阶段播放、奖励 / 优势、概率动画、学习轨迹与 clipping 实验。
- `src/shared.tsx` / `src/style.css`：Framer Motion、KaTeX、浅色玻璃 UI、键盘操作及 reduced-motion 支持。
- `tests/model.test.mjs`：可复现采样、优势统计、finite-difference 梯度核对、概率归一化和 clipping 方向。

计算完全在浏览器本地完成，无 API 密钥、模型下载或 GPU 要求。`npm test` 编译计算模型并运行 Node test runner；`npm run build` 进行类型检查与静态构建。

GitHub Actions 在 `main` 推送后使用 Node 24 执行测试、构建、Pages 部署。首次部署需将 Pages source 设为 GitHub Actions。

## Research series

[Transformer Architecture Lab](https://richardchen99.github.io/transformer-architecture-lab/) ·
[Position Encoding Lab](https://richardchen99.github.io/position-encoding-lab/) ·
[LLM Inference Lab](https://richardchen99.github.io/llm-inference-lab/) ·
[Tokenizer Playground](https://richardchen99.github.io/tokenizer-playground/)

## Sources

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
