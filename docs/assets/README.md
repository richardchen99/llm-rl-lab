# 图片来源与实验状态

截图均采自实际运行的 [LLM Reinforcement Learning Lab](https://richardchen99.github.io/llm-rl-lab/)，使用公开的数学验证案例，仅包含应用内容，也用于[配套研究笔记](https://richardchen99.github.io/blog/llm-rl-lab-note/)。截图日期：**2026-09-18**。

统一配置：**GRPO、随机种子 42、采样组大小 8、裁剪阈值 0.2、KL 系数 0.05、学习率 0.6；同一采样组进行六次更新**。

| 文件 | 截图状态 |
| --- | --- |
| [`overview.jpg`](overview.jpg) | 学习阶段与评估均已完成 |
| [`policy.jpg`](policy.jpg) | 优化后的候选回答概率 |
| [`optimizer.jpg`](optimizer.jpg) | 优化器参数、裁剪实验与 E0–E6 更新轨迹 |

初始期望奖励约为 0.378，六次更新后约为 0.466；最终策略相对参考策略的 KL 散度约为 0.0212。期望奖励在全部四个候选上精确计算，显示时进行舍入，不代表独立测试集得分。

框架图采用统一布局，区分旧采样策略与固定参考策略，并展示有限动作空间中的更新循环。具体范围见 [README](../../README.md#数学机制与实现范围)。

- 中文版：[高清 PNG](architecture.png) · [可编辑 SVG](architecture.svg)
- 英文版：[高清 PNG](architecture.en.png) · [可编辑 SVG](architecture.en.svg)

PNG 宽度为 3,200 像素。框架图由矢量图形与文字绘制，截图来自真实界面，均未使用文生图模型。图片不包含编辑器窗口、浏览器边框、本机路径或私人账户信息；数值仅对应所述实验，不作为性能基准。
