import { useState } from "react";
import { motion } from "framer-motion";
import {
  Shell,
  Panel,
  Formula,
  Tabs,
  Range,
  Stat,
  Bars,
  Note,
  Transport,
  usePlayback,
} from "./shared";
import {
  tasks,
  reference,
  experiment,
  kl,
  surrogate,
  type Task,
  type Algorithm,
} from "./model";
const labels = [
  "Policy snapshot",
  "Rollout · sample a group",
  "Reward · inspect the evidence",
  "Advantage · establish a baseline",
  "Objective · clip and regularize",
  ...Array.from({ length: 6 }, (_, i) => `Update · epoch ${i + 1}`),
  "Evaluate · new policy",
];
export default function App() {
  const [task, setTask] = useState<Task>("math"),
    [algorithm, setAlgorithm] = useState<Algorithm>("grpo"),
    [seed, setSeed] = useState(42),
    [size, setSize] = useState(8),
    [epsilon, setEpsilon] = useState(0.2),
    [beta, setBeta] = useState(0.05),
    [lr, setLr] = useState(0.6),
    [ratio, setRatio] = useState(1.4),
    [sign, setSign] = useState(1);
  const spec = tasks[task],
    exp = experiment(task, algorithm, size, seed, epsilon, beta, lr),
    play = usePlayback(
      labels.length - 1,
      `${task}|${algorithm}|${seed}|${size}|${epsilon}|${beta}|${lr}`,
      1600,
    );
  const epoch = Math.max(0, Math.min(6, play.step - 4)),
    prob = exp.history[epoch],
    expected = prob.reduce((s, p, i) => s + p * spec.rewards[i], 0),
    objective =
      surrogate(prob, reference, exp.indices, exp.adv, epsilon) -
      beta * kl(prob, reference);
  const mean = exp.rewards.reduce((a, b) => a + b, 0) / size,
    sd = Math.sqrt(exp.rewards.reduce((s, r) => s + (r - mean) ** 2, 0) / size);
  const clippedRatio = Math.max(1 - epsilon, Math.min(1 + epsilon, ratio)),
    clipped = Math.min(ratio * sign, clippedRatio * sign);
  return (
    <Shell
      slug="llm-rl-lab"
      title="LLM Reinforcement Learning Lab"
      subtitle="模型怎样从反馈中改变回答倾向？用一个可以算清楚的小策略，观察采样、奖励、优势与策略更新，而不把训练过程藏在一条进度条后面。"
      sources={[
        ["InstructGPT · RLHF", "https://arxiv.org/abs/2203.02155"],
        ["Proximal Policy Optimization", "https://arxiv.org/abs/1707.06347"],
        ["DeepSeekMath · GRPO", "https://arxiv.org/abs/2402.03300"],
        ["Direct Preference Optimization", "https://arxiv.org/abs/2305.18290"],
      ]}
    >
      <Panel
        title="From a useful answer to a learning signal"
        eyebrow="01 / THE TRAINING CHAIN"
      >
        <div className="stageRail">
          {[
            "Pretraining",
            "SFT",
            "Rollouts",
            "Reward",
            "Policy update",
            "Evaluation",
          ].map((s, i) => (
            <div
              key={s}
              className={
                i ===
                (play.step < 1
                  ? 1
                  : play.step < 3
                    ? play.step + 1
                    : play.step < play.total
                      ? 4
                      : 5)
                  ? "active"
                  : ""
              }
            >
              <small>0{i + 1}</small>
              {s}
            </div>
          ))}
        </div>
        <div className="grid three">
          <div className="prose">
            <h3>Learn to answer</h3>
            <p>
              预训练学语言与知识，SFT 学示范回答。RL
              从一个可用策略出发，采样回答，再用结果反馈改变回答概率。
            </p>
          </div>
          <div className="prose">
            <h3>Define a reward</h3>
            <p>
              RLHF 可从人类偏好训练奖励模型；RLVR
              可用数学答案或测试程序验证结果。奖励的定义决定了优化方向。
            </p>
          </div>
          <div className="prose">
            <h3>Update with restraint</h3>
            <p>
              PPO / GRPO 利用优势信号更新策略，用 clipping
              约束局部变化，也常用参考策略的 KL
              惩罚抑制漂移。它们不能替代独立评测。
            </p>
          </div>
        </div>
      </Panel>
      <div className="grid">
        <Panel
          title="One prompt. A group of attempts."
          eyebrow="02 / LIVE TRAINING TRACE"
        >
          <Tabs
            options={[
              { id: "math", label: "Math verifier" },
              { id: "json", label: "JSON verifier" },
              { id: "preference", label: "Preference reward" },
            ]}
            value={task}
            onChange={(v) => setTask(v as Task)}
            label="Reward example"
          />
          <div className="prompt">
            <small>USER PROMPT</small>
            <p>{spec.prompt}</p>
          </div>
          <div className="row" style={{ margin: "18px 0" }}>
            <h3>{labels[play.step]}</h3>
            <span className="badge">
              {play.playing
                ? "TRAINING TRACE"
                : play.step === play.total
                  ? "EVALUATED"
                  : "STEP MODE"}
            </span>
          </div>
          <div className="rollouts">
            {play.step === 0 ? (
              <Note>
                先记录旧策略。下一步将按这个概率分布采样一组回答；重复回答也保留，因为它们确实被采样到了。
              </Note>
            ) : (
              exp.indices.map((index, i) => (
                <motion.div
                  className="rollout"
                  key={`${seed}-${i}-${index}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.035 }}
                >
                  <small>R{String(i + 1).padStart(2, "0")}</small>
                  <p>{spec.candidates[index]}</p>
                  <div>
                    <span>
                      Reward{" "}
                      <b>{play.step >= 2 ? exp.rewards[i].toFixed(2) : "—"}</b>
                    </span>
                    <span>
                      Advantage{" "}
                      <b className={exp.adv[i] < 0 ? "minus" : "plus"}>
                        {play.step >= 3 ? exp.adv[i].toFixed(3) : "—"}
                      </b>
                    </span>
                  </div>
                </motion.div>
              ))
            )}
          </div>
          <Transport {...play} labels={labels} />
          <div className="buttonRow">
            <button onClick={() => setSeed((s) => s + 1)}>
              Resample group
            </button>
            <span className="caption">
              Seed {seed} · group size {size}
            </span>
          </div>
          <Note>{spec.rule}</Note>
        </Panel>
        <Panel
          title="Watch the policy change"
          eyebrow="03 / PROBABILITY, NOT A SCRIPTED ANSWER"
        >
          <Tabs
            options={[
              { id: "ppo", label: "PPO · value baseline" },
              { id: "grpo", label: "GRPO · group baseline" },
            ]}
            value={algorithm}
            onChange={(v) => setAlgorithm(v as Algorithm)}
            label="Policy optimizer"
          />
          <div className="stack">
            {spec.candidates.map((answer, i) => (
              <div key={i}>
                <p className="candidateText">
                  <b>C{i + 1}</b> {answer}
                </p>
                <Bars
                  items={[
                    {
                      label: "Old policy",
                      value: reference[i],
                      display: `${(reference[i] * 100).toFixed(1)}%`,
                    },
                    {
                      label: "New policy",
                      value: prob[i],
                      display: `${(prob[i] * 100).toFixed(1)}%`,
                      active: true,
                    },
                  ]}
                />
              </div>
            ))}
          </div>
          <div className="stats">
            <Stat
              label="EXPECTED REWARD"
              value={expected.toFixed(3)}
              detail={`Before ${exp.baseline.toFixed(3)}`}
            />
            <Stat
              label="KL TO REFERENCE"
              value={kl(prob, reference).toFixed(4)}
            />
            <Stat label="UPDATE EPOCH" value={`${epoch} / 6`} />
          </div>
          <Formula
            tex={
              algorithm === "grpo"
                ? String.raw`\hat A_i=\frac{r_i-\bar r}{\sigma_r+\delta}`
                : String.raw`\hat A_i=r_i-V_{\mathrm{old}}(s)`
            }
          />
          <Note>
            {algorithm === "grpo"
              ? "GRPO 通过同一 prompt 的组内奖励构造相对优势，不需要另训一个 value critic。此处采用 population standard deviation。"
              : "真实 PPO 常训练 value critic，并结合 token 级回报与 GAE。这里是单步回答选择问题，使用旧策略的精确期望奖励作基线，隔离策略更新本身。"}
          </Note>
          {play.step >= 3 && sd < 1e-8 && algorithm === "grpo" && (
            <p className="note">
              本组奖励完全相同，优势为零，没有组内学习信号；可以重新采样。
            </p>
          )}
        </Panel>
      </div>
      <Panel title="Open the optimizer" eyebrow="04 / CLIPPING & KL">
        <Formula
          tex={String.raw`J=\frac1G\sum_i\min\!\left(\rho_i\hat A_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat A_i\right)-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})`}
        />
        <div className="grid equal">
          <div className="controls">
            <Range
              label="Group size"
              min={4}
              max={16}
              value={size}
              onChange={setSize}
            />
            <Range
              label="Clip epsilon"
              min={0.05}
              max={0.4}
              step={0.05}
              value={epsilon}
              onChange={(v) => setEpsilon(+v.toFixed(2))}
            />
            <Range
              label="KL coefficient"
              min={0}
              max={1}
              step={0.05}
              value={beta}
              onChange={(v) => setBeta(+v.toFixed(2))}
            />
            <Range
              label="Learning rate"
              min={0.1}
              max={1}
              step={0.1}
              value={lr}
              onChange={(v) => setLr(+v.toFixed(1))}
            />
          </div>
          <div className="prose">
            <p>
              概率比是新策略相对 <b>rollout 时的旧策略</b> 的变化。参考策略则是
              KL 的锚点，两者的角色不同。
            </p>
            <p>
              本实验对四种完整回答定义 softmax
              策略，真实计算采样与梯度，使用解析 KL 和共享的 clipped surrogate
              做 6 次优化。它是单步、序列级教学模型，不是完整 token 级 PPO /
              GRPO 训练器。
            </p>
            <p>修改参数会重置演示，防止把不同实验条件混在一条轨迹里。</p>
          </div>
        </div>
        <div className="stats">
          <Stat
            label="SURROGATE − βKL"
            value={play.step >= 4 ? objective.toFixed(4) : "—"}
          />
          <Stat
            label="GROUP MEAN"
            value={play.step >= 2 ? mean.toFixed(3) : "—"}
          />
          <Stat
            label="GROUP STD"
            value={play.step >= 3 ? sd.toFixed(3) : "—"}
          />
        </div>
        <div className="callout">
          <h3>Learning trajectory</h3>
          <svg
            className="diagram"
            viewBox="0 0 720 190"
            role="img"
            aria-label="Expected reward over six policy updates"
          >
            <path d="M46 15 V155 H690" fill="none" stroke="#1d273330" />
            {[0, 0.5, 1].map((v) => (
              <text key={v} x="10" y={159 - v * 130} className="subtext">
                {v.toFixed(1)}
              </text>
            ))}
            <motion.path
              initial={false}
              animate={{
                d: exp.history
                  .slice(0, epoch + 1)
                  .map(
                    (p, i) =>
                      `${i ? "L" : "M"} ${60 + i * 100} ${155 - p.reduce((s, w, j) => s + w * spec.rewards[j], 0) * 130}`,
                  )
                  .join(" "),
              }}
              fill="none"
              stroke="#b8934f"
              strokeWidth="2.5"
            />
            {exp.history.map((p, i) => (
              <g key={i}>
                <motion.circle
                  initial={false}
                  animate={{
                    opacity: i <= epoch ? 1 : 0,
                    cy:
                      155 -
                      p.reduce((s, w, j) => s + w * spec.rewards[j], 0) * 130,
                  }}
                  cx={60 + i * 100}
                  r={i === epoch ? 5 : 3}
                  fill={i === epoch ? "#b8934f" : "#6fa8dc"}
                />
                <text
                  x={60 + i * 100}
                  y="177"
                  textAnchor="middle"
                  className="subtext"
                >
                  E{i}
                </text>
              </g>
            ))}
            <text x="46" y="12" className="subtext">
              Expected reward · all four candidate answers
            </text>
          </svg>
          <Note>
            每次更新都重新计算完整候选空间的期望奖励。有限采样、clipping 和 KL
            共同影响轨迹，因此奖励不保证每步单调上升。
          </Note>
        </div>
        <div className="callout">
          <h3>Try a single likelihood ratio</h3>
          <div className="grid equal">
            <div className="stack">
              <Range
                label="Likelihood ratio"
                min={0.2}
                max={2}
                step={0.05}
                value={ratio}
                onChange={(v) => setRatio(+v.toFixed(2))}
              />
              <Tabs
                options={[
                  { id: "1", label: "Positive advantage" },
                  { id: "-1", label: "Negative advantage" },
                ]}
                value={String(sign)}
                onChange={(v) => setSign(+v)}
                label="Advantage sign"
              />
            </div>
            <Bars
              signed
              items={[
                { label: "Unclipped", value: ratio * sign },
                { label: "Clipped surrogate", value: clipped, active: true },
              ]}
            />
          </div>
          <Note>
            Clipping
            只限制某些方向继续获得收益，不是把所有梯度硬裁剪，也不能保证新策略一定离旧策略很近。优势为负时，过度降低该回答概率的一侧会被截住。
          </Note>
        </div>
      </Panel>
      <Panel
        title="What the reward cannot tell you"
        eyebrow="05 / FROM THIS DEMO TO AN LLM"
      >
        <div className="grid three">
          <div className="prose">
            <h3>Credit assignment</h3>
            <p>
              真实回答是一串
              token，序列奖励需要分配到多步决策。正确答案不代表每个推理步骤都正确；答案奖励和过程奖励提供不同的监督。
            </p>
          </div>
          <div className="prose">
            <h3>Reward hacking</h3>
            <p>
              如果奖励只看关键词，模型可能学会堆关键词；如果只看最终值，模型可能偶然猜对。应检查评分器漏洞、分布外样本与独立保留集。
            </p>
          </div>
          <div className="prose">
            <h3>RLHF, RLVR, DPO</h3>
            <p>
              RLHF / RLVR 描述反馈来源，PPO / GRPO 是优化方法。DPO
              直接用偏好对构造损失，通常不需要同样的在线 rollout 与显式 reward
              model 流程。
            </p>
          </div>
        </div>
      </Panel>
    </Shell>
  );
}
