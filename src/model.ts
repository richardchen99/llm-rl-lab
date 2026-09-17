export const tasks = {
  math: {
    prompt: "Solve 3x + 2 = 11. Give x.",
    rule: "答案可用方程代入核验；正确得 1，错误得 0。",
    candidates: ["x = 3", "x = 4", "3x = 9, so x = 3", "x = 9"],
    rewards: [1, 0, 1, 0],
  },
  json: {
    prompt: 'What is 17 + 25? Return only a JSON object with key "answer".',
    rule: "JSON 可解析且 answer 为 42 得 1；格式或数值不符得 0。",
    candidates: [
      '{"answer":42}',
      "The answer is 42.",
      '{"answer":43}',
      '{"answer":"forty-two"}',
    ],
    rewards: [1, 0, 0, 0],
  },
  preference: {
    prompt: "Explain KV Cache to a first-year student in one short paragraph.",
    rule: "教学偏好分：准确、简洁且适合初学者。此处为公开给定的示例评分，不是训练过的 reward model。",
    candidates: [
      "复用历史 K/V，新 Query 仍读取历史缓存。",
      "缓存所有未来答案，生成时直接查找。",
      "KV Cache 很快。",
      "保存前缀各层的键和值，避免每步重复投影旧 token。",
    ],
    rewards: [0.85, 0, 0.25, 1],
  },
};
export type Task = keyof typeof tasks;
export type Algorithm = "ppo" | "grpo";
export const softmax = (x: number[]) => {
  const m = Math.max(...x),
    e = x.map((v) => Math.exp(v - m)),
    s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};
export const initialLogits = [-0.3, 0.1, -0.4, 0.2];
export const reference = softmax(initialLogits);
export function rng(seed: number) {
  let n = seed >>> 0;
  return () => {
    n = (1664525 * n + 1013904223) >>> 0;
    return n / 4294967296;
  };
}
export function sample(prob: number[], size: number, seed: number) {
  const random = rng(seed);
  return Array.from({ length: size }, () => {
    const x = random();
    let sum = 0;
    for (let i = 0; i < prob.length; i++) {
      sum += prob[i];
      if (x < sum) return i;
    }
    return prob.length - 1;
  });
}
export function advantages(
  rewards: number[],
  algorithm: Algorithm,
  baseline: number,
) {
  const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length,
    sd = Math.sqrt(
      rewards.reduce((s, v) => s + (v - mean) ** 2, 0) / rewards.length,
    );
  return rewards.map((r) =>
    algorithm === "grpo" ? (r - mean) / (sd + 1e-8) : r - baseline,
  );
}
export function kl(prob: number[], ref: number[]) {
  return prob.reduce((s, p, i) => s + p * Math.log(p / ref[i]), 0);
}
export function surrogate(
  prob: number[],
  old: number[],
  indices: number[],
  adv: number[],
  epsilon: number,
) {
  return (
    indices.reduce((s, index, j) => {
      const ratio = prob[index] / old[index];
      return (
        s +
        Math.min(
          ratio * adv[j],
          Math.max(1 - epsilon, Math.min(1 + epsilon, ratio)) * adv[j],
        )
      );
    }, 0) / indices.length
  );
}
export function update(
  logits: number[],
  old: number[],
  ref: number[],
  indices: number[],
  adv: number[],
  epsilon: number,
  beta: number,
  lr: number,
) {
  const prob = softmax(logits),
    div = kl(prob, ref);
  const gradient = logits.map(
    (_, j) => -beta * prob[j] * (Math.log(prob[j] / ref[j]) - div),
  );
  indices.forEach((index, s) => {
    const ratio = prob[index] / old[index],
      a = adv[s];
    if ((a >= 0 && ratio > 1 + epsilon) || (a < 0 && ratio < 1 - epsilon))
      return;
    for (let j = 0; j < gradient.length; j++)
      gradient[j] +=
        (ratio * a * ((j === index ? 1 : 0) - prob[j])) / indices.length;
  });
  return logits.map((x, i) => x + lr * gradient[i]);
}
export function experiment(
  task: Task,
  algorithm: Algorithm,
  size: number,
  seed: number,
  epsilon: number,
  beta: number,
  lr: number,
  epochs = 6,
) {
  const spec = tasks[task],
    old = reference,
    indices = sample(old, size, seed),
    rewards = indices.map((i) => spec.rewards[i]),
    baseline = old.reduce((s, p, i) => s + p * spec.rewards[i], 0),
    adv = advantages(rewards, algorithm, baseline);
  let logits = [...initialLogits];
  const history = [old];
  for (let i = 0; i < epochs; i++) {
    logits = update(logits, old, reference, indices, adv, epsilon, beta, lr);
    history.push(softmax(logits));
  }
  return { indices, rewards, adv, baseline, history };
}
