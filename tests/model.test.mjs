import test from "node:test";
import assert from "node:assert/strict";
import { model as m } from "../scripts/load-model.mjs";
test("Seeded categorical rollout is deterministic", () => {
  assert.deepEqual(
    m.sample(m.reference, 16, 42),
    m.sample(m.reference, 16, 42),
  );
  assert.ok(m.sample(m.reference, 100, 42).every((i) => i >= 0 && i < 4));
});
test("GRPO centers the group and gives no reward gradient on equal rewards", () => {
  assert.ok(
    Math.abs(m.advantages([1, 0, 1, 0], "grpo", 0).reduce((a, b) => a + b, 0)) <
      1e-12,
  );
  assert.deepEqual(m.advantages([1, 1, 1, 1], "grpo", 0), [0, 0, 0, 0]);
});
test("Analytic clipped policy gradient including KL matches finite differences", () => {
  const x = [0.1, 0.2, -0.3, 0.4],
    old = m.reference,
    indices = [0, 1, 2, 3],
    adv = [1, -1, 0.5, -0.5],
    eps = 0.2,
    beta = 0.1,
    delta = 1e-5;
  const next = m.update(x, old, old, indices, adv, eps, beta, 1);
  const objective = (z) =>
    m.surrogate(m.softmax(z), old, indices, adv, eps) -
    beta * m.kl(m.softmax(z), old);
  x.forEach((_, i) => {
    const hi = [...x],
      lo = [...x];
    hi[i] += delta;
    lo[i] -= delta;
    assert.ok(
      Math.abs(next[i] - x[i] - (objective(hi) - objective(lo)) / (2 * delta)) <
        1e-7,
    );
  });
});
test("Demo updates keep a valid distribution and improve default verifiable reward", () => {
  for (const alg of ["ppo", "grpo"]) {
    const e = m.experiment("math", alg, 8, 42, 0.2, 0.05, 0.6);
    for (const p of e.history) {
      assert.ok(p.every((v) => v > 0 && v < 1));
      assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    }
    const after = e.history
      .at(-1)
      .reduce((s, p, i) => s + p * m.tasks.math.rewards[i], 0);
    assert.ok(after > e.baseline);
  }
});
test("Clipping is sign aware", () => {
  assert.equal(m.surrogate([0.7, 0.3], [0.5, 0.5], [0], [1], 0.2), 1.2);
  assert.equal(m.surrogate([0.3, 0.7], [0.5, 0.5], [0], [-1], 0.2), -0.8);
});
