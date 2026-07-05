# -*- coding: utf-8 -*-
"""双向闭环上真河: 水库算出的中间状态写回链持久存, 下一步喂回水库。

任务: 带重置累积奇偶 s(t)=s(t-1) XOR u(t); reset(t)=1 则 s(t)=0。
每个 tick:
  1. 读上一状态, 校验 self_hash(篡改退2)
  2. 恢复水库 x、读出层 Wout、上一步预测 s_prev(链持久存的那一位=双向反馈)
  3. 用"创世种子+步数"派生 u_t 和 reset_t(可复验, 不可预存答案)
  4. 水库吃 u_t,reset_t,s_prev 推进 -> 读出预测 s_hat
  5. s_hat 写回链持久态(下步喂回自己)=闭环
  6. 父哈希链写新状态
红线: 真 parity 绝不入账, 只临时打分(靠重放派生序列现场算)。首tick teacher-forced训练Wout。
"""
import json
import os
import sys
import hashlib
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(ROOT, "chain_bidir", "state.json")
GENESIS = "0" * 64
N_RES = 150
SEED = 20260705
IN_SCALE = 0.6
P_RESET = 1.0 / 80.0


def sha256_of(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_reservoir():
    """随机固定水库 + 三路输入权重(u/reset/s_prev)。全从SEED派生, 可复验。"""
    rng = np.random.default_rng(SEED)
    W = rng.uniform(-1, 1, size=(N_RES, N_RES))
    W[rng.uniform(size=(N_RES, N_RES)) > 0.1] = 0.0
    W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))
    r = np.random.default_rng(SEED + 3)
    Win_u = r.uniform(-1, 1, N_RES)
    Win_r = r.uniform(-1, 1, N_RES)
    Win_s = r.uniform(-1, 1, N_RES)
    return W, Win_u, Win_r, Win_s


def derive_input(step):
    """创世种子+步数派生 (u_t, reset_t)。任何人可独立重放复验。"""
    h = hashlib.sha256(f"{SEED}:{step}".encode()).hexdigest()
    u_t = float(int(h[:8], 16) & 1)
    reset_t = 1.0 if (int(h[8:16], 16) / 0xFFFFFFFF) < P_RESET else 0.0
    return u_t, reset_t


def train_readout(W, Win_u, Win_r, Win_s):
    """一次性 teacher-forced 训练: 用真 s_prev 教水库算 s(t)。答案只在训练时用, 不入账。"""
    n = 5000
    x = np.zeros(N_RES)
    States = np.zeros((n, N_RES))
    s_true = np.zeros(n)
    cur = 0
    for t in range(n):
        u_t, reset_t = derive_input(t)
        s_prev = 0.0 if t == 0 else s_true[t - 1]
        x = np.tanh(IN_SCALE * (Win_u * u_t + Win_r * reset_t + Win_s * s_prev) + W @ x)
        States[t] = x
        cur = 0 if reset_t else (cur ^ int(u_t))
        s_true[t] = cur
    warm = 300
    X = np.hstack([States[warm:], np.ones((n - warm, 1))])
    A = X.T @ X + 1e-6 * np.eye(X.shape[1])
    Wout = np.linalg.solve(A, X.T @ s_true[warm:])
    return Wout


def true_parity_at(step):
    """重放派生序列, 现场算 step 处的真 parity。不入账, 仅打分。"""
    cur = 0
    for t in range(step + 1):
        u_t, reset_t = derive_input(t)
        cur = 0 if reset_t else (cur ^ int(u_t))
    return cur


def load_state():
    if not os.path.exists(STATE_PATH):
        return None
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    W, Win_u, Win_r, Win_s = build_reservoir()
    prev = load_state()

    if prev is None:
        Wout = train_readout(W, Win_u, Win_r, Win_s)
        x = np.zeros(N_RES)
        s_prev = 0.0
        step = 0
        parent_hash = GENESIS
        hits = tries = 0
        print(f"冷启动: teacher-forced训练Wout({len(Wout)}维)完毕, 双向闭环链从创世开始")
    else:
        recorded = prev["self_hash"]
        if recorded != sha256_of({k: v for k, v in prev.items() if k != "self_hash"}):
            print("STATE HASH MISMATCH", file=sys.stderr)
            sys.exit(2)
        parent_hash = recorded
        step = prev["step"]
        x = np.array(prev["reservoir_state"])
        Wout = np.array(prev["Wout"])
        s_prev = float(prev["s_prev"])          # 链持久存的水库预测态(双向反馈)
        hits, tries = prev["hits"], prev["tries"]

    # 派生新输入(可复验, 不可预存答案)
    u_t, reset_t = derive_input(step)

    # 水库吃 u_t,reset_t,s_prev 推进一步, 读出对 s(t) 的预测
    x = np.tanh(IN_SCALE * (Win_u * u_t + Win_r * reset_t + Win_s * s_prev) + W @ x)
    pred = float(np.hstack([x, 1.0]) @ Wout)
    s_hat = 0.0 if reset_t else (1.0 if pred > 0.5 else 0.0)

    # 打分: 真值靠重放现场算(不入账)
    true_s = true_parity_at(step)
    correct = (int(s_hat) == true_s)
    tries += 1
    hits += int(correct)

    x_round = [round(v, 8) for v in x.tolist()]
    body = {
        "step": step + 1,
        "task": "running_parity_with_reset (s=s_prev^u, reset->0)",
        "new_input_bit": u_t,
        "reset_bit": reset_t,
        "prediction_bit": int(s_hat),        # 存"我猜的状态", 不存正确答案
        "s_prev": s_hat,                     # 写回链持久存, 下步喂回自己(双向闭环)
        "correct_this_step": bool(correct),
        "hits": hits, "tries": tries,
        "accuracy": round(hits / tries, 4) if tries else None,
        "reservoir_digest": hashlib.sha256(json.dumps(x_round).encode()).hexdigest()[:16],
        "reservoir_state": x_round,
        "Wout": [round(v, 8) for v in Wout.tolist()],
        "parent_hash": parent_hash,
        "seed": SEED,
    }
    body["self_hash"] = sha256_of(body)

    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)
        f.write("\n")

    accs = f"{hits}/{tries}={hits/tries*100:.0f}%" if tries else "-"
    mark = "对" if correct else "错"
    print(f"tick {step}->{step+1}: u={int(u_t)} reset={int(reset_t)} s_prev={int(s_prev)} "
          f"水库预测s={int(s_hat)} 真值={true_s} [{mark}] 累计 {accs} | self={body['self_hash'][:10]}")


if __name__ == "__main__":
    main()
