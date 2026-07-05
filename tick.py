# -*- coding: utf-8 -*-
"""真合流: 链上水库做真计算(不是空转记账)。

对比上一版 reservoir_chain.py(水库空晃, 大半记账), 这版让链上水库
每个 tick 真算延迟 XOR: pred(t) = u(t-2) XOR u(t-5)。
关键红线(防记账):
  - 答案(正确XOR)绝不写进账本状态, 只在 tick 结束时临时对照打分
  - 账本只存: 水库状态摘要、读出层Wout、预测对错计数、哈希链
  - 每个 tick 喂一个"当场新生成"的随机位, 水库现场算它2/5步前的XOR
  - 因为输入是每 tick 新掷的, 账本里不可能预存答案 -> 答对=现算

一个 tick:
  1. 读上一状态, 校验 self_hash(篡改则退2)
  2. 恢复水库向量 x 和 已训练的读出层 Wout
  3. 掷一个新随机位 u_t, 推进水库一步, 读出预测, 和真XOR对照
  4. 父哈希链写新状态(累计命中率进账, 但答案不进账)
首个tick做一次性训练(拿一段随机序列训练Wout), 之后纯推理。
"""
import json
import os
import sys
import hashlib
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(ROOT, "chain_compute", "state.json")
GENESIS = "0" * 64
N_RES = 150
SEED = 20260705
IN_SCALE = 0.6
D1, D2 = 2, 5


def sha256_of(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_reservoir():
    rng = np.random.default_rng(SEED)
    W = rng.uniform(-1, 1, size=(N_RES, N_RES))
    W[rng.uniform(size=(N_RES, N_RES)) > 0.1] = 0.0
    W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.uniform(-1, 1, size=(N_RES,))
    return W, Win


def train_readout(W, Win):
    """一次性训练读出层: 用一段随机序列教水库算延迟XOR。答案只在训练时用。"""
    rng = np.random.default_rng(SEED + 1)
    n = 4000
    u = rng.integers(0, 2, size=n).astype(float)
    x = np.zeros(N_RES)
    States = np.zeros((n, N_RES))
    for t in range(n):
        x = np.tanh(IN_SCALE * Win * u[t] + W @ x)
        States[t] = x
    y = np.array([float(int(u[t - D1]) ^ int(u[t - D2])) if t >= D2 else 0.0
                  for t in range(n)])
    warm = 200
    X = np.hstack([States[warm:], np.ones((n - warm, 1))])
    A = X.T @ X + 1e-6 * np.eye(X.shape[1])
    Wout = np.linalg.solve(A, X.T @ y[warm:])
    return Wout


def load_state():
    if not os.path.exists(STATE_PATH):
        return None
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    W, Win = build_reservoir()
    prev = load_state()

    if prev is None:
        Wout = train_readout(W, Win)
        x = np.zeros(N_RES)
        recent = [0.0] * (D2 + 1)          # 最近输入位(算XOR真值用, 不入账)
        step = 0
        parent_hash = GENESIS
        hits = tries = 0
        print(f"冷启动: 训练读出层完毕(Wout {len(Wout)}维), 链从创世开始")
    else:
        recorded = prev["self_hash"]
        if recorded != sha256_of({k: v for k, v in prev.items() if k != "self_hash"}):
            print("STATE HASH MISMATCH", file=sys.stderr)
            sys.exit(2)
        parent_hash = recorded
        step = prev["step"]
        x = np.array(prev["reservoir_state"])
        Wout = np.array(prev["Wout"])
        recent = prev["_recent_inputs"]
        hits, tries = prev["hits"], prev["tries"]

    # 掷一个当场新随机位(用链步数+父哈希派生, 可复验又不可预存答案)
    seed_t = int(hashlib.sha256((parent_hash + str(step)).encode()).hexdigest(), 16) % (2**32)
    u_t = float(np.random.default_rng(seed_t).integers(0, 2))

    # 水库推进一步, 读出对 u(t-2) XOR u(t-5) 的预测
    x = np.tanh(IN_SCALE * Win * u_t + W @ x)
    pred = float(np.hstack([x, 1.0]) @ Wout)
    pred_bit = 1 if pred > 0.5 else 0

    # 真值: 用最近输入算(仅打分, 不写进账本)
    true_xor = int(recent[-D1]) ^ int(recent[-D2])
    correct = (pred_bit == true_xor)
    if step >= D2:
        tries += 1
        hits += int(correct)

    recent = (recent + [u_t])[-(D2 + 1):]
    x_round = [round(v, 8) for v in x.tolist()]

    body = {
        "step": step + 1,
        "task": "delayed_XOR(u[t-2]^u[t-5])",
        "new_input_bit": u_t,
        "prediction_bit": pred_bit,        # 存"我猜的", 不存"正确答案"
        "correct_this_step": bool(correct) if step >= D2 else None,
        "hits": hits, "tries": tries,
        "accuracy": round(hits / tries, 4) if tries else None,
        "reservoir_digest": hashlib.sha256(json.dumps(x_round).encode()).hexdigest()[:16],
        "reservoir_state": x_round,
        "Wout": [round(v, 8) for v in Wout.tolist()],
        "_recent_inputs": recent,
        "parent_hash": parent_hash,
        "seed": SEED,
    }
    body["self_hash"] = sha256_of(body)

    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)
        f.write("\n")

    accs = f"{hits}/{tries}={hits/tries*100:.0f}%" if tries else "预热中"
    mark = "对" if correct else "错"
    print(f"tick {step}->{step+1}: 新位={int(u_t)} 水库预测XOR={pred_bit} "
          f"真值={true_xor if step>=D2 else '-'} [{mark if step>=D2 else '预热'}] "
          f"累计命中 {accs} | self={body['self_hash'][:10]}")


if __name__ == "__main__":
    main()
