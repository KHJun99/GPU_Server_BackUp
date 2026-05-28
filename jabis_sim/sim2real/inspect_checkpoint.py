"""체크포인트의 obs/action space + network 구조 추출."""
import torch
from pathlib import Path
import yaml

CKPT = Path.home() / "isaac_so_arm101/logs/rsl_rl/lift/2026-05-08_10-58-49/model_3999.pt"
PARAMS_DIR = CKPT.parent / "params"

print("=" * 60)
print("CHECKPOINT INSPECTION")
print("=" * 60)
print(f"path: {CKPT}")

# 1. params 폴더 내용 (env config + agent config)
print("\n[params/]")
for f in sorted(PARAMS_DIR.glob("*")):
    print(f"  {f.name}")

# 2. agent config (PPO 설정 + network arch)
agent_yaml = PARAMS_DIR / "agent.yaml"
if agent_yaml.exists():
    print(f"\n[agent.yaml]")
    with open(agent_yaml) as f:
        cfg = yaml.unsafe_load(f) if hasattr(yaml, "unsafe_load") else yaml.safe_load(f)
    # network 관련만
    if isinstance(cfg, dict):
        for k in ("policy", "actor_critic", "network"):
            if k in cfg:
                print(f"  {k}: {cfg[k]}")
    else:
        print(f"  (object) {type(cfg).__name__}")
        # rsl_rl 객체일 수 있음
        for attr in ("policy", "actor_critic"):
            if hasattr(cfg, attr):
                print(f"  cfg.{attr} = {getattr(cfg, attr)}")

# 3. env config — obs/action space 직접
env_yaml = PARAMS_DIR / "env.yaml"
if env_yaml.exists():
    print(f"\n[env.yaml] (확인용 첫 50줄)")
    with open(env_yaml) as f:
        for i, line in enumerate(f):
            if i >= 50: break
            print(f"  {line.rstrip()}")

# 4. checkpoint state_dict 구조
print(f"\n[checkpoint state_dict]")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"  keys: {list(ckpt.keys())}")

if "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
elif "actor_critic" in ckpt:
    sd = ckpt["actor_critic"]
elif "policy" in ckpt:
    sd = ckpt["policy"]
else:
    sd = ckpt

print(f"\n  state_dict keys (총 {len(sd)}개):")
for k, v in sd.items():
    if hasattr(v, "shape"):
        print(f"    {k}: {tuple(v.shape)}")
