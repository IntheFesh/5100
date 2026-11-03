# 环境记录（第 1 轮）

- 操作系统：Linux 6.18.44-fc-v230 (x86_64); Ubuntu 24.04.5 LTS
- CPU：Intel(R) Xeon(R) Processor @ 2.80GHz × 4 核；内存 15 GB；无 GPU（JAX backend = cpu）
- Python：Python 3.12.3（.venv，由 /usr/bin/python3.12 创建）
- 日期：2026-11-03

安装命令（版本全部按提示词固定，没有换任何版本）：

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "jax==0.4.38" "jaxlib==0.4.38" "flax==0.10.2" "optax==0.2.4" "chex==0.1.88" "gymnax==0.0.6" \
    six numpy matplotlib imageio imageio-ffmpeg pillow pytest
```

开发附加包（不属于训练环境，只用于 tools/check_notebook.py 与 tests/test_notebook.py）：`pip install ipython`。

说明：scipy 由 jax 依赖自动装入（analyze.py 有 scipy 时用 scipy 的 t 分位数，否则用内置表，两者一致）。numpy 解析为 2.5.4；test_v5.py 在该组合上全部通过。
gymnax 0.0.6 依赖 gym 0.26.2，导入时会打印一段 Gym 过时警告，可忽略。

## pip freeze（第 1–9 轮结束时，含 ipython）

```
absl-py==2.5.1
asttokens==3.0.2
chex==0.1.88
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
etils==1.15.0
executing==2.3.0
flax==0.10.2
fonttools==4.68.0
fsspec==2026.9.0
gym==0.26.2
gym-notices==0.1.0
gymnax==0.0.6
humanize==4.16.0
ImageIO==2.38.1
imageio-ffmpeg==0.8.1
iniconfig==2.3.1
ipython==9.18.1
ipython_pygments_lexers==1.1.1
jax==0.4.38
jaxlib==0.4.38
jedi==0.20.1
kiwisolver==1.5.1
markdown-it-py==4.2.0
matplotlib==3.11.2
matplotlib-inline==0.2.2
mdurl==0.1.2
ml_dtypes==0.6.0
msgpack==1.2.4
nest-asyncio==1.6.0
numpy==2.5.4
opt_einsum==3.4.0
optax==0.2.4
orbax-checkpoint==0.11.5
packaging==26.3
parso==0.8.7
pexpect==4.9.0
pillow==13.0.0
pluggy==1.7.0
prompt_toolkit==3.0.53
protobuf==7.36.2
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.4
Pygments==2.21.0
pyparsing==3.3.3
pytest==9.2.0
python-dateutil==4.0.0
PyYAML==6.0.3
rich==15.0.0
scipy==1.18.1
setuptools==84.2.1
simplejson==4.2.0
six==1.17.0
stack-data==0.6.3
tensorstore==0.1.86
toolz==1.2.0
traitlets==5.16.1
typing_extensions==4.16.0
tzdata==2026.6
wcwidth==0.9.4
zipp==4.1.2
```
