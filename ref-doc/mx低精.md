
## 1.模型量化技术简介

### 1.1 什么是量化

数值通常可以以浮点数的形式表示，即带有正负号和小数点的数字。这些数值是由bits组成。根据 IEEE-754 标准，这些 “bits” 可以用来表示三个不同的部分以构成一个完整的数值：**​符号位、指数部分以及尾数部分。​**这三个部分结合起来，就能计算出具体的数值。

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090812756467/51556745/da0e134250b6487982a738ab1767f0a4.png)

一般来说，**用来表示数值（value）的 “bit” 越多，得到的数值（value）精度就越高。**​**可用的 “bits” 数量越多，所占用的内存越大。**

模型量化的核心在于将模型参数的精度从较高位宽（例如32位浮点）降低到较低位宽（例如 8 位整数）。

在减少参数的 “bits” 数量时，通常会出现一定的精度损失（即丢失一些数值细节）。

**模型量化的主要目的就是减少表示原始参数所需的 “bits” 数量，同时尽可能保留原始参数的精度。**

以 BF16量化→INT8 举例说明：

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090812756467/51556744/2794da5bb1b34a628a8db57de52ca8a1.png)

### 1.2 量化常见分类

#### 1.2.1 按量化与训练的关系分类

* **PTQ**
  Post-Training Quantization（PTQ）在训练完模型之后对模型的参数（包括权重和激活值）进行量化。PTQ在众多量化技术中是最为流行的一种，这种方法也是我们目前最常用的。
* **QAT**
  Quantization Aware Training（QAT）的目标是在训练过程中学习量化过程。QAT 通常比 PTQ 更准确，因为在训练过程中已经考虑了量化。DeepSeek、Qwen、GLM等现在都会发布量化的开源权重，这些原始权重无需二次量化，可直接使用。

#### 1.2.2 按量化bit分类

量化通常主要关注线性层的激活值（Activation）和权重（Weight），如果激活值保持16bit不量化，仅将权重量化到4-bit，那我们就称这种量化是W4A16量化。如果将激活值和权重都量化到8-bit，那就可以简称是W8A8量化。

```
位宽越来越高（精度↑，压缩率↓）
W8A8 (INT8)   ── W4A8 (INT4/INT8) ── W4A16 (INT4/BF16) ── W8A16 (INT8/BF16) ── BF16(不量化)
                        │
                        └─> FP8 家族（8 bit，带指数）
                               ├─ 普通 FP8（block-wise/tensor scale，float32 共享 scale）
                               └─ MXFP8（Microscaling：E4M3 + E8M0 每 32 元素一个共享指数）# A5 主打
```

- **整数量化（INT8/INT4）**：把浮点缩放到 int 区间，用每通道/token 一个 **float32 scale**。硬件支持最广、精度与性能均衡，但 scale 粒度粗（每通道一个），对离群值敏感。
- **FP8**：8 bit 浮点（E4M3/E5M2），自带指数动态范围，分布非均匀更贴合权重/激活实际分布。Ascend 用 **block-wise FP8**（`weight_block_size` 如 128×128 一个 float32 scale）。
  - E4M3（1 符号位 + 4 指数位 + 3 尾数位）：精度优先（±448），适合前向的激活与权重，是默认选择。
  - E5M2（1 符号位 + 5 指数位 + 2 尾数位）：范围优先（±57344），适合反向传播的梯度。
- **MXFP8（微缩放）**：**数据仍存 8bit（E4M3），但 scale 用 E8M0（纯指数、无尾数），每 32 个元素一个共享指数**。默认把张量按 32 元素一份独立缩放，使不同数量级的注意力头/离群值各归其位，精度比 block-FP8 更稳；在 A5 硬件上有专门的 `npu_quant_matmul` / `npu_grouped_matmul` 原生支持。**这就是本文的主角**。

以deepseek-V4为例：

- 针对效率敏感的部分（MoE）：使用 MXFP4 格式，搭配 E8M0 缩放因子。
- 针对精度敏感的部分（Attention）：保留在 FP8 或更高精度。

#### 1.2.3 按量化scale的生成阶段分类

量化与反量化都包含了一个重要参数`scale`，scale生成的好坏决定了量化精度的高低。根据scale生成的阶段，我们可以将量化分为动态量化和静态量化。

* **静态量化**
  scale在推理前已经预先离线生成。**权重量化其实都属于静态量化**，部分量化算法也可以提前生成用于量化激活值的scale，此时激活值也是静态量化。
* **动态量化**
  scale在推理过程中动态生成。**激活值往往是动态量化**，这是因为激活值是在推理过程中动态生成的，此时根据真实激活值的数据分布进行量化，精度相比静态量化更高。

#### 1.2.4 按量化粒度分类

量化精度损失的主要来源是较大的异常值导致scale被拉大，和离群值一起被量化的元素都需要承受较高的量化损失。因此可以通过更细的量化粒度，将离群值隔离到更小的范围，使它影响更小。

量化粒度越细，精度越高，代价是需要存储更多的scale，导致压缩比降低，以及更多的计算。

* **Per-Tensor 全局共享**
  ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090812756467/51556742/7c83a87ff960421dbb7d66a58b4433d4.png)
* **Per-Token/Per-Channel 行/列共享**
  ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090812756467/51556743/025dfd3b88bd42eebb71d25a0a05c253.png)
* **Per-Group(Per-Block) 块内共享**
  ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090812756467/51556741/8e67165835484a5187eb2cd61b59593c.png)

### 1.3 优势

1. **显存/带宽减半或更多**：BF16（2 字节）→ INT8/FP8（1 字节）→ INT4/MXFP4（0.5 字节）。
2. **吞吐/算力提升**：低位宽的 `npu_quant_matmul` / `npu_grouped_matmul` 在 A5 硬件上有更高 TOPS，且减少带宽瓶颈下的等待。
3. **MX 相对普通 FP8 的精度优势**：**E8M0 每 32 元素一个共享指数**，比 block-FP8（128×128 一个 float32 scale）粒度细得多——**一个极端离群值只会影响它所在的 32 元素 group，不会压低整个 block 的有效位**，因此 MX 在长尾/离群分布（权重、激活峰值）下精度更稳。这是 A5 把它当「推荐低精路径」的根本原因。
4. **省去离线重量化**：普通 block-FP8 checkpoint 在 A5 上由 `is_950()` **运行时自动再量化为 MXFP8**（`fp8_block.py`），无需额外离线转换或按模型打补丁。
5. **KV Cache 也可低精**：`C8`/INT8 注意力等方案可同时压 KV cache 显存，进一步省内存。

## 2 权重如何量化

### 2.1 通用量化工序

以**权重离线量化**为例：

```
① 扫描原始权重（BF16）
   收集该层权重的数值分布（min/max、分位数、离群值）
② 定分块（决定 scale 粒度）
   选 per-tensor / per-channel / per-block(128×128) / per-group(32)
③ 求每个分块的 scale
   通常 scale = 该分块最大绝对值 / 该格式最大可表示值
   INT8:  fp32 scale = max(|w|) / 127
   FP8:   fp32 scale（per-block 128×128）
   MX:    E8M0 指数 scale（per-group 32），由 max(|w|) 推出 2 的指数幂
④ 量化（截断/舍入到目标格式）
   每个元素 / scale
   → 得到 int8 / float8_e4m3fn / float4_e2m1fn_x2 数据
⑤ 存盘
   weight(量化数据) + weight_scale(按粒度存的 scale) + 可选 weight_offset
   → 写入 checkpoint / `quant_model_description.json`
```

当前量化的完整流程如下：
BF16 Model --> 量化工具（Modelslim/LLM-Compressor等） --> 量化权重 --> vLLM-Ascend推理

### 2.2 ModelSlim（主打）

LLM模型：王建新 00601343
多模态：李启统 00845740

昇腾的离线量化/压缩工具 **ModelSlim** 完成上述工序后，输出：

- 量化后的**权重/scale文件**；
- 一个 **`quant_model_description.json`**：**逐层（per-layer）描述**每个 prefix 用什么 quant_type（如 `"W8A8_MXFP8"`、`"W4A8_MXFP"`）以及 group_size 等。

#### 2.2.1 参考链接

量化基础知识与ModelSlim工具介绍：
https://onebox.huawei.com/v/da877a7be6b194d1459a64350ce9f547

ModelSlim入门：
https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/quick_start/quantization_quick_start.md#msmodelslim-%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8
https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md#%E6%9D%83%E9%87%8D%E9%87%8F%E5%8C%96%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97

常见模型量化实践：
https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_flash_w4a4c8.yaml
https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/glm_5_2/glm_5_3_w8a8c8_mxfp8.yaml

## 3 vllm量化框架

vLLM 的量化框架位于 `vllm/model_executor/layers/quantization/`

### 3.1 核心元素

| 角色 | 类 / 概念 | 职责 |
| ---- | --------- | -------- | ---- |
| **角色 1 · 配置层** | `QuantizationConfig` | 对应一种"量化方法格式"（如 fp8），解析 checkpoint 中 config.json 的 `quantization_config` 字段；决定每个 layer 用什么量化方法 |
| **角色 2 · 方法层** | `QuantizeMethodBase` | 三个基类：① `LinearMethodBase` —— 普通 Linear 层；② `FusedMoEMethodBase` —— MoE 专家层；③ `BaseKVCacheMethod` —— KV Cache / Attention 层 |
| **角色 3 · 被量化的层** | `Layer` | 模型构建时逐层挂载，每个 Linear / FusedMoE / Attention 层挂上一个 `quant_method` 属性；前向计算时调用 quant_method 的方法 |

### 3.2 端到端流程

> 下面 6 步是 vLLM **上游通用**框架流程；vllm-ascend 对应的具体代码链路见 4.3（阶段一～四）。

1. **读取量化配置** —— 读 config.json 的 `quantization_config`（或 `--quantization` 参数），按 quant_method 名查已注册的 Config 类（注册机制详见 4.3.1）
2. **构建模型 · 逐层分发量化方法**
   
   - 逐层调用 `quant_config.get_quant_method(layer, prefix)`
   - 返回该层的 QuantizeMethodBase 实例，存入 `layer.quant_method`
   - > ⚠️ 未量化的层 → 返回 `Unquantized*Method`（回退路径）
3. **`create_weights(...)` —— 声明参数**
   
   - 按量化格式声明参数（weight / weight_scale / weight_offset / input_scale ...）
   - 此时只分配空张量
4. **加载 checkpoint 权重**
   
   - weight_loader 填充参数
5. **`process_weights_after_loading(layer)` —— 加载后处理**
   
   - 转置、NZ 格式转换、weight pack / unpack、预计算 deq_scale 等
   - **硬件相关优化的关键 hook**
6. **推理 —— `quant_method.apply(...)`**
   
   - 每个 forward 中调用，在 apply 内部完成：
     `激活量化 → 量化矩阵乘 → 反量化输出`

## 4 vllm-ascend量化模块

代码位于 `vllm_ascend/quantization/`，采用 **"Config 层 → Adapter 层 → Scheme 层"** 三层设计。

### 4.1 目录结构与三层划分

- **Config 层** —— 量化格式的"入口"
- **Adapter 层** —— vLLM 接口的包装器
- **Scheme 层** —— 算法实现与注册机制

```
vllm_ascend/quantization/
├── __init__.py            # 懒加载入口（避免循环导入）
├── quant_type.py          # QuantType 枚举（W8A8 / W4A16 / W8A8MXFP ...）
├── quant_parser.py        # MX 类型量化参数映射表
├── method_adapters.py   # ★ Adapter 层：对接 vLLM 接口的包装类
├── utils.py               # 工具函数（量化方法自动检测等）
├── configs/             # ★ Config 层：对接 vLLM 注册机制
│   ├── modelslim_config.py           # "ascend"（ModelSlim）
│   ├── compressed_tensors_config.py  # "compressed-tensors"（LLM-Compressor）
│   ├── fp8_config.py                 # "fp8" / "deepseek_v4_fp8"（原生 FP8）
│   └── modelopt_mxfp8_config.py      # "mxfp8" / "modelopt_mxfp8"（ModelOpt）
└── methods/             # ★ Scheme 层：具体量化算法实现
    ├── base.py          # 三个抽象基类
    ├── registry.py      # @register_scheme 注册机制
    ├── w8a8/           # W8A8 家族（int8 静态/动态、fp8、mxfp8、pdmix）
    ├── w4a8/           # W4A8 家族（int8 激活、mxfp）
    ├── w4a4/           # W4A4 家族（mxfp4、flatquant、laos）
    ├── wna16/          # W4A16 / W8A16（权重压缩、激活不量化）
    └── kv_cache/       # KV Cache / Attention 量化
```

### 4.2 三层调用链全景

```
step1  决定用哪个量化方法（决定 quant_config）
        └─ 名字来源：
             ├─ 用户显式 --quantization <name>
             ├─ override （ModelConfig 阶段，如 deepseek_v4 的 "fp8"→"deepseek_v4_fp8"）
             └─ 自动探测 （quant_model_description.json → "ascend"；config.json → "fp8"）
        └─ get_quantization_config(name) → 拿到注册表里那个 quant_config 类
             （这个类的"名字→类"映射在阶段一已由 vllm-ascend import/注册覆盖好）

step2  模型构造时，每层拿 quant_method
        └─ quant_config.get_quant_method(layer, prefix)
             └─ 每个 quant_config 决定"这层返回哪个 scheme"（linear/moe/attention + 是哪个 WxAx）
             └─ 返回的 quant_method（scheme）就是"这层的量化实现者"

step3  权重创建 / 布局 / 前向，全由该 quant_method 管
        ├─ create_weights() / get_weight()：按该方案的布局建 weight/scale
        ├─ process_weights_after_loading()：转置、配对、再量化（如 fp8→MXFP8）
        └─ apply()：前向调用真实算子（如 npu_dynamic_mx_quant + npu_quant_matmul）
```

### 4.3 代码链路

> 承接 3.2 的通用 6 步，下面 4 个阶段是这些步骤在 vllm-ascend 中的具体代码落地：注册 → 验证/探测 → 装配 quant_method → 前向。

#### 4.3.1 阶段一：命令启动 → 注册Ascend 量化 Config

---

> 本阶段主要是「**vllm 启动** → **vllm-ascend 平台注册**」的第一次交界。
> 
> - **vllm**：解析 CLI 参数、构造 parser
> - **vllm-ascend**：`NPUPlatform.pre_register_and_update` 完成三个操作：① 打全局 patch（和当前内容关联不大）；② 往 `--quantization`的可选项里塞入`"ascend"`（使得可以识别--quantization ascend）；③ **触发 import Ascend 的各个量化 Config，注册进 vllm 的映射表**

```
vllm serve ...
  → vllm/vllm/entrypoints/cli/launch.py
    → vllm/vllm/engine/arg_utils.py
        current_platform.pre_register_and_update(parser)        # CLI 阶段（NPUPlatform(Platform)，实际使用替换后的vllm-ascend实现）
          → vllm-ascend/vllm_ascend/platform.py  NPUPlatform.pre_register_and_update
              ├─ adapt_patch(is_global_patch=True)              # 打 vllm-ascend的全局patch
              ├─ 往--quantization 的 choices 里追加 "ascend"   # 使得可以解析vllm serve --quantization ascend
              ├─── quant_action.choices.append(ASCEND_QUANTIZATION_METHOD) # ASCEND_QUANTIZATION_METHOD="ascend"
              └─ 标准量化后端（A5 是 QuantizationBackendFamily.STANDARD）
                   vllm_ascend/quantization/__init__.py
                     ├─ configs/modelopt_mxfp8_config.py  → @register_quantization_config("mxfp8") ("modelopt_mxfp8")
                     ├─ configs/fp8_config.py             → @register_quantization_config("fp8") ("deepseek_v4_fp8")
                     ├─ configs/compressed_tensors_config.py
                     └─ configs/modelslim_config.py       → @register_quantization_config("ascend")
```

**关键步骤**：每个 `@register_quantization_config("xxx")` 把 "xxx"对应的Config类注册到`_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`，最终在 `get_quantization_config()`时将vllm-ascend实现替换vllm原生实现

#### 4.3.2 阶段二：Engine 主进程构建 VllmConfig → 解析模型配置，重建model_config.quantization、vllm_config.quant_config

```
EngineCore / LLMEngine 启动
  → vllm/vllm/engine/arg_utils.py:1906
      create_engine_config()
        → self.create_model_config()
            → ModelConfig.__post_init__  (vllm/vllm/config/model.py)
                └─ _verify_quantization()  (vllm/vllm/config/model.py)
                    ├─ 读 config.json 的 quantization_config（hf_quant_config）
                    ├─ 遍历 overrides 列表（含 "deepseek_v4_fp8" 等）
                    │     调 get_quantization_config(name).override_quantization_method(quant_cfg, self.quantization, hf_config)
                    │     → 命中则 model_config.quantization = 改写后的名字（如 "fp8"→"deepseek_v4_fp8"）
                    └─ 有用户指定config时，使用用户指定的（不常见）
                  # 此时 model_config.quantization 已确定（override/用户指定）
        → VllmConfig构造
            → VllmConfig.__post_init__  (vllm/vllm/config/vllm.py:972) 
                ├─ (1) 第一次解析  line 1028:
                │      if self.quant_config is None:
                │          self.quant_config = VllmConfig._get_quantization_config(model_config, load_config)
                ├─ (2) 平台 hook   line 1270: current_platform.apply_config_platform_defaults(self)   # cudagraph_capture_size的调整，和mx无关
                └─ (3) 平台 hook   line 1488: current_platform.check_and_update_config(self)          # 量化相关实现
                        → vllm-ascend/vllm_ascend/platform.py  NPUPlatform.check_and_update_config
                            ├─ 第 3 步  maybe_auto_detect_quantization(vllm_config)   ← 仅当 model_config.quantization 仍为 None（override/用户指定都没处理到）才自动探测+重建
```

```python
# deepseek v4的改写逻辑
 def override_quantization_method(
     cls, hf_quant_cfg, user_quant, hf_config=None
 ) -> QuantizationMethods | None:
     if not (
         isinstance(hf_quant_cfg, dict)
         and (
             hf_quant_cfg.get("quant_method") in ("fp8", "deepseek_v4_fp8")
             or (
                 hf_quant_cfg.get("quant_method") == "quark"
                 and cls._is_quark_mxfp4_ocp(hf_quant_cfg)
             )
         )
     ):
         return None
     model_type = getattr(hf_config, "model_type", None)
     if model_type == "deepseek_v4" or user_quant == "deepseek_v4_fp8":
         return "deepseek_v4_fp8"
     return None
```

```python
# 探测重建quant_config和quantization
# 🔧 vllm-ascend/vllm_ascend/quantization/utils.py:191
maybe_auto_detect_quantization(vllm_config):
    detected = detect_quantization_method(model, revision)   # 识别实际的量化方式
    if user_quant is None:
        # 写入model_config.quantization："fp8" / "ascend"
        model_config.quantization = detected
        # 重建 quant_config（否则 __post_init__ 已把 quant_config 定死为 None）
    ...
    vllm_config.quant_config = VllmConfig._get_quantization_config(model_config, load_config)
```

`detected = detect_quantization_method()`识别优先级：

1. 存在`quant_model_description.json` → 返回`"ascend"`（ModelSlim量化生成json的场景，使用ModelSlim的配置即可）
2. `config.json` 的 quantization_config.quant_method为"compressed-tensors" → 返回`"compressed-tensors"`
3. `config.json` 的 quantization_config.quant_method为"fp8" → 返回`"fp8"`

几种config配置示例：

1. quant_model_description.json

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090812756467/51263522/3d9a1b2e8aec499780ac85555da8f8ce.png)

2. config.json，quantization_config.quant_method为"fp8"，deepseek-v4

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090812756467/51263907/e88416bde54d49879c1a640ad47dbbba.png)

3. config.json，quantization_config.quant_method为"mxfp8"，minimax-m3-mxfp8
   [config.json · MiniMaxAI/MiniMax-M3-MXFP8 at main](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8/blob/main/config.json)

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090812756467/51357510/94df368a43274b02a33dbca26723c4d0.png)

目前仓上已有的model_config.quantization和对应的量化config/量化method的映射关系：

```
model_config.quantization (字符串, vllm 字段)
    "mxfp8"/"modelopt_mxfp8"(minimax/ModelOpt(NVIDIA) 格式的权重)  ──►  AscendModelOptMxFp8Config
                                                                 └─ get_quant_method → AscendLinearMethod(AscendW8A8MXFP8DynamicLinearMethod)  [原生 MX]
    "fp8"(普通fp8的权重)        ──►  AscendFp8Config
                              └─ AscendLinearMethod(AscendFp8BlockLinearMethod)  [is_950() → 再量化为 MXFP8]
    "deepseek_v4_fp8"(DeepSeek-V4系列)         ──►  AscendDeepseekV4FP8Config
                                              └─ ...AscendW8A8MXFP8DSDynamicLinearMethod  [DS 版 MX]
    "ascend" (ModelSlim)     ──►  🔧 AscendModelSlimConfig
                             └─ 按config中逐层的json方案进行
```

#### 4.3.3 阶段三：加载模型 → 给每个 Layer 落 MX quant_method

```
Engine 主进程
  └─ Executor 拉起 NPUWorker，把 VllmConfig 传给 worker
        → vllm-ascend/vllm_ascend/worker/worker.py NPUWorker.load_model()
            └─ (set_current_vllm_config(self.vllm_config))
                → vllm-ascend/vllm_ascend/worker/model_runner_v1.py  NPUModelRunner.load_model()
                    → get_model(vllm_config=...)
                        → vllm/.../model_loader/utils.py  initialize_model()
                            ├─ configure_quant_config(quant_config, model_class)
                            └─ model_class(**...)   # 构造模型
                                └─ 每个 Linear/MoE 层 __init__ 
                                     → vllm/model_executor/layers/linear.py
                                           quant_method = quant_config.get_quant_method(self, prefix) # 这里的quant_config已经是此前AscendCompressedTensorsConfig,AscendFp8Config,AscendModelOptMxFp8Config,AscendModelSlimConfig这些config了
                                           → AscendModelOptMxFp8Config.get_quant_method # 以AscendModelOptMxFp8Config为例
                                               → AscendLinearMethod(AscendW8A8MXFP8DynamicLinearMethod())   # MX 分支
                                       → self.quant_method.create_weights(...)   # 按 MX 布局（AscendW8A8MXFP8DynamicLinearMethod）建 weight/scale

    # 权重加载完成后
    → DefaultModelLoader.load_model (default_loader.py:417,457)
        → process_weights_after_loading(model, ...)      # vllm/.../model_loader/utils.py:100
            for module in model.named_modules():
                module.quant_method.process_weights_after_loading(module)
                → 🔧 AscendW8A8MXFP8DynamicLinearMethod.process_weights_after_loading  # MX （AscendW8A8MXFP8DynamicLinearMethod）转置/配对
                → 🔧 AscendFp8BlockLinearMethod.process_weights_after_loading          # is_950() → 再量化 MXFP8
```

#### 4.3.4 阶段四：前向执行

> vllm-ascend 实现各 MX 方案（各个quant method）的 `apply`函数、选择激活/权重数据类型与 group_size、把 `scale_alg` 传进算子。

`AscendW8A8MXFP8DynamicLinearMethod.apply`（`w8a8_mxfp8.py:80`）：

```python
quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
    x, dst_type=torch.float8_e4m3fn, scale_alg=self.dynamic_mx_quant_scale_alg)
output = torch_npu.npu_quant_matmul(
    quantized_x, layer.weight, layer.weight_scale,
    scale_dtype=float8_e8m0fnu, pertoken_scale=pertoken_scale,
    pertoken_scale_dtype=float8_e8m0fnu, group_sizes=[1,1,self.group_size], ...)
```

### 4.4 A5 上 fp8 与 mxfp8 的区别

#### 4.4.1 "fp8 能跑" 和 "mxfp8 能跑" 指的是不同 checkpoint

| | **fp8** | **mxfp8** |
|---|---|---|
| 名字 / Config | `"fp8"` → `AscendFp8Config` | `"mxfp8"` / `"modelopt_mxfp8"` → `AscendModelOptMxFp8Config` |
| 面向的 checkpoint | 普通 **block-FP8** 权重（Qwen3-FP8 等，`weight_block_size=128×128`） | **原生 MX** 权重（ModelOpt / MiniMax 风格，E8M0 per-group 32） |
| 权重 dtype | `float8_e4m3fn` | `float8_e4m3fn`（数据本身一样） |
| 权重 scale | **float32**，**per-block 128×128** 一个 | **E8M0**（纯指数），**per-group 32** 一个 |

> **两者的"数据"都是 8 bit E4M3；真正的区别在 scale 的组织**——fp8 用 float32、every 128×128 block 一个；mxfp8 用 E8M0 纯指数、every 32 元素一个。

#### 4.4.2 在 A5 上"跑到 MX"，两者动作不同

- **mxfp8**：权重**本来就是 MX 布局**（E8M0 per-group 32），原生直接走 `npu_dynamic_mx_quant` + `npu_quant_matmul`，**零转换**。
- **普通 fp8**：block-FP8 布局**不是** A5 喜欢的 MX 布局，所以 `AscendFp8BlockLinearMethod` 在 **`is_950()`** 时会把它**转成 MXFP8 再执行**。转的方式有两种（以服务器 code-0902-catlass 版本为准）：
  - **普通 `"fp8"` 路径**：**重新扫描权重数值 `_mx_quantize`**，把 block-FP8 权重真正重算成 MXFP8（有重量化开销，但能复用普通 FP8 checkpoint 到单卡）；
  - **DSV4 路径（`"deepseek_v4_fp8"`）**：**不提重量化**，只把 128×128 的 float32 指数**提出来重排**成 MX 的 32×1 E8M0 布局。

## 5 当前框架支持的量化方式/量化算法全景图

### 5.1 全量量化 Config 清单

当前 vLLM Ascend 注册了 ​**5 个量化格式入口**​（对应 4 个文件、5 个 Config 类）：

| `quant_method`名             | Config 类                           | 权重来源                                               | 描述文件                                             | 特点                                               |
| ---------------------------------- | ------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| `ascend`                     | `AscendModelSlimConfig`         | ​**ModelSlim**​（华为昇腾官方量化工具）        | `quant_model_description.json`                   | 按层粒度混合精度，一个模型可同时含多种 quant_type |
| `compressed-tensors`         | `AscendCompressedTensorsConfig` | ​**LLM-Compressor**​（vLLM 官方压缩库）        | `config.json`的`quantization_config`         | 标准化 schema，按 config_groups/targets 匹配层    |
| `fp8`                        | `AscendFp8Config`               | ​**HF 原生 FP8 checkpoint**​（官方发布即量化） | `config.json`（`weight_block_size`必须存在） | 免离线量化，仅支持 block-wise + dynamic 激活       |
| `deepseek_v4_fp8`            | `AscendDeepseekV4FP8Config`     | DeepSeek V4 原生 FP8 checkpoint                        | 同上                                                 | DS 原生 128×128 block 布局，专家层可为 FP4        |
| `mxfp8`/`modelopt_mxfp8` | `AscendModelOptMxFp8Config`     | ​**ModelOpt**​（NVIDIA TensorRT 生态）         | `config.json`                                    | MXFP8 权重 + 动态 MXFP8 激活                       |

#### 5.1.1 `ascend`（ModelSlim）—— 主力路径

* ModelSlim 量化后在模型目录生成 `quant_model_description.json`，**逐层**记录量化类型：

```json
{
    "model.layers.0.linear_attn.in_proj_qkvz.weight": "W8A8_DYNAMIC",
    "model.layers.0.linear_attn.out_proj.weight": "FLOAT",
    "model.layers.0.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC"
}
​
```

* `get_quant_method()` 逐层查询该表：`FLOAT` 或缺失 → 回退 `AscendUnquantizedLinearMethod` / `AscendUnquantizedFusedMoEMethod`；`W8A8_DYNAMIC` 等则经 registry 创建对应 Scheme。
* 支持​**融合模块映射**​（`packed_modules_mapping` / `UPDATED_PACKED_MODULES_MAPPING`）：checkpoint 里 `gate_proj`+`up_proj` 分开存、vLLM 模型里是融合的 `gate_up_proj`，Config 负责对齐两边命名，并校验各 shard 量化类型一致。
* 同时承载 ​**KV Cache 量化元数据**​（见 5.3 节）：`fa_quant_type`（MLA K cache）、`kv_cache_type: C8`（dense KV cache）、`indexer_quant_type`（SFA indexer）。
* 使用方式：`vllm serve <model> --quantization ascend`。

#### 5.1.2 `compressed-tensors`（LLM-Compressor）

* 从 `quantization_config.config_groups[*]` 解析出 `target → (weights QuantizationArgs, input_activations QuantizationArgs, format)` 映射。
* `_detect_quant_type()` 根据权重/激活的 `type / num_bits / strategy / dynamic / group_size / format` 六元组推断出内部 quant_type：

| LLM-Compressor schema                                       | 推断出的 quant_type         |
| ------------------------------------------------------------- | ------------------------------ |
| INT8 channel 权重 + INT8 tensor 静态激活                    | `W8A8`                   |
| INT8 channel 权重 + INT8 token 动态激活                     | `W8A8_DYNAMIC`           |
| FP8 channel 权重 + FP8 token 动态激活                       | `W8A8FP8_DYNAMIC`        |
| INT4 channel 权重 + INT8 token 动态激活                     | `W4A8_DYNAMIC`（仅 MoE） |
| INT4 group 权重、无激活量化                                 | `W4A16`（仅 MoE）        |
| format=mxfp8（FP8 group32 静态权重 + FP8 group32 动态激活） | `W8A8_MXFP8`             |
| format=mxfp4-pack（FP4 group32）                            | `W4A4_MXFP4`             |

* 使用方式：模型已带 `quantization_config` 时无需任何参数，自动检测。

#### 5.1.3 `fp8`（原生 FP8 checkpoint）

* 适配官方发布的 FP8 模型（如 Qwen3.8-27B-FP8、GLM-5.3-Flash）：`float8_e4m3fn` 权重 + 每 block 一个 FP32 scale（`weight_scale_inv`）。
* ​**仅支持 block-wise**​：无 `weight_block_size`（per-tensor/per-channel scale）的 FP8 checkpoint 会直接报错，需改用 ModelSlim / LLM-Compressor 重量化。
* `ignored_layers` / `modules_to_not_convert` 中的层回退非量化路径。
* ​**硬件自适应**​（`fp8_block.py`）：
  * ​**Ascend 950**​：加载时把 block scale 重组为 MXFP8，权重保持 1 字节/元素，走原生 FP8 matmul。
  * ​**其他昇腾芯片**​：加载时将 block scale 解析回模型 dtype（BF16 matmul 执行），数值等价但权重内存约 2 倍。
* `deepseek_v4_fp8` 是 DeepSeek V4 专属变体：linear 走 `FP8/ds_linear`（128×128 block），MoE 专家层 `expert_dtype == "fp4"` 时走 `FP8/ds_w4a8_moe`。

#### 5.1.4 `mxfp8`（ModelOpt）

* 复用上游 `ModelOptMxFp8Config` 的解析逻辑（排除模块、checkpoint 布局），仅替换执行方法为 Ascend 的 `W8A8_MXFP8` Scheme；vision 模块（`vision_tower` 等）自动回退非量化。

### 5.2 全量量化Method（Scheme）清单

以下按 `layer_type` 分类，汇总 `@register_scheme` 注册的全部实现（源码 `vllm_ascend/quantization/methods/`）。

#### 5.2.1 Linear 层（12 个）

| quant_type                        | 实现类                                            | 权重                   | 激活                          | 说明                                                                 |
| ------------------------------------ | --------------------------------------------------- | ------------------------ | ------------------------------- | ---------------------------------------------------------------------- |
| `W8A8`                         | `AscendW8A8LinearMethod`                      | INT8 per-channel       | INT8 per-tensor**静态** | 走`torch.ops.vllm.quantize`+`npu_quant_matmul`（deq_scale） |
| `W8A8_DYNAMIC`                 | `AscendW8A8DynamicLinearMethod`               | INT8 per-channel       | INT8 per-token**动态**  | `npu_dynamic_quant`；加载后权重转 NZ 格式加速                    |
| `W8A8FP8_DYNAMIC`              | `AscendW8A8FP8DynamicLinearMethod`            | FP8 per-channel        | FP8 per-token 动态            | 继承 W8A8_DYNAMIC，换`act_quant_type`                           |
| `W8A16`                        | `AscendW8A16LinearMethod`                     | INT8 per-channel       | BF16/FP16 不量化              | 权重加载后反量化                                                     |
| `W8A8_MXFP8`                   | `AscendW8A8MXFP8DynamicLinearMethod`          | MXFP8                  | MXFP8                         | MX 标准，E8M0 scale，`npu_dynamic_mx_quant`                      |
| `W8A8_MIX`                     | `AscendW8A8PDMixLinearMethod`                 | INT8                   | INT8                          | PD 分离部署：Prefill 动态 + Decode 静态混合量化                      |
| `W4A8_MXFP`                    | `AscendW4A8MXFPDynamicLinearMethod`           | MXFP4                  | MXFP8                         | 权重 FP4、激活 MXFP8                                                 |
| `W4A4_MXFP4`                   | `AscendW4A4MXFP4DynamicLinearMethod`          | MXFP4（uint8 打包）    | MXFP4                         | MX 标准 4bit；支持 TP 边界未对齐 MX 组                               |
| `W4A4_DYNAMIC`                 | `AscendW4A4LaosDynamicLinearMethod`           | INT4 per-channel       | INT4 per-token                | LAOS 动态 4bit 方案                                                  |
| `W4A4_FLATQUANT_DYNAMIC`       | `AscendW4A4FlatQuantDynamicLinearMethod`      | INT4（Kronecker 变换） | INT4 per-token                | FlatQuant：先用可学习正交变换平滑激活分布再量化                      |
| `W4A4_MXFP4_FLATQUANT_DYNAMIC` | `AscendW4A4MXFP4FlatQuantDynamicLinearMethod` | MXFP4 + Kronecker      | MXFP4                         | FlatQuant 与 MXFP4 结合                                              |
| `fp8`（block-wise）            | `AscendFp8BlockLinearMethod`                  | FP8 block + FP32 scale | FP8 动态                      | 原生 FP8 checkpoint；950 上重组为 MXFP8，其余硬件反量化为 BF16       |
| `fp8`/`ds_linear`          | `AscendW8A8MXFP8DSDynamicLinearMethod`        | FP8 block 128×128     | FP8 动态                      | DeepSeek V4 原生布局专用                                             |

#### 5.2.2 MoE 层（FusedMoE / RoutedExperts，10 个）

| quant_type                 | 实现类                                      | 权重                             | 激活                | 说明                                                                     |
| ----------------------------- | --------------------------------------------- | ---------------------------------- | --------------------- | -------------------------------------------------------------------------- |
| `W8A8_DYNAMIC`          | `AscendW8A8DynamicFusedMoEMethod`       | INT8 per-channel                 | INT8 per-token 动态 | 支持 EPLB；gmm1+silu+quant 融合路径                                      |
| `W8A8FP8_DYNAMIC`       | `AscendW8A8FP8DynamicFusedMoEMethod`    | FP8 per-channel                  | FP8 per-token 动态  |                                                                          |
| `W8A8_MXFP8`            | `AscendW8A8MXFP8DynamicFusedMoEMethod`  | MXFP8                            | MXFP8               | ModelOpt / LLM-Compressor MXFP8                                          |
| `W4A8_DYNAMIC`          | `AscendW4A8DynamicFusedMoEMethod`       | INT4 per-channel                 | INT8 per-token 动态 | 兼容 ModelSlim 与 LLM-Compressor 两种权重布局；TP>16 时有 scale 布局适配 |
| `W4A8_MXFP`             | `AscendW4A8MXFPDynamicFusedMoEMethod`   | MXFP4                            | MXFP8               |                                                                          |
| `fp8`/`ds_w4a8_moe` | `AscendW4A8MXFPDSDynamicFusedMoEMethod` | MXFP4（DS 原生）                 | MXFP8               | DeepSeek V4`expert_dtype=fp4`                                        |
| `W4A16`                 | `AscendW4A16FusedMoEMethod`             | INT4 per-group（int32 打包 ×8） | 不量化              | LLM-Compressor 布局（如 Kimi-K2-Thinking）；加载后 unpack→转置→repack  |
| `W4A16_MXFP4`           | `AscendW4A16MXFP4FusedMoEMethod`        | MXFP4                            | 不量化              |                                                                          |
| `W4A4_MXFP4`            | `AscendW4A4MXFP4DynamicFusedMoEMethod`  | MXFP8                            | MXFP8               |                                                                          |
| `fp8`（block-wise）     | `AscendFp8BlockFusedMoEMethod`          | FP8 block                        | FP8 动态            | 原生 FP8 checkpoint 的专家层                                             |

## 6 转测模型拆解

### GLM5.1

[模型性能拆解.xlsx](https://onebox.huawei.com/v/acc42dff262ca76ac7e0fbdd891ee213?type=1&sheet=GLM5.1%20744B%E9%AB%98%E4%BD%8E%E7%B2%BE)

### Qwen3.5-397B

[模型性能拆解.xlsx](https://onebox.huawei.com/v/acc42dff262ca76ac7e0fbdd891ee213?type=1&sheet=Qwen3.5%20397B%E9%AB%98%E4%BD%8E%E7%B2%BE)
