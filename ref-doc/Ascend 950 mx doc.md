## 1. 量化手段：从 INT8 到 MX 的谱系

量化（Quantization）是把浮点权重/激活（BF16/FP16）压缩成更低精度的表示，用「数值精度换内存/算力」。按**权重 W、激活 A 的位宽与格式**，主流谱系如下（对应 `vllm_ascend/quantization/quant_type.py` 的 `QuantType`）：

```
位宽越来越高（精度↑，压缩率↓）
W8A8 (INT8)   ── W4A8 (INT4/INT8) ── W4A16 (INT4/BF16) ── W8A16 (INT8/BF16) ── BF16(不量化)
                        │
                        └─> FP8 家族（8 bit，带指数）
                               ├─ 普通 FP8（block-wise/tensor scale，float32 共享 scale）
                               └─ MXFP8（Microscaling：E4M3 + E8M0 每 32 元素一个共享指数）# A5 主打
              更激进的 4-bit MX：W4A8MXFP / W4A4MXFP / W4A16MXFP
```

- **整数量化（INT8/INT4）**：把浮点缩放到 int 区间，用 per-channel/token 的 **float32 scale**。通用、成熟，但 scale 粒度粗（每通道一个），对离群值敏感。
- **FP8**：8 bit 浮点（E4M3/E5M2），本身带指数动态范围。Ascend 用 **block-wise FP8**（`weight_block_size` 如 128×128 一个 float32 scale）。
- **MXFP8（微缩放）**：**数据仍存 8bit（E4M3），但 scale 用 E8M0（纯指数、无尾数），每 32 个元素一个共享指数**。scale 粒度高 → 比 block-FP8 更能抵御离群值、精度更稳，同时在 A5 硬件上有专门的 `npu_quant_matmul` / `npu_grouped_matmul` 原生支持。**这就是本文的主角**。

> 一个模型「具体走哪种手段」由 `model_config.quantization` 的名字（`"mxfp8"`/`"fp8"`/`"ascend"`/`"deepseek_v4_fp8"`…）决定，见第 11.6 节；每种名字落到对应 Config 与 quant_method（第 11.1 节）。

## 2. 主流量化格式：INT8 / FP8 / MXFP8

| 维度 | **INT8（W8A8/W4A8）** | **Block-FP8（fp8）** | **MXFP8（mxfp8）** |
|---|---|---|---|
| 数据位宽 | 8（INT）/ 4（W4） | 8（FP8 E4M3） | 8（FP8 E4M3） |
| scale 类型 | float32（per-channel/token） | float32（每 block，如 128×128 一个） | **E8M0（纯指数），每 32 元素一个** |
| scale 粒度 | 粗（每通道） | 中（每 block） | **细（每 group 32）** |
| 离群值鲁棒性 | 低 | 中 | **高**（离群值不会打垮整个 block） |
| A5 原生算子 | int GEMM | `npu_quant_matmul`（普通 FP8） | `npu_dynamic_mx_quant` + `npu_quant_matmul`（group_sizes） |
| A5 上走向 | 保持非 MX（整数路径） | **被 `is_950()` 再量化成 MXFP8** | **原生 MX（直接走）** |
| 底层 | `torch_npu.npu_quant_matmul` / int8 GMM | `torch_npu.npu_quant_matmul` | `npu_dynamic_mx_quant`(E8M0) + `npu_grouped_matmul` |

**MX 的 E8M0 指数怎么来（核心）**：`npu_dynamic_mx_quant`（CANN `DynamicMxQuantV3`）对每个 group（32 元素）取最大绝对值 → 推一个共享 E8M0 指数 → 把组内数缩到 E4M3。`scale_alg=1` 时先映射到 FP8 范围再把指数向上取整防溢出（MiniMax-M3 用）；`scale_alg=0` 直接用 max 推指数。详见第 7.5 节。

## 3. 量化优势

1. **显存/带宽减半或更多**：BF16（2 字节）→ INT8/FP8（1 字节）→ INT4/MXFP4（0.5 字节）。FP8 权重在 950 上经 MX 折叠后仍 1 字节/元素，**让大模型（如 27B FP8）能塞进单卡**（`fp8_block.py` 头注释：footprint 从 BF16 的 2 字节降到 1 字节）。
2. **吞吐/算力提升**：低位宽的 `npu_quant_matmul` / `npu_grouped_matmul` 在 A5 硬件上有更高 TOPS，且减少带宽瓶颈下的等待。
3. **MX 相对普通 FP8 的精度优势**：**E8M0 每 32 元素一个共享指数**，比 block-FP8（128×128 一个 float32 scale）粒度细得多——**一个极端离群值只会影响它所在的 32 元素 group，不会压低整个 block 的有效位**，因此 MX 在长尾/离群分布（权重、激活峰值）下精度更稳。这是 A5 把它当「推荐低精路径」的根本原因。
4. **省去离线重量化**：普通 block-FP8 checkpoint 在 A5 上由 `is_950()` **运行时自动再量化为 MXFP8**（`fp8_block.py`），无需额外离线转换或按模型打补丁。
5. **KV Cache 也可低精**：`C8`/INT8 注意力等方案可同时压 KV cache 显存（`kv_c8.py`），进一步省内存。

> 代价/注意：量化都有精度损失，尤其在**极小 group_size 的极端激进度（W4A4）**下；A5 上若某层 reduction 不是 32 的整数倍会**回退到模型 dtype**（见 7.5.3 / 排查清单），并非所有层都保证命中 MX。

## 4. 量化工具ModelSlim使用

量化基础知识与ModelSlim工具介绍：
https://onebox.huawei.com/v/da877a7be6b194d1459a64350ce9f547

ModelSlim入门：
https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/quick_start/quantization_quick_start.md#msmodelslim-%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8
https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md#%E6%9D%83%E9%87%8D%E9%87%8F%E5%8C%96%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97

常见模型量化实践：
https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_flash_w4a4c8.yaml
https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/glm_5_2/glm_5_3_w8a8c8_mxfp8.yaml

## 5. Ascend 950下vllm-ascend如何走入MX分支

该部分的前提是已经有了量化权重，主要解释在服务启动后，如何吧权重的配置读取到，在接收请求时，如何走到MX排布的分支。

**整体链路总结**：

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
             └─ 每个 quant_config 决定"这层返回哪个 scheme"（linear/moe/attention + 是哪个 WxA?）
             └─ 返回的 quant_method（scheme）就是"这层的量化实现者"

step3  权重创建 / 布局 / 前向，全由该 quant_method 管
        ├─ create_weights() / get_weight()：按该方案的布局建 weight/scale
        ├─ process_weights_after_loading()：转置、配对、再量化（如 fp8→MXFP8）
        └─ apply()：前向调用真实算子（如 npu_dynamic_mx_quant + npu_quant_matmul）
```

## 5.1 阶段一：命令启动 → 注册Ascend 量化 Config

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

```
from vllm_ascend.quantization import (
   AscendCompressedTensorsConfig,
   AscendFp8Config,
   AscendModelOptMxFp8Config,
   AscendModelSlimConfig,
)
```

**核心步骤**：每个 `@register_quantization_config("xxx")` 会把 "xxx"对应的Config 类写入 `vllm/model_executor/layers/quantization/__init__.py`的`_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`， 最终在 `get_quantization_config()`里`method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)`覆盖同名的vllm内置配置，将vllm-ascend实现替换vllm原生实现。

**解耦点**：

- 名字→Config 的**查找**由 ✅ vllm 的 `get_quantization_config` 完成，但 **Config 类实现**是 🔧 vllm-ascend 注册覆盖的。
- 解析动作全部在 **Engine 主进程的 `VllmConfig.__post_init__`** 完成（flag：名字字符串）；
- 真正落到每层的 MX 适配器在 **Worker 的模型构造**阶段（flag：`get_quant_method` + `is_950()`），均为 🔧 vllm-ascend 实现。

## 5.2 阶段二：Engine 主进程构建 VllmConfig → 解析模型配置，重建model_config.quantization、vllm_config.quant_config

> - **vllm-ascend**：实现了2个hook（`NPUPlatform.apply_config_platform_defaults` / `check_and_update_config`）以及**自动探测逻辑**（`maybe_auto_detect_quantization` / `detect_quantization_method`）。

```
EngineCore / LLMEngine 启动
  → vllm/vllm/engine/arg_utils.py:1906
      create_engine_config()
        → self.create_model_config()                          # DeepSeek V4特殊分支
            → ModelConfig.__post_init__  (vllm/vllm/config/model.py)
                └─ _verify_quantization()  (vllm/vllm/config/model.py)
                    ├─ 读 config.json 的 quantization_config（hf_quant_config）
                    ├─ 遍历 overrides 列表（含 "deepseek_v4_fp8" 等）
                    │     调 get_quantization_config(name).override_quantization_method(quant_cfg, self.quantization, hf_config)
                    │     → 命中则 model_config.quantization = 改写后的名字（如 "fp8"→"deepseek_v4_fp8"）
                    ├─ 若用户显式 --quantization 且未被 override 命中 → 保留用户值
                    └─ 兜底：quantization 仍为 None → = config.json 的 quant_method
                  # 此时 model_config.quantization 已确定（override/用户/config.json）
        → VllmConfig构造
            → VllmConfig.__post_init__  (vllm/vllm/config/vllm.py:972) 
                ├─ (1) 第一次解析  line 1028:
                │      if self.quant_config is None:
                │          self.quant_config = VllmConfig._get_quantization_config(model_config, load_config)
                ├─ (2) 平台 hook   line 1270: current_platform.apply_config_platform_defaults(self)   # cudagraph_capture_size的调整，和mx无关
                └─ (3) 平台 hook   line 1488: current_platform.check_and_update_config(self)          # 量化相关实现
                        → vllm-ascend/vllm_ascend/platform.py  NPUPlatform.check_and_update_config
                            ├─ 第 3 步  maybe_auto_detect_quantization(vllm_config)   ← 仅当 model_config.quantization 仍为 None（override/用户都没定）才自动探测+重建
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

探测 + 重建代码

```python
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

集中config配置示例：

1. quant_model_description.json

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090812756467/51263522/3d9a1b2e8aec499780ac85555da8f8ce.png)
2. config.json，quantization_config.quant_method为"fp8"

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090812756467/51263907/e88416bde54d49879c1a640ad47dbbba.png)

目前已有的这几个model_config.quantization和对应的量化method的映射关系：

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

## 5.3 阶段三：加载模型 → 给每个 Layer 落 MX quant_method

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

## 5.4 阶段四：前向执行

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

## 6. 量化方法全览

前几章介绍了MX量化的整理流程，包含quantization_config的注册和选择，前向执行时各层的MX quant_method如何使用，这里介绍一下所有量化方法的归档路径和区别：

### 6.1 quant method 各自的区别

vllm-ascend 的量化合集在 `vllm_ascend/quantization/methods/`，每个文件里的**方案类（scheme）**都通过 `@register_scheme(quant_type: str, layer_type: str)` 注册。
**第一维度**是 `quant_type`——它描述「**权重 W 和激活 A 各是什么位宽/格式**」：

```
W8A8       W 和 A 都是 INT8（整数）
W4A8       W 是 INT4，A 是 INT8
W4A16      W 是 INT4，A 是 BF16/FP16
W8A16      W 是 INT8，A 是 BF16/FP16
W8A8FP     W 和 A 都是 FP8（如上 F8-Dynamic）
W8A8MXFP   W 和 A 都是 MXFP8（E4M3 + E8M0 共享指数）   ← A5 主打
W4A8MXFP   W 是 MXFP4，A 是 MXFP8                       ← DSV4 MoE
W4A16MXFP  W 是 MXFP4，A 是 BF16/FP16
W4A4MXFP   W 和 A 都是 MXFP4
```

**第二维度**是`layer_type`：同一个量化方式会有 `linear`/`moe`/`attention` 三种 scheme，分别处理普通线性层、MoE、attention，因为它们的权重布局和调用算子不同。

`is_mx_quant_type`（`methods/__init__.py:68`）明确区分 **MX 家族** vs **非 MX 家族**：

| 文件 | 注册的 scheme | W/A | 是否 MX | 说明 |
|---|---|---|---|---|
| `w8a8_mxfp8.py` | `W8A8_MXFP8`（linear/moe）、`FP8 ds_linear`、`FP8 ds_w4a8_moe` | FP8/E8M0 | ✅ MX | A5 线性层/DSV4 主路径 |
| `w4a8_mxfp4.py` | `W4A8_MXFP`（linear/moe） | FP4/FP8 | ✅ MX | |
| `w4a4_mxfp4.py` | `W4A4_MXFP4`（linear/moe） | FP4/FP4 | ✅ MX | |
| `w4a4_mxfp4_flatquant.py` | `W4A4_MXFP4_FLATQUANT` | FP4/FP4 | ✅ MX | flatquant 稀疏 |
| `w4a16_mxfp4.py` | `W4A16_MXFP4`（moe） | FP4/BF16 | ✅ MX | |
| `w4a4_flatquant.py` | `W4A4_FLATQUANT_DYNAMIC` | INT4/INT8? | ❌ 非 MX | flatquant |
| `w4a4_laos_dynamic.py` | `W4A4_DYNAMIC` | INT4 | ❌ 非 MX | LAOS |
| `w8a8_dynamic.py` | `W8A8_DYNAMIC`（linear/moe） | INT8 | ❌ 非 MX | |
| `w8a8_static.py` | `W8A8`（linear） | INT8 | ❌ 非 MX | 静态 |
| `w8a8fp8_dynamic.py` | `W8A8FP8_DYNAMIC`（linear/moe） | FP8 | ❌ 非 MX | 动态 FP8 |
| `w8a8_pdmix.py` | `W8A8_MIX`（linear） | INT8 | ❌ 非 MX | PDMix |
| `w8a16.py` | `W8A16`（linear） | INT8/BF16 | ❌ 非 MX | |
| `w4a16.py` | `W4A16`（moe） | INT4/BF16 | ❌ 非 MX | |
| `w4a8.py` | `W4A8_DYNAMIC`（moe） | INT4/INT8 | ❌ 非 MX | |
| `kv_c8.py` | `FAKQuant` / `INT8_DYNAMIC`（attention） | — | ❌ 非 MX | 注意力 KV 量化 |

> **关键区别**：MX 家族用 **E8M0 共享指数**做 scale（`float8_e8m0fnu`），非 MX 家族用 **tensor/per-channel/per-token 的 float32 scale**（INT8）或普通 FP8 scale。MX 的 scale 粒度是「每 32 个元素一个指数」（group_size=32），非 MX 是「每通道一个」。

### 6.2 Ascend 950 下是否还有普通的非 MX 量化模式

**会有。** MX 是 A5 的「特色/推荐低精路径」，但**不是唯一、也不强制**。具体分三类：

- **FP8 类 → 在 A5 被「转」成 MX**：`"fp8"` 名字落到 `AscendFp8Config`，其方案 `AscendFp8BlockLinearMethod` 里 `is_950()` 为真时会把权重 `_mx_quantize` `再量化成 MXFP8`（`fp8_block.py:143`）。所以「普通 FP8 权重」在 A5 上实际也会变成 MX 低精执行（这正是 FP8 checkpoint 在 950 上能压缩到单卡的原因）。
- **INT 类 → 保持非 MX**：`W8A8_DYNAMIC`（INT8）、`W4A8_DYNAMIC`、`W4A16`、`W8A16` 这些 INT 方案在 A5 上**不会自动转 MX**，仍走整数 GEMM。证据在 `device_op.py` 的 `A5DeviceAdaptor.npu_dynamic_quant`（:1141）：当 `use_mxfp_quant=False` 时走 `BaseDeviceAdaptor.npu_dynamic_quant` → `torch_npu.npu_dynamic_quant`（普通 INT8/FP8 动态量化，非 MX）。
- **显式选择**：用户 `--quantization` 指定的方法优先；只有 FP8/config 这类才在 A5 上做了「隐性转 MX」。

> 结论：A5 同时支持 MX 与非 MX；**MX 的「自动/推荐」体现在 FP8 权重会被 `is_950()` 折算成 MXFP8**，而 INT8/W4A16 等仍按各自整数路径跑。所以一个模型在 950 上「是否走 MX」要看你用的是哪种量化位宽方案，不是 A5「一律 MX」。
