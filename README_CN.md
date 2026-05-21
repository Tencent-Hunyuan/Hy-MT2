<p align="left">
   <a href="README.md">English</a>&nbsp;｜&nbsp;中文
</p>
<br>

<p align="center">
 <img src="imgs/logo-zh.png" width="400"/> <br> 
</p>

<div align="center" style="line-height: 1;">


[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Tencent%20Hy-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/collections/tencent/hy-mt2)
&nbsp;&nbsp;
[![ModelScope](https://img.shields.io/badge/ModelScope-Tencent%20Hy-624aff)](https://modelscope.cn/collections/Tencent-Hunyuan/Hy-MT2)

</div>

<p align="center">
    🖥️&nbsp;<a href="https://aistudio.tencent.com/llm/zh?tabIndex=0"><b>官方网站</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    💬&nbsp;<a href="https://github.com/Tencent-Hunyuan/Hy-MT2"><b>GitHub</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🪡&nbsp;<a href="https://github.com/Tencent/AngelSlim/tree/main"><b>AngelSlim</b></a></p>


## 模型介绍


Hy-MT2 是一款面向真实复杂场景的“快思考”多语言翻译模型家族，涵盖 1.8B、7B 和 30B-A3B（MoE）三种体量，支持 33 种语言互译并具备强大的多语言指令遵循能力。在端侧部署上，得益于 AngelSlim 1.25-bit 极端量化，其 1.8B 模型仅需 440MB 存储空间，推理速度显著提升 1.5 倍。多维度评测表明，Hy-MT2 在通用、真实业务、专业领域及指令遵循等翻译任务中表现卓越：7B 和 30B-A3B 模型性能不仅超越了 DeepSeek-V4-Pro、Kimi K2.6 等开源模型在快思考模式下的表现，轻量级 1.8B 模型亦在整体上超越了微软和豆包等主流商业 API。

同时，本次我们也开源了一个针对翻译指令遵循能力的评测集[IFMTBench](./IFMTBench/README_zh.md)。

也欢迎大家使用我们发布的 Hy-MT2-Translator Skill，可以方便接入Hy-MT2系列模型完成翻译任务，下载链接[ClawHub](https://clawhub.ai/tencent-adm/hy-mt2-translator-skill)和[SkillHub](https://skillhub.cn/skills/hy-mt2-translator)。

## 新闻

* 2026.5.21  我们在HuggingFace和ModelScope上开源了 **Hy-MT2-1.8B**/**Hy-MT2-7B**/**Hy-MT2-30B-A3B**/**IFMTBench**
* 2025.12.30 我们在HuggingFace和ModelScope开源了 **HY-MT1.5-1.8B**和**HY-MT1.5-7B**
* 2025.9.1 我们在HuggingFace和ModelScope开源了 **Hunyuan-MT-7B**和**Hunyuan-MT-Chimera-7B**。


## 效果
<div align='center'>
<img src="imgs/main_result.png" width = "100%" />
</div>

更多的实验效果和分析可以参考我们的[报告](./HY_MT2_0_Report.pdf)。

&nbsp;

## 模型链接
| 模型名  | 简介 | 下载链接 |
| ----------- | ----------- |-----------
| Hy-MT2-1.8B  | 混元1.8B翻译模型 |🤗 [Model](https://huggingface.co/tencent/Hy-MT2-1.8B)|
| Hy-MT2-1.8B-FP8 | 混元1.8B翻译模型，fp8量化    | 🤗 [Model](https://huggingface.co/tencent/Hy-MT2-1.8B-FP8)|
| Hy-MT2-1.8B-GGUF | 混元1.8B翻译模型， llama.cpp    | 🤗 [Model](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)|
| Hy-MT2-1.8B-2bit-GGUF | 混元1.8B翻译模型， llama.cpp, 2bit    | 🤗 [Model](https://huggingface.co/tencent/Hy-MT2-1.8B-2bit-GGUF)|
| Hy-MT2-1.8B-1.25bit-GGUF | 混元1.8B翻译模型， llama.cpp, 1.25bit    | 🤗 [Model](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25bit-GGUF)|
| Hy-MT2-7B | 混元7B翻译模型    | 🤗 [Model](https://huggingface.co/tencent/Hy-MT2-7B)|
| Hy-MT2-7B-FP8 | 混元7B翻译模型，fp8量化     | 🤗 [Model](https://huggingface.co/tencent/Hy-MT2-7B-FP8)|
| Hy-MT2-7B-GGUF | 混元7B翻译模型， llama.cpp    | 🤗 [Model](https://huggingface.co/tencent/Hy-MT2-7B-GGUF)|
| Hy-MT2-30B-A3B | 混元30B-A3B翻译模型    | 🤗 [Model](https://huggingface.co/tencent/Hy-MT2-30B-A3B)|
| Hy-MT2-30B-A3B-FP8 | 混元30B-A3B翻译模型，fp8量化     | 🤗 [Model](https://huggingface.co/tencent/Hy-MT2-30B-A3B-FP8)|



## Hy-MT2 翻译任务指令示例（中英文对照）

*注意：下面的source_lang和target_lang都使用语言的全称，中文使用中文全称，英文使用英文全称。*

| Type | Chinese prompt | English prompt |
|---|---|---|
| **Default Translation** | 将以下文本翻译为 `{target_lang}`，注意**只需要输出翻译后的结果，不要额外解释**：<br><br>`{source_text}` | Translate the following text into `{target_lang}`. Note that you should **only output the translated result without any additional explanation**:<br><br>`{source_text}` |
| **Terminology** | *参考下面的翻译：*<br>`{text}` 翻译成 `{text}`<br>`{text}` 翻译成 `{text}`<br>`{text}` 翻译成 `{text}`<br>将以下文本翻译为 `{target_lang}`，注意**只需要输出翻译后的结果，不要额外解释**：<br><br>`{source_text}` | *Reference the following translations:*<br>`{text}` translates to `{text}`<br>`{text}` translates to `{text}`<br>`{text}` translates to `{text}`<br><br>Translate the following text into `{target_lang}`. Note that you must **ONLY output the translated result without any additional explanation**:<br><br>`{source_text}` |
| **Style** | 请将以下文本翻译为 `{target_lang}`。<br>注意翻译的风格要严格符合【**`{target_style}`**】<br><br>`{source_text}` | Please translate the following text into `{target_lang}`. Note that the translation style must strictly conform to [**`{target_style}`**]:<br><br>`{source_text}` |
| **Personalization** | *【待翻译文本】*<br>`{source_text}`<br><br>*【翻译任务】*<br>1、**`{user_preferences}`**<br>2、**`{user_preferences}`**<br>3、……<br>4、将【待翻译文本】翻译为 `{target_lang}`。 | *[Source Text]*<br>`{source_text}`<br><br>*[Translation Tasks]*<br>1. **`{user_preferences}`**<br>2. **`{user_preferences}`**<br>3. ...<br>4. Translate the [Source Text] into `{target_lang}`. |
| **Delimiters** | 请将以下文本准确翻译为 `{target_lang}`。<br>你必须在译文中**保留等量的分隔符，绝对不可遗漏、转义或翻译该符号，并注意分隔符的位置**。<br><br>`{source_text}` | Please accurately translate the following text into `{target_lang}`.<br>You must **retain the exact same number of delimiters in the translation. Strictly do not omit, escape, or translate these symbols, and pay close attention to their placement**.<br><br>`{source_text}` |
| **Structured Data 1** | *# 任务目标*<br>将下方 `{source_text}` 中的 `{format_type}` 格式数据翻译为 `{target_lang}`。<br><br>*# 严格约束*<br>1. **结构锁定**：绝对保持原有的 `{format_type}` 数据结构、缩进和层级完全不变。<br>2. **选择性翻译**：仅翻译面向用户展示的可见文本内容。<br>3. **禁止修改**：**严禁**翻译或更改任何代码标签、键名 (Key)、变量占位符（如 `{{var}}`、`${var}`、`%s`、`%d` 等）或代码属性。<br><br>*# 数据输入*<br>`{source_text}` | *### Task*<br>Translate the user-facing text within the following `{format_type}` data into `{target_lang}`.<br><br>*### Strict Rules*<br>1. **Structure Preservation:** You MUST preserve the original `{format_type}` data structure, nesting, hierarchy, and indentation exactly as they are.<br>2. **Selective Translation:** Translate ONLY the visible, user-facing text content/values.<br>3. **Strict Non-Translation:** NEVER translate or alter code tags, keys, properties, object names, or variable placeholders. Leave them exactly in their original English/code form.<br><br>*### Source Data*<br>`{source_text}` |
| **Structured Data 2** | *【背景信息】*<br>`{background_text}`<br><br>请结合背景信息将以下文本翻译为 `{target_lang}`。<br><br>*【待翻译文本】*<br>`{source_text}` | *[Background Information]*<br>`{background_text}`<br><br>Please translate the following text into `{target_lang}`, taking the provided background information into consideration.<br><br>*[Source Text]*<br>`{source_text}` |

---

## 推理和部署
### transformers

transformers>=5.6.0

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "tencent/Hy-MT2-30B-A3B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

# Example inference
prompt = "将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n今天天气真好。"
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
    )
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

### vllm

Build vLLM from source:
```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install --editable . --torch-backend=auto
```

Start the vLLM server:

```bash
vllm serve tencent/Hy-MT2-30B-A3B --tensor-parallel-size 1
```

### sglang

Build SGLang from source:
```bash
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install "transformers>=5.6.0"
pip3 install -e "python"
```

Launch SGLang server:

```bash
python3 -m sglang.launch_server --model tencent/Hy-MT2-30B-A3B --tp 1
```


### llama_cpp
**❕❕ This gguf depends on our STQ kernel, which is released at [PR #22836](https://github.com/ggml-org/llama.cpp/pull/22836).**

#### Clone llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
```

#### Enter the llama.cpp folder

```bash
cd llama.cpp
```

#### Build llama.cpp

```bash
cmake -B build
cmake --build build --config Release
```

#### Run a completion example

```bash
./build/bin/llama-completion \
  --model model.gguf  \
  -p "Translate the following segment into Chinese, without additional explanation：Hello" \
  --jinja \
  -ngl 0 \
  -n 64 -st 
```

#### Run the llama.cpp benchmark

```bash
./build/bin/llama-bench -m model_zoo/model.gguf  -ngl 0
```


对于1.8B和7B，我们推荐使用下面这组参数进行推理。注意，我们的模型没有默认 system_prompt。

```json

{
  "temperature": 0.7,
  "top_p": 0.6,
  "top_k": 20,
  "repetition_penalty": 1.05,
  "max_tokens": 4096
}
```

对于30B-A3B，我们推荐使用下面这组参数进行推理。注意，我们的模型没有默认 system_prompt。

```json

{
  "temperature": 0.7,
  "top_p": 1.0,
  "top_k": -1,
  "repetition_penalty": 1.0,
  "max_tokens": 4096
}
```


## 模型训练
Hy-MT2提供了完整的模型训练流程，支持全量微调和 LoRA 微调，同时支持 DeepSpeed ZeRO 多种配置以及 LLaMA-Factory 集成。

详细的训练文档请参考：[模型训练指南](./train/README_CN.md)

## 量化工具

我们提供了 [AngelSlim](https://github.com/tencent/AngelSlim)——一套易用、全面、高效的大模型压缩工具包，涵盖常用量化算法、低比特量化和投机采样等能力。


## 支持的语种
| Languages         | Abbr.   | Chinese Names   |
|-------------------|---------|-----------------|
| Chinese           | zh      | 中文            |
| English           | en      | 英语            |
| French            | fr      | 法语            |
| Portuguese        | pt      | 葡萄牙语        |
| Spanish           | es      | 西班牙语        |
| Japanese          | ja      | 日语            |
| Turkish           | tr      | 土耳其语        |
| Russian           | ru      | 俄语            |
| Arabic            | ar      | 阿拉伯语        |
| Korean            | ko      | 韩语            |
| Thai              | th      | 泰语            |
| Italian           | it      | 意大利语        |
| German            | de      | 德语            |
| Vietnamese        | vi      | 越南语          |
| Malay             | ms      | 马来语          |
| Indonesian        | id      | 印尼语          |
| Filipino          | tl      | 菲律宾语        |
| Hindi             | hi      | 印地语          |
| Traditional Chinese | zh-Hant| 繁体中文        |
| Polish            | pl      | 波兰语          |
| Czech             | cs      | 捷克语          |
| Dutch             | nl      | 荷兰语          |
| Khmer             | km      | 高棉语          |
| Burmese           | my      | 缅甸语          |
| Persian           | fa      | 波斯语          |
| Gujarati          | gu      | 古吉拉特语      |
| Urdu              | ur      | 乌尔都语        |
| Telugu            | te      | 泰卢固语        |
| Marathi           | mr      | 马拉地语        |
| Hebrew            | he      | 希伯来语        |
| Bengali           | bn      | 孟加拉语        |
| Tamil             | ta      | 泰米尔语        |
| Ukrainian         | uk      | 乌克兰语        |
| Tibetan           | bo      | 藏语            |
| Kazakh            | kk      | 哈萨克语        |
| Mongolian         | mn      | 蒙古语          |
| Uyghur            | ug      | 维吾尔语        |
| Cantonese         | yue     | 粤语            |



## Citing Hy-MT2

```bibtex
@misc{hy-mt1.5,
      title={HY-MT1.5 Technical Report}, 
      author={Mao Zheng and Zheng Li and Tao Chen and Mingyang Song and Di Wang},
      year={2025},
      eprint={2512.24092},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.24092}, 
}
```

## 联系我们
如果你想给我们的研发和产品团队留言，欢迎联系我们腾讯混元LLM团队。你可以通过邮件（hunyuan_opensource@tencent.com）联系我们。
