# Embformer: An Embedding-Weight-Only Transformer Architecture


<p align="center">
  <a href="https://doi.org/10.5281/zenodo.15736957">
    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15736957-blue.svg" alt="DOI"></a>
  <a href="https://huggingface.co/spaces/HighCWu/embformer-demo"><img src="https://img.shields.io/badge/HF%20Spaces-🤗-yellow" alt="demo"></a>
  <a href="https://huggingface.co/collections/HighCWu/embformer-minimind-685be74dc761610439241bd5"><img src="https://img.shields.io/badge/Model-🤗-yellow" alt="model"></a>
  <a href="https://discord.gg/zcJszfPrZs"><img src="https://img.shields.io/discord/857781525553741874" alt="Discord"></a>
  <a href="https://qm.qq.com/q/nZjcgtvfry"><img src="https://img.shields.io/badge/QQ-1044867291-0086F9" alt="Discord"></a>
</p>


This is the official implementation of [Embformer: An Embedding-Weight-Only Transformer Architecture](https://doi.org/10.5281/zenodo.15736957).

## Getting Started

Run commands in the terminal:
```sh
pip install -r requirements.txt
```

The following contains a code snippet illustrating how to use the model generate content based on given inputs.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "HighCWu/Embformer-MiniMind-Base-0.1B"
# model_name = "HighCWu/Embformer-MiniMind-Seqlen512-0.1B"
model_name = "HighCWu/Embformer-MiniMind-0.1B"
# model_name = "HighCWu/Embformer-MiniMind-RLHF-0.1B"
# model_name = "HighCWu/Embformer-MiniMind-R1-0.1B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=".cache"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    cache_dir=".cache"
)

# prepare the model input
prompt = "请为我讲解“大语言模型”这个概念。"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    input_ids=model_inputs['input_ids'],
    attention_mask=model_inputs['attention_mask'],
    max_new_tokens=8192
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

## Training

The current training code is modified based on [MiniMind](https://github.com/jingyaogong/minimind) and integrated in the submodule `embformer-minimind`

## License Agreement

All our open-weight models are licensed under Apache 2.0. You can find the license files in the respective Hugging Face repositories.

## Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@manual{wu_2025_15736957,
  title        = {Embformer: An Embedding-Weight-Only Transformer
                   Architecture
                  },
  author       = {Wu, Hecong},
  month        = jun,
  year         = 2025,
  doi          = {10.5281/zenodo.15736957},
  url          = {https://doi.org/10.5281/zenodo.15736957},
}
```