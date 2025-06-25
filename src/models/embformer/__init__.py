# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_embformer import EmbformerConfig
from .modeling_embformer import EmbformerForCausalLM, EmbformerModel

AutoConfig.register(EmbformerConfig.model_type, EmbformerConfig, exist_ok=True)
AutoModel.register(EmbformerConfig, EmbformerModel, exist_ok=True)
AutoModelForCausalLM.register(EmbformerConfig, EmbformerForCausalLM, exist_ok=True)


__all__ = ['EmbformerConfig', 'EmbformerForCausalLM', 'EmbformerModel']
