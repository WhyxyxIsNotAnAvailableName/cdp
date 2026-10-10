# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

from typing import List, Optional
import torch
from cdp.models.utils import has_flash_attn
from cdp.training.logging import initialize_overwatch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
logger = initialize_overwatch(__name__)
import torch.nn as nn

class _QWen_VL_Interface(nn.Module):
    def __init__(self, config: Optional[dict]=None, **kwargs):
        super().__init__()
        qwenvl_config = config.framework.get('qwenvl', {})
        model_id = qwenvl_config['base_vlm']
        attn_implementation = qwenvl_config.get('attn_implementation', 'sdpa')
        if attn_implementation == 'flash_attention_2':
            if not has_flash_attn():
                print('[WARNING] flash_attn not installed, falling back to sdpa')
                attn_implementation = 'sdpa'
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, attn_implementation=attn_implementation, torch_dtype='auto')
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = 'left'
        self.model = model
        self.processor = processor
        self.config = config
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, image_grid_thw: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=True, return_dict: Optional[bool]=True, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        return outputs

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        messages = []
        assert len(images) == len(instructions), 'Images and instructions must have the same length'
        for (imgs, instruction) in zip(images, instructions):
            content = [{'type': 'image', 'image': img} for img in imgs]
            prompt = instruction
            content.append({'type': 'text', 'text': prompt})
            msg = [{'role': 'user', 'content': content}]
            messages.append(msg)
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        (image_inputs, video_inputs) = process_vision_info(messages)
        batch_input = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
        return batch_input.to(self.model.device)
