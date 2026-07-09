# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional secondary 3D vision encoder support (for CT volumes).

Public entry point: `attach_vision_3d_encoder(model, processor, args)`, invoked from
`BaseArguments.get_model_processor` only when `--vision_3d_model` is set.
"""
from .attach import attach_vision_3d_encoder, get_llm_hidden_size, infer_encoder_dim
from .backbone import Vision3DBackbone
from .conv_inflate import conv2d_to_conv3d, ensure_conv3d_patch_embed, find_patch_embed_conv
from .loader import extract_vision_tower, load_vision_3d_source_model

__all__ = [
    'attach_vision_3d_encoder',
    'Vision3DBackbone',
    'ensure_conv3d_patch_embed',
    'conv2d_to_conv3d',
    'find_patch_embed_conv',
    'extract_vision_tower',
    'load_vision_3d_source_model',
    'get_llm_hidden_size',
    'infer_encoder_dim',
]
