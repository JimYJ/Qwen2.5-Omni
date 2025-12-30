import io
import json
import sys
import time
import base64
import importlib.util
import os
from typing import List, Optional, Literal, Union, Any, Dict

import torch
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# 设置环境变量以避免并行相关的问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable parallelism to avoid parallel style configuration issues
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU only
os.environ["WORLD_SIZE"] = "1"  # Single process
os.environ["RANK"] = "0"  # Master process

from awq.models.base import BaseAWQForCausalLM
from transformers import Qwen2_5OmniProcessor

from qwen_omni_utils import process_mm_info
from modeling_qwen2_5_omni_low_VRAM_mode import (
    Qwen2_5OmniDecoderLayer,
    Qwen2_5OmniForConditionalGeneration,
)


def replace_transformers_module():
    """
    使用低显存版本的 Qwen2.5-Omni 实现替换 transformers 内置实现。
    """
    original_mod_name = "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"
    new_mod_path = "modeling_qwen2_5_omni_low_VRAM_mode.py"

    if original_mod_name in sys.modules:
        del sys.modules[original_mod_name]

    spec = importlib.util.spec_from_file_location(original_mod_name, new_mod_path)
    new_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(new_mod)

    sys.modules[original_mod_name] = new_mod


replace_transformers_module()


class Qwen2_5_OmniAWQForConditionalGeneration(BaseAWQForCausalLM):
    layer_type = "Qwen2_5OmniDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["visual"]

    @staticmethod
    def get_model_layers(model: "Qwen2_5OmniForConditionalGeneration"):
        return model.thinker.model.layers

    @staticmethod
    def get_act_for_scaling(module: "Qwen2_5OmniForConditionalGeneration"):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: "Qwen2_5OmniForConditionalGeneration", device: str):
        model.thinker.model.embed_tokens = model.thinker.model.embed_tokens.to(device)
        model.thinker.visual = model.thinker.visual.to(device)
        model.thinker.audio_tower = model.thinker.audio_tower.to(device)

        model.thinker.visual.rotary_pos_emb = model.thinker.visual.rotary_pos_emb.to(
            device
        )
        model.thinker.model.rotary_emb = model.thinker.model.rotary_emb.to(device)

        for layer in model.thinker.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(
        module: "Qwen2_5OmniDecoderLayer", input_feat, module_kwargs
    ):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers


device = "cuda"
model_path = "/data/qwen2.5-omni-7b-awq"


# 初始化量化模型与处理器（全局单例，避免重复加载）
# Load model similar to the working demo, without device_map initially
model = Qwen2_5_OmniAWQForConditionalGeneration.from_quantized(
    model_path,
    model_type="qwen2_5_omni",
    torch_dtype=torch.float16,  # Use float16 like the demo
    # attn_implementation="flash_attention_2",
)

spk_path = model_path + "/spk_dict.pt"
model.model.load_speakers(spk_path)

# Manually place components on devices like the demo
model.model.thinker.model.embed_tokens = model.model.thinker.model.embed_tokens.to(device)
model.model.thinker.visual = model.model.thinker.visual.to(device)
model.model.thinker.audio_tower = model.model.thinker.audio_tower.to(device)
model.model.thinker.visual.rotary_pos_emb = model.model.thinker.visual.rotary_pos_emb.to(device)
model.model.thinker.model.rotary_emb = model.model.thinker.model.rotary_emb.to(device)

for layer in model.model.thinker.model.layers:
    layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

processor = Qwen2_5OmniProcessor.from_pretrained(model_path, local_files_only=True)


# -------- OpenAI 兼容的数据模型 --------


class ChatCompletionMessageContentItem(BaseModel):
    type: Literal["text", "image", "audio", "video"]
    text: Optional[str] = None
    image: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    # 兼容两种形式：老版 string，和新版多模态 list
    content: Union[str, List[ChatCompletionMessageContentItem]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1


def _convert_openai_messages_to_internal(
    messages: List[ChatCompletionMessage],
) -> List[Dict[str, Any]]:
    """
    将 OpenAI ChatCompletion 的 messages 转换为 Qwen Omni 内部使用的 messages 格式：
    [
        {"role": "system", "content": [{"type": "text", "text": "..."}]},
        {"role": "user", "content": [{"type": "video", "video": "xxx.mp4"}]},
        ...
    ]
    """
    converted: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m.content, str):
            converted.append(
                {
                    "role": m.role,
                    "content": [{"type": "text", "text": m.content}],
                }
            )
        else:
            items = []
            for item in m.content:
                if item.type == "text" and item.text is not None:
                    items.append({"type": "text", "text": item.text})
                elif item.type == "image" and item.image is not None:
                    items.append({"type": "image", "image": item.image})
                elif item.type == "audio" and item.audio is not None:
                    items.append({"type": "audio", "audio": item.audio})
                elif item.type == "video" and item.video is not None:
                    items.append({"type": "video", "video": item.video})
            converted.append({"role": m.role, "content": items})
    return converted


def _run_omni_generate(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    调用 Qwen2.5-Omni AWQ 模型生成文本 + 语音。
    返回：
    {
        "text": str,
        "audio_wav_bytes": bytes,
    }
    """
    # 按 low_VRAM_demo_awq 的方式构造输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(device)

    output = model.generate(
        **inputs,
        use_audio_in_video=True,
        return_audio=True,
    )

    text_ids = output[0]
    audio_tensor = output[2]

    decoded = processor.batch_decode(
        text_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response_text = decoded[0]

    # 将音频 tensor 转为 wav bytes
    audio_np = audio_tensor.reshape(-1).detach().cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, samplerate=24000, format="WAV")
    buf.seek(0)
    audio_bytes = buf.read()

    return {"text": response_text, "audio_wav_bytes": audio_bytes}


app = FastAPI(title="Qwen2.5-Omni AWQ OpenAI-Compatible API")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI 兼容的 ChatCompletion API：
    - 支持 messages（文本/多模态）
    - 返回文本 + 音频（wav，以 base64 编码）
    - 当 stream=true 时，使用 SSE 文本流返回（OpenAI 风格的 data: ...\n\n）
    """
    internal_messages = _convert_openai_messages_to_internal(req.messages)

    if not req.stream:
        result = _run_omni_generate(internal_messages)

        audio_b64 = base64.b64encode(result["audio_wav_bytes"]).decode("utf-8")
        now = int(time.time())
        completion = {
            "id": f"chatcmpl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": result["text"]},
                            {
                                "type": "audio",
                                "audio": {
                                    "format": "wav",
                                    "b64": audio_b64,
                                },
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        return JSONResponse(completion)

    # 流式响应：SSE
    async def event_stream():
        now = int(time.time())
        completion_id = f"chatcmpl-{now}"
        result = _run_omni_generate(internal_messages)
        audio_b64 = base64.b64encode(result["audio_wav_bytes"]).decode("utf-8")

        # 文本块
        text_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": result["text"]},
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(text_chunk, ensure_ascii=False)}\n\n"

        # 音频块
        audio_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "audio",
                                "audio": {
                                    "format": "wav",
                                    "b64": audio_b64,
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(audio_chunk, ensure_ascii=False)}\n\n"

        # 结束标记
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


