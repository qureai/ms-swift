# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import inspect
import os
import sglang as sgl
import torch
from copy import deepcopy
from PIL import Image
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from transformers import GenerationConfig
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from swift.metrics import Metric
from swift.model import get_processor
from swift.template import Template
from swift.utils import get_logger, safe_snapshot_download
from .infer_engine import InferEngine
from .protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                       ChatCompletionStreamResponse, ChatMessage, DeltaMessage, EmbeddingResponse,
                       EmbeddingResponseData, InferRequest, RequestConfig, random_uuid)
from .utils import InferStreamer

logger = get_logger()


class SglangEngine(InferEngine):

    def __init__(
        self,
        model_id_or_path: str,
        *,
        template: Optional[Template] = None,
        torch_dtype: Optional[torch.dtype] = None,
        model_type: Optional[str] = None,
        template_type: Optional[str] = None,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        # engine kwargs
        tp_size: int = 1,
        pp_size: int = 1,
        dp_size: int = 1,
        ep_size: int = 1,
        enable_ep_moe: bool = False,
        mem_fraction_static: Optional[float] = None,
        context_length: Optional[int] = None,
        disable_cuda_graph: bool = False,
        quantization: Optional[str] = None,
        task_type: Optional[str] = None,
        kv_cache_dtype: str = 'auto',
        enable_dp_attention: bool = False,
        disable_custom_all_reduce: bool = True,
        speculative_algorithm: Optional[str] = None,
        speculative_num_steps: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
        log_level='error',
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_id_or_path = model_id_or_path
        self.torch_dtype = torch_dtype
        self.model_type = model_type
        self.use_hf = use_hf
        self.hub_token = hub_token
        self.revision = revision
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.ep_size = ep_size
        self.enable_ep_moe = enable_ep_moe
        self.mem_fraction_static = mem_fraction_static
        self.context_length = context_length
        self.disable_cuda_graph = disable_cuda_graph
        self.quantization = quantization
        self.task_type = task_type
        self.kv_cache_dtype = kv_cache_dtype
        self.enable_dp_attention = enable_dp_attention
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.speculative_algorithm = speculative_algorithm
        self.speculative_num_steps = speculative_num_steps
        self.speculative_eagle_topk = speculative_eagle_topk
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.log_level = log_level
        if template is None:
            processor = self._get_processor()
            template = self._get_template(processor, template_type=template_type)
        else:
            safe_snapshot_download(
                model_id_or_path,
                revision=revision,
                download_model=True,
                use_hf=use_hf,
                ignore_patterns=getattr(template.model_meta, 'ignore_patterns', None),
                hub_token=hub_token)
        super().__init__(template)
        self._prepare_server_args(engine_kwargs)
        self.engine = sgl.Engine(server_args=self.server_args)
        self._load_generation_config()
        if speculative_num_draft_tokens is not None:
            self.max_tokens_offset = -speculative_num_draft_tokens

    def _get_processor(self):
        return get_processor(
            model_id_or_path=self.model_id_or_path,
            torch_dtype=self.torch_dtype,
            download_model=True,
            model_type=self.model_type,
            use_hf=self.use_hf,
            hub_token=self.hub_token,
            revision=self.revision,
            task_type=self.task_type)

    def _prepare_server_args(self, engine_kwargs):
        if engine_kwargs is None:
            engine_kwargs = {}
        if self.context_length is not None:
            self.max_model_len = self.context_length
            logger.info(f'Setting max_model_len: {self.context_length}')
        if self.max_model_len is not None:
            self.max_model_len -= 1
        parameters = inspect.signature(ServerArgs).parameters
        if 'pp_size' in parameters:
            engine_kwargs['pp_size'] = self.pp_size
        if 'enable_ep_moe' in parameters:
            engine_kwargs['enable_ep_moe'] = self.enable_ep_moe
        # Env-gated passthrough for ServerArgs fields swift does not expose (e.g.
        # mamba_scheduler_strategy, required for speculative decoding on hybrid models).
        # SGLANG_EXTRA_SERVER_ARGS='{"mamba_scheduler_strategy": "extra_buffer"}'
        extra = os.environ.get('SGLANG_EXTRA_SERVER_ARGS')
        if extra:
            import json as _json
            for k, v in _json.loads(extra).items():
                if k in parameters:
                    engine_kwargs[k] = v
                    logger.info(f'[sglang] extra server arg from env: {k}={v}')
                else:
                    logger.warning(f'[sglang] SGLANG_EXTRA_SERVER_ARGS key ignored '
                                   f'(not a ServerArgs field): {k}')
        self.server_args = ServerArgs(
            model_path=self.model_dir,
            dtype=self.model_info.torch_dtype,
            tp_size=self.tp_size,
            dp_size=self.dp_size,
            ep_size=self.ep_size,
            mem_fraction_static=self.mem_fraction_static,
            context_length=self.context_length,
            disable_cuda_graph=self.disable_cuda_graph,
            quantization=self.quantization,
            kv_cache_dtype=self.kv_cache_dtype,
            enable_dp_attention=self.enable_dp_attention,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            speculative_algorithm=self.speculative_algorithm,
            speculative_num_steps=self.speculative_num_steps,
            speculative_eagle_topk=self.speculative_eagle_topk,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            log_level=self.log_level,
            skip_tokenizer_init=engine_kwargs.pop('skip_tokenizer_init', True),   # grammar backends need the tokenizer -> allow override via SGLANG_EXTRA_SERVER_ARGS
            trust_remote_code=True,
            **engine_kwargs,
        )
        if self.task_type == 'embedding':
            self.server_args.is_embedding = True

    def _load_generation_config(self) -> None:
        generation_config_path = os.path.join(self.model_dir, 'generation_config.json')
        if os.path.isfile(generation_config_path):
            generation_config = GenerationConfig.from_pretrained(self.model_dir)
        else:
            generation_config = GenerationConfig()
        kwargs = generation_config.to_dict()
        top_k = kwargs.get('top_k')
        if top_k == 0:
            kwargs['top_k'] = -1

        parameters = inspect.signature(SamplingParams).parameters
        self.generation_config = {k: v for k, v in kwargs.items() if k in parameters and v is not None}

    def _prepare_generation_config(self, request_config: RequestConfig) -> Dict[str, Any]:
        kwargs = {'max_new_tokens': request_config.max_tokens}
        for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:
            new_value = getattr(request_config, key)
            if new_value is None:
                kwargs[key] = self.generation_config.get(key)
            else:
                kwargs[key] = new_value
        for key in ['n', 'frequency_penalty', 'presence_penalty']:
            kwargs[key] = getattr(request_config, key)

        return kwargs

    def _add_stop_words(self, generation_config: Dict[str, Any], request_config: RequestConfig) -> None:
        template_meta = self.template.template_meta
        stop_words = (request_config.stop or []) + (self.generation_config.get('stop') or []) + template_meta.stop_words
        generation_config['stop_token_ids'] = self._get_stop_token_ids(stop_words)
        # structured outputs: pass sglang grammar constraints through when set on the request
        # (json_schema / regex / ebnf as extra attributes, or the vLLM-style structured_outputs_regex field)
        for k in ('json_schema', 'regex', 'ebnf'):
            v = getattr(request_config, k, None)
            if v: generation_config[k] = v
        if getattr(request_config, 'structured_outputs_regex', None) and 'regex' not in generation_config:
            generation_config['regex'] = request_config.structured_outputs_regex

    @staticmethod
    def _get_logprob_kwargs(request_config: RequestConfig) -> Dict[str, Any]:
        """sglang takes the logprob switches as `async_generate` kwargs, not as sampling_params."""
        if not request_config.logprobs and not request_config.token_ids_logprob:
            return {}
        kwargs = {'return_logprob': True}
        if request_config.top_logprobs is not None:
            kwargs['top_logprobs_num'] = request_config.top_logprobs
        if request_config.token_ids_logprob:
            kwargs['token_ids_logprob'] = list(request_config.token_ids_logprob)
        return kwargs

    def _get_sglang_logprobs(self, meta_info, token_ids: List[int],
                            request_config: RequestConfig) -> Optional[Dict[str, Any]]:
        """Convert sglang's meta_info logprobs into swift's OpenAI-style logprobs dict.

        sglang returns `output_token_logprobs` as [(logprob, token_id, text), ...] and, when
        `top_logprobs_num > 0`, `output_top_logprobs` as one such list per position. Swift's
        `_get_logprobs` expects one {token_id: logprob} mapping per position instead.
        """
        token_logprobs = meta_info.get('output_token_logprobs')
        if not token_logprobs:
            return None
        top_logprobs = meta_info.get('output_top_logprobs') or []
        logprobs_list = []
        for i, item in enumerate(token_logprobs):
            logprob, token_id = item[0], item[1]
            mapping = {top[1]: top[0] for top in top_logprobs[i]} if i < len(top_logprobs) else {}
            # the sampled token is not guaranteed to be inside top-k; it always wins.
            mapping[token_id] = logprob
            logprobs_list.append(mapping)
        res = self._get_logprobs(logprobs_list, token_ids, request_config.top_logprobs)
        # `token_ids_logprob` entries are usually far outside top-k, so they get their own
        # key rather than being merged into (and then truncated out of) `top_logprobs`.
        requested = meta_info.get('output_token_ids_logprobs')
        if res is not None and requested:
            requested = requested[-len(res['content']):]
            for pos, entry in zip(res['content'], requested):
                pos['token_ids_logprobs'] = [{
                    'token': self.tokenizer.decode(item[1]),
                    'token_id': item[1],
                    'logprob': item[0],
                } for item in entry]
        return res

    def _create_chat_completion_response(self, output, inputs, request_config: RequestConfig):
        assert output is not None
        return_details = request_config.return_details
        meta_info = output['meta_info']
        usage_info = self._get_usage_info(meta_info['prompt_tokens'], meta_info['completion_tokens'])
        response = self.template.decode(output['output_ids'], template_inputs=inputs['template_inputs'])
        toolcall = self._get_toolcall(response)
        token_ids = output['output_ids'] if return_details else None
        logprobs = self._get_sglang_logprobs(meta_info, output['output_ids'], request_config)
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
            finish_reason=meta_info['finish_reason']['type'],
            logprobs=logprobs,
            token_ids=token_ids)
        prompt_token_ids = None
        images_size = None
        if return_details:
            prompt_token_ids = output.get('prompt_token_ids')
            images = inputs['template_inputs'].images
            if all(isinstance(image, Image.Image) for image in images):
                images_size = [image.size for image in images]
        return ChatCompletionResponse(
            model=self.model_name,
            choices=[choice],
            usage=usage_info,
            id=random_uuid(),
            prompt_token_ids=prompt_token_ids,
            images_size=images_size)

    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        use_tqdm: Optional[bool] = None,
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        return super().infer(infer_requests, request_config, metrics, use_tqdm=use_tqdm)

    async def infer_async(self,
                          infer_request: InferRequest,
                          request_config: Optional[RequestConfig] = None,
                          *,
                          pre_infer_hook=None,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        request_config = deepcopy(request_config or RequestConfig())
        self.template.set_mode('sglang')
        loop = asyncio.get_running_loop()
        with torch.inference_mode():
            inputs = await loop.run_in_executor(None, self.template.encode, infer_request, True)
        if self.task_type == 'embedding':
            inputs.pop('length', None)
        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config)
        kwargs.update({'inputs': inputs, 'generation_config': generation_config, 'request_config': request_config})
        if pre_infer_hook:
            kwargs = pre_infer_hook(kwargs)
        if request_config.stream:
            return self._infer_stream_async(**kwargs)
        elif self.task_type == 'embedding':
            kwargs.pop('generation_config', None)
            return await self._infer_embedding_async(**kwargs)
        else:
            return await self._infer_full_async(**kwargs)

    async def _infer_embedding_async(self, inputs: Dict[str, Any], **kwargs) -> EmbeddingResponse:
        from sglang.srt.managers.io_struct import EmbeddingReqInput
        obj = EmbeddingReqInput(
            input_ids=inputs['input_ids'], image_data=inputs.get('images'), audio_data=inputs.get('audios'))
        generator = self.engine.tokenizer_manager.generate_request(obj, None)
        output = await generator.__anext__()
        usage_info = self._get_usage_info(output['meta_info']['prompt_tokens'], 0)
        return EmbeddingResponse(
            model=self.model_name,
            data=[EmbeddingResponseData(embedding=output['embedding'])],
            usage=usage_info,
            id=random_uuid())

    async def _infer_full_async(self, inputs: Dict[str, Any], generation_config: Dict[str, Any],
                                request_config: RequestConfig) -> ChatCompletionResponse:
        # sglang's async_generate expects `image_data`/`audio_data`/`video_data`, but swift's
        # `inputs` carries `images`/`audios`/`videos` (as the embedding path above already remaps).
        _media_key_map = {'images': 'image_data', 'audios': 'audio_data', 'videos': 'video_data'}
        engine_inputs = {}
        for k, v in inputs.items():
            if k == 'template_inputs':
                continue
            engine_inputs[_media_key_map.get(k, k)] = v
        engine_inputs.update(self._get_logprob_kwargs(request_config))
        output = await self.engine.async_generate(**engine_inputs, sampling_params=generation_config)
        output['prompt_token_ids'] = inputs['input_ids']
        return self._create_chat_completion_response(output, inputs, request_config)

    async def _infer_stream_async(self, inputs: Dict[str, Any], generation_config: Dict[str, Any],
                                  request_config: RequestConfig, **kwargs) -> AsyncIterator[
                                      ChatCompletionStreamResponse]:
        engine_inputs = {k: v for k, v in inputs.items() if k != 'template_inputs'}
        engine_inputs.update(self._get_logprob_kwargs(request_config))
        result_generator = await self.engine.async_generate(
            **engine_inputs, sampling_params=generation_config, stream=True)
        infer_streamer = InferStreamer(self.template, template_inputs=inputs['template_inputs'])
        # sglang streams cumulative output_ids/logprobs; only the new tokens belong in each chunk.
        num_emitted = 0
        async for output in result_generator:
            res = self._create_chat_completion_stream_response(output, infer_streamer, request_config, num_emitted)
            num_emitted = len(output['output_ids'])
            if res is None:
                continue
            yield res

    def _create_chat_completion_stream_response(self, output, infer_streamer, request_config: RequestConfig,
                                                num_emitted: int = 0) -> Optional[ChatCompletionStreamResponse]:
        assert output is not None
        meta_info = output['meta_info']
        finish_reason = meta_info['finish_reason']
        is_finished = finish_reason is not None
        delta_text = infer_streamer.get_printable_text(output['output_ids'], is_finished)
        if not delta_text and not is_finished:
            return
        toolcall = None
        if is_finished:
            finish_reason = finish_reason['type']
            toolcall = self._get_toolcall(self.template.decode(output['output_ids'], **infer_streamer.decode_kwargs))
        meta_info = output['meta_info']
        usage_info = self._get_usage_info(meta_info['prompt_tokens'], meta_info['completion_tokens'])
        logprobs = self._get_sglang_logprobs(meta_info, output['output_ids'][num_emitted:], request_config)
        choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),
            finish_reason=finish_reason,
            logprobs=logprobs)
        return ChatCompletionStreamResponse(model=self.model_name, choices=[choice], usage=usage_info)
