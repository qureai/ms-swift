# Copyright (c) ModelScope Contributors. All rights reserved.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
import torch
import torch.distributed as dist
from contextlib import contextmanager, nullcontext
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import TrainerCallback
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
from swift.sequence_parallel import sequence_parallel
from swift.utils import HfConfigFactory, JsonlWriter, Serializer, gc_collect, get_logger, unwrap_model_for_generation
from .arguments import Seq2SeqTrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin
from .utils import per_token_loss_func, per_token_loss_func_sp

logger = get_logger()


class _ViTWarmupFreezeCallback(TrainerCallback):
    """Freeze the pretrained 3D-encoder BLOCKS for the first `freeze_steps` steps, then unfreeze. Freezing at
    train-begin (AFTER the optimizer is built, so the params stay in the optimizer group) makes grads None
    while frozen -> no update; unfreezing restores training. Adapter/patch-embed/projector stay trainable."""

    def __init__(self, model, freeze_steps: int):
        self.freeze_steps = int(freeze_steps)
        self._frozen = []
        self._done = False
        self._model = model

    def _blocks(self):
        for n, p in self._model.named_parameters():
            if 'visual_3d' in n and 'encoder.blocks' in n:
                yield n, p

    def on_train_begin(self, args, state, control, **kwargs):
        if self._done or self.freeze_steps <= 0:
            return
        for _n, p in self._blocks():
            if p.requires_grad:
                p.requires_grad_(False)
                self._frozen.append(p)
        get_logger().info(f'vision_3d: ViT warmup-freeze ON -> froze {len(self._frozen)} encoder-block tensors '
                          f'for the first {self.freeze_steps} steps.')

    def on_step_end(self, args, state, control, **kwargs):
        if not self._done and state.global_step >= self.freeze_steps and self._frozen:
            for p in self._frozen:
                p.requires_grad_(True)
            get_logger().info(f'vision_3d: ViT warmup-freeze released at step {state.global_step} '
                              f'({len(self._frozen)} tensors unfrozen).')
            self._frozen = []
            self._done = True


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            self.infer_engine = TransformersEngine(
                self.model, template=self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))
        # Presence-boolean weighting: known from args at init so we can request output_hidden_states before
        # the forward. Token ids / lm_head are resolved lazily (need the built model) in _maybe_setup.
        self._presence_enabled = (self.args.presence_ce_w_true != 1.0 or self.args.presence_ce_w_false != 1.0)
        self._presence_setup_done = False
        # Per-dataset (channel) loss logging (monitoring only). Enabled by the same env that tags datasets.
        import os as _os
        self._ch_enabled = bool(_os.environ.get('CT_DATASET_CHANNELS'))
        self._ch_setup_done = False
        # VICReg anti-collapse rank-reg (reads backbone._rankreg_feats stashed during the 3D forward).
        self._rankreg_lambda = float(getattr(self.args, 'vision_3d_rankreg_lambda', 0.0) or 0.0)
        self._collapse_enabled = bool(getattr(self.args, 'log_collapse_metrics', False))
        # Fast-channel (post-proj <video> token) anti-collapse: VICReg on the fast tokens + orthogonality /
        # weight-decay on proj_fast. Fixes the observed fast-token rank-~1 collapse (pre-proj stays high-rank).
        self._rankreg_fast_lambda = float(getattr(self.args, 'vision_3d_rankreg_fast_lambda', 0.0) or 0.0)
        self._proj_ortho_lambda = float(getattr(self.args, 'vision_3d_proj_ortho_lambda', 0.0) or 0.0)
        self._proj_fast_wd = float(getattr(self.args, 'vision_3d_proj_fast_wd', 0.0) or 0.0)
        # Per-tag AUROC on cls_json val (teacher-forced P(true)); reuses the presence true/false ids + lm_head.
        self._train_acc_enabled = bool(getattr(self.args, 'log_train_acc', False))
        self._cls_auc_enabled = bool(getattr(self.args, 'eval_cls_auc', False))
        # Final hidden state is captured via a hook on the LAST decoder layer (fires under Liger) instead of
        # output_hidden_states=True -- the latter makes Liger fall back to materialising FULL-vocab logits
        # (huge memory / OOM on long sequences). We apply the final norm ourselves before the lm_head.
        self._hidden_cap_ready = False
        self._captured_hidden = None
        self._final_norm = None
        self._capture_on = False
        self._cls_auc_buf = {}          # slot -> ([scores], [labels]), accumulated across an eval pass
        self._cls_auc_tag_names = None
        _tags = getattr(self.args, 'eval_cls_auc_tags', None)
        if self._cls_auc_enabled and _tags and os.path.isfile(_tags):
            with open(_tags) as _f:
                self._cls_auc_tag_names = [ln.strip() for ln in _f if ln.strip()]
        # ViT warmup-freeze recipe: freeze pretrained encoder blocks for the first N steps.
        _fz = int(getattr(self.args, 'vision_3d_vit_freeze_steps', 0) or 0)
        if _fz > 0:
            self.add_callback(_ViTWarmupFreezeCallback(self.model, _fz))

    def _maybe_setup_presence_weighting(self):
        """Lazy, once: resolve the `true`/`false` value-token ids (from the LIVE tokenizer, no hardcoding),
        the lm_head, and the text hidden size. The final hidden state is read from outputs.hidden_states[-1]
        (requested via output_hidden_states=True on the forward) rather than a module hook, because Liger /
        torch.compile bypass the norm module's Python forward so a hook never fires."""
        if self._presence_setup_done:
            return
        self._presence_setup_done = True
        if not (self._presence_enabled or self._cls_auc_enabled):   # cls_auc reuses the same true/false ids
            return
        if self.template.padding_free or self.template.sequence_parallel_size > 1:
            logger.warning('presence_ce / eval_cls_auc need a padded, non-SP setup; disabling both.')
            self._presence_enabled = False
            self._cls_auc_enabled = False
            return
        tok = getattr(self.template, 'tokenizer', None) or getattr(self, 'processing_class', None)
        # Derive the `true`/`false` value-token ids from THIS tokenizer by diffing two JSON snippets that
        # differ ONLY in the boolean (identical context both sides -> the first divergent token IS the boolean).
        # Robust to BPE prefix re-tokenisation and to any base model, so nothing is hardcoded.
        t_ids = tok.encode('{"presence": true, "size"', add_special_tokens=False)
        f_ids = tok.encode('{"presence": false, "size"', add_special_tokens=False)
        i = 0
        while i < len(t_ids) and i < len(f_ids) and t_ids[i] == f_ids[i]:
            i += 1
        if i >= len(t_ids) or i >= len(f_ids):
            raise ValueError('presence_ce: could not derive true/false token ids from the tokenizer.')
        self._presence_true_id = t_ids[i]
        self._presence_false_id = f_ids[i]
        base_model = self.template.get_base_model(self.model)
        self._presence_lm_head = base_model.get_output_embeddings()
        cfg = base_model.config
        text_cfg = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
        self._presence_hidden_size = int(getattr(text_cfg, 'hidden_size'))
        logger.info(f'presence_ce weighting ON: channel={self.args.presence_ce_channel!r} '
                    f'w_true={self.args.presence_ce_w_true} w_false={self.args.presence_ce_w_false} '
                    f'true_id={self._presence_true_id} false_id={self._presence_false_id} '
                    f'hidden_size={self._presence_hidden_size} lm_head={type(self._presence_lm_head).__name__} '
                    f'(source=outputs.hidden_states[-1])')

    def _presence_weight_extra_loss(self, outputs, labels, channels, num_items_in_batch):
        """Extra loss = sum over presence value-token positions of (w-1)*CE, / num_items_in_batch, so those
        tokens count w-times in the LM loss: w_true for a `true` GT token, w_false for `false`. With
        w_true > w_false, predicting `false` for a GT-`true` finding is the hardest loss. Scoped to samples
        whose channel == presence_ce_channel. Hidden = outputs.hidden_states[-1] sliced to the logits_to_keep
        window so it aligns token-for-token with labels; scored only at the boolean positions (Liger stays on,
        no full-vocab logits). No grad path is disabled -- this term backprops into the model.
        Disabled (returns None) when weights are 1.0/1.0, or there is no channel info to scope by."""
        if channels is None or len(channels) != labels.shape[0]:
            return None
        h = self._final_hidden(labels)
        if h is None:
            return None
        rolled = torch.roll(labels, shifts=-1, dims=-1)   # rolled[:,j] == labels[:,j+1]; h[j] predicts it
        is_bool = (rolled == self._presence_true_id) | (rolled == self._presence_false_id)
        gate = torch.tensor([c == self.args.presence_ce_channel for c in channels],
                            device=rolled.device).view(rolled.shape[0], 1)
        is_bool = is_bool & gate
        if not is_bool.any():
            return None
        mask_flat = is_bool.reshape(-1)
        tgt = rolled.reshape(-1)[mask_flat]
        hb = h.reshape(-1, h.shape[-1])[mask_flat]
        logits_b = self._presence_lm_head(hb).float()
        ce = nn.functional.cross_entropy(logits_b, tgt, reduction='none')
        w = torch.where(tgt == self._presence_true_id,
                        torch.as_tensor(self.args.presence_ce_w_true, device=ce.device, dtype=ce.dtype),
                        torch.as_tensor(self.args.presence_ce_w_false, device=ce.device, dtype=ce.dtype))
        denom = num_items_in_batch
        if denom is None:
            denom = (labels != -100).sum()
        return ((w - 1.0) * ce).sum() / denom

    def _setup_hidden_capture(self, model):
        """Register (once) a forward hook on the LAST decoder layer to capture its output, and resolve the
        final norm. Replaces output_hidden_states=True, which forces Liger to materialise full-vocab logits."""
        if self._hidden_cap_ready:
            return
        self._hidden_cap_ready = True
        tm = None
        for m in model.modules():
            if isinstance(getattr(m, 'layers', None), nn.ModuleList) and hasattr(m, 'norm') and len(m.layers) > 0:
                tm = m
                break
        if tm is None:
            logger.warning('hidden-capture: could not locate text model (.layers + .norm); '
                           'presence/channel/AUC/train-acc monitoring disabled.')
            self._presence_enabled = self._cls_auc_enabled = self._train_acc_enabled = self._ch_enabled = False
            return
        self._final_norm = tm.norm

        def _hook(mod, inp, out):
            self._captured_hidden = (out[0] if isinstance(out, (tuple, list)) else out) if self._capture_on else None

        tm.layers[-1].register_forward_hook(_hook)

    def _final_hidden(self, labels):
        """Post-final-norm last hidden state aligned to `labels` (last-N slice), from the captured decoder
        output. None if unavailable / misaligned."""
        h = self._captured_hidden
        if h is None or self._final_norm is None:
            return None
        h = self._final_norm(h)
        if h.shape[1] != labels.shape[1]:
            h = h[:, -labels.shape[1]:, :]
        if h.dim() != 3 or h.shape[0] != labels.shape[0] or h.shape[1] != labels.shape[1]:
            return None
        return h

    def _log_train_token_acc(self, outputs, labels, mode):
        """TRAIN token accuracy from hidden_states[-1] + lm_head (chunked) when Liger left no logits. Called
        on logging steps only. h[j] predicts labels[j+1]; scored on response tokens (labels != -100)."""
        self._ch_setup()                                       # resolves self._ch_lm_head
        h = self._final_hidden(labels)
        if h is None:
            return
        shift_h = h[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        if not mask.any():
            return
        hb = shift_h[mask]
        tgt = shift_labels[mask]
        correct = total = 0
        with torch.no_grad():
            for i in range(0, hb.shape[0], 4096):              # chunk the full-vocab argmax to bound memory
                preds = self._ch_lm_head(hb[i:i + 4096]).argmax(dim=-1)
                correct += (preds == tgt[i:i + 4096]).sum().item()
                total += preds.shape[0]
        if total:
            self.custom_metrics[mode]['acc'].update(torch.tensor(correct / total))

    def _cls_auc_accumulate(self, outputs, labels, channels):
        """Accumulate teacher-forced P(true) + GT per tag-SLOT (order of appearance in the fixed roll-call)
        for cls_json rows, from the presence boolean positions. Reuses the presence true/false ids + lm_head."""
        if channels is None or len(channels) != labels.shape[0] or not hasattr(self, '_presence_true_id'):
            return
        h = self._final_hidden(labels)
        if h is None:
            return
        rolled = torch.roll(labels, shifts=-1, dims=-1)               # h[j] predicts rolled[:,j]
        is_bool = (rolled == self._presence_true_id) | (rolled == self._presence_false_id)
        gate = torch.tensor([c == self.args.presence_ce_channel for c in channels],
                            device=rolled.device).view(rolled.shape[0], 1)
        is_bool = is_bool & gate
        if not is_bool.any():
            return
        slot = torch.cumsum(is_bool.long(), dim=1) - 1               # per-row running index of bool positions
        mask_flat = is_bool.reshape(-1)
        tgt = rolled.reshape(-1)[mask_flat]
        slot_flat = slot.reshape(-1)[mask_flat]
        hb = h.reshape(-1, h.shape[-1])[mask_flat]
        with torch.no_grad():
            logits_b = self._presence_lm_head(hb).float()
            ptrue = torch.sigmoid(logits_b[:, self._presence_true_id] - logits_b[:, self._presence_false_id])
        gt = (tgt == self._presence_true_id).float()
        for s, p, g in zip(slot_flat.tolist(), ptrue.cpu().tolist(), gt.cpu().tolist()):
            sc, la = self._cls_auc_buf.setdefault(int(s), ([], []))
            sc.append(p)
            la.append(g)

    def _cls_auc_compute_and_log(self, mode='eval'):
        """At eval end: per-slot AUROC (skipping slots without both classes) -> macro + wandb `cls_macro_auc`."""
        if not self._cls_auc_buf:
            return
        try:
            from sklearn.metrics import roc_auc_score
        except Exception:
            self._cls_auc_buf = {}
            return
        aucs, lines = [], []
        for s in sorted(self._cls_auc_buf):
            sc, la = self._cls_auc_buf[s]
            if len(set(la)) < 2:                                     # AUROC needs both classes present
                continue
            try:
                a = float(roc_auc_score(la, sc))
            except Exception:
                continue
            aucs.append(a)
            name = (self._cls_auc_tag_names[s] if self._cls_auc_tag_names and s < len(self._cls_auc_tag_names)
                    else f'slot{s}')
            lines.append(f'{name}={a:.3f}(n={len(la)})')
        if aucs:
            macro = sum(aucs) / len(aucs)
            self.custom_metrics[mode]['cls_macro_auc'].update(torch.tensor(macro))
            logger.info(f'cls_json per-tag AUC (macro={macro:.4f} over {len(aucs)} tags): '
                        + ', '.join(lines[:40]))
        self._cls_auc_buf = {}

    def evaluate(self, *args, **kwargs):
        if self._cls_auc_enabled:
            self._cls_auc_buf = {}
        out = super().evaluate(*args, **kwargs)
        if self._cls_auc_enabled:
            self._cls_auc_compute_and_log('eval')
        return out

    def _ch_setup(self):
        if self._ch_setup_done:
            return
        self._ch_setup_done = True
        base_model = self.template.get_base_model(self.model)
        self._ch_lm_head = base_model.get_output_embeddings()

    def _log_channel_losses(self, outputs, labels, channels, mode):
        """Monitoring only (no grad): mean next-token CE per dataset `channel`, from outputs.hidden_states[-1]
        (sliced to the logits_to_keep window so it aligns with labels) + lm_head, chunked. Updates
        custom_metrics -> wandb `loss/<channel>`. Never affects the training loss or gradients."""
        if channels is None:
            return
        self._ch_setup()
        lm_head = self._ch_lm_head
        if lm_head is None:
            return
        with torch.no_grad():
            h = self._final_hidden(labels)
            if h is None:
                return
            rolled = torch.roll(labels, shifts=-1, dims=-1)   # rolled[:,j]==labels[:,j+1]; h[j] predicts it
            valid = rolled != -100
            for i in range(labels.shape[0]):
                ci = channels[i] if i < len(channels) else None
                if ci is None:
                    continue
                vi = valid[i]
                n = int(vi.sum())
                if n == 0:
                    continue
                hi, ti = h[i][vi], rolled[i][vi]
                ce_sum = hi.new_zeros(())
                for st in range(0, n, 1024):              # chunk to bound the (chunk x vocab) logits
                    lg = lm_head(hi[st:st + 1024]).float()
                    ce_sum = ce_sum + nn.functional.cross_entropy(lg, ti[st:st + 1024], reduction='sum')
                self.custom_metrics[mode][f'loss/{ci}'].update(ce_sum / n)

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        packing = self.template.packing
        padding_free = self.template.padding_free
        self.template.packing = False
        self.template.padding_free = False
        try:
            yield
        finally:
            self.template.packing = packing
            self.template.padding_free = padding_free
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            with self.template.forward_context(self.model, inputs):
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
            )

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def _prepare_inputs(self, inputs):
        args = self.args
        inputs = super()._prepare_inputs(inputs)
        if self.template.sequence_parallel_size > 1:
            sequence_parallel.prepare_inputs(inputs)

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            if args.tuner_backend == 'unsloth' and isinstance(inputs['logits_to_keep'], torch.Tensor):
                inputs['logits_to_keep'] = int(inputs['logits_to_keep'].sum())

        base_model = self.template.get_base_model(self.model)
        forward_params = inspect.signature(base_model.forward).parameters
        if self.model.model_info.is_moe_model and any(key in forward_params
                                                      for key in ['output_router_logits', 'kwargs']):
            HfConfigFactory.set_config_attr(base_model.config, 'router_aux_loss_coef', args.router_aux_loss_coef)
            base_model.router_aux_loss_coef = args.router_aux_loss_coef
            logger.info_once(f'router_aux_loss_coef: {args.router_aux_loss_coef}')
            if args.router_aux_loss_coef > 0:
                inputs['output_router_logits'] = True
        inputs['compute_loss_func'] = self.compute_loss_func
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = None
        compute_loss_func: Callable = inputs.pop('compute_loss_func', None)
        loss_scale = inputs.pop('loss_scale', None)
        text_position_ids = inputs.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = inputs.get('position_ids')
        channels = inputs.pop('channel', None)

        if (self.label_smoother is not None or compute_loss_func is not None or loss_scale is not None
                or self.args.enable_dft_loss or self.args.enable_channel_loss
                or self.template.sequence_parallel_size > 1) and 'labels' in inputs:
            if self.args.use_liger_kernel:
                logger.warning_once('The cross_entropy loss function defined in Liger Kernel will not '
                                    'take effect, potentially leading to increased GPU memory consumption.')
            labels = inputs.pop('labels')
        # Final hidden state is needed for presence weighting and/or per-channel loss logging; request it on
        # the forward (Liger/compile bypass the norm module so a hook can't capture it). Channel logging runs
        # only on logging steps during training (every eval step), so the extra tensor is occasional.
        # Presence weighting & per-channel logging read hidden_states[-1] and rely on it aligning with
        # `labels`. That holds only when labels was left as a contiguous last-N slice by prepare_logits_to_keep
        # (its batch>1 branch, logits_to_keep = int) or not sliced at all (None). Its batch==1 branch (used by
        # eval) COMPACTS labels to scattered response tokens and makes logits_to_keep a MASK TENSOR -> the
        # last-N slice would silently misalign. So require training + a non-tensor logits_to_keep.
        # Alignment holds when labels stayed a contiguous last-N slice (prepare_logits_to_keep batch>1 branch,
        # logits_to_keep=int) or was not sliced (None). The batch==1 branch (eval with eval_bs=1) COMPACTS
        # labels and makes logits_to_keep a MASK TENSOR -> misaligned, so skip it.
        _aligned = (not torch.is_tensor(inputs.get('logits_to_keep'))
                    and not self.template.padding_free and self.template.sequence_parallel_size == 1)
        # per-channel logging (monitoring, no grad): training logging-steps AND every eval batch, when aligned.
        self._ch_log_now = self._ch_enabled and _aligned and (
            not self.model.training or (self.state.global_step + 1) % max(self.args.logging_steps, 1) == 0)
        # presence weighting: TRAINING only (must not alter the eval loss).
        self._presence_now = self._presence_enabled and _aligned and self.model.training
        # per-tag AUC: EVAL only (teacher-forced P(true) at presence positions).
        self._cls_auc_now = self._cls_auc_enabled and _aligned and (not self.model.training)
        # train token acc (Liger has no logits): TRAINING, on logging steps only.
        self._train_acc_now = self._train_acc_enabled and _aligned and self.model.training and (
            (self.state.global_step + 1) % max(self.args.logging_steps, 1) == 0)
        # Capture the final hidden state via a decoder-layer hook (NOT output_hidden_states, which forces
        # Liger to materialise full-vocab logits -> OOM on long sequences). The hook stores it only this step.
        self._capture_on = bool(self._ch_log_now or self._presence_now or self._cls_auc_now or self._train_acc_now)
        if self._capture_on:
            self._setup_hidden_capture(model)
        # EVAL with Liger: Liger only fuses the CE during training (skip_logits defaults to self.training), so
        # eval otherwise materialises FULL-vocab logits over the (long) response -> OOM. Force the fused path;
        # the eval loss still comes from the fused kernel and the per-tag AUC reads the captured hidden state.
        if not self.model.training and getattr(self.args, 'use_liger_kernel', False) \
                and not getattr(self.args, 'predict_with_generate', False):
            inputs['skip_logits'] = True
        outputs = model(**inputs)
        mode = 'train' if self.model.training else 'eval'
        if getattr(outputs, 'aux_loss', None) is not None:
            self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if hasattr(self.args, 'past_index') and self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs.loss = None
            if (self.args.enable_dft_loss or loss_scale is not None or self.args.enable_channel_loss
                    or self.template.sequence_parallel_size > 1):
                if self.template.sequence_parallel_size > 1:
                    outputs.loss = per_token_loss_func_sp(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
                else:
                    outputs.loss = per_token_loss_func(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)

                if loss_scale is not None:
                    loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                    outputs.loss = outputs.loss * loss_scale

                if self.args.enable_channel_loss:
                    metrics = self.custom_metrics[mode]
                    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
                    if self.template.padding_free:
                        cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
                    else:
                        cu_seqlens = torch.arange(0, labels.shape[0] + 1) * labels.shape[1]
                    for i in range(cu_seqlens.shape[0] - 1):
                        channel = None if channels is None else channels[i]
                        slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        metrics[f'loss_{channel}'].update(outputs.loss[slice_][masks[slice_]])

            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(
                    outputs, labels, num_items_in_batch=num_items_in_batch, loss_scale=loss_scale, trainer=self)
            elif self.label_smoother is None:
                # Handle the outputs.loss generated by loss_scale.
                if num_items_in_batch is None:
                    # https://github.com/huggingface/transformers/blob/9dff7ca5c9693f4c02cdd2a9c2abc4772fcea5da/src/transformers/trainer.py#L2137
                    num_items_in_batch = (labels != -100).sum()  # compat SP
                    if self.template.sequence_parallel_size > 1:
                        # labels are sharded by SP; outputs.loss was gathered
                        # to full length via GatherLoss. Reduce the denominator
                        # across the SP group so it matches the gathered loss.
                        dist.all_reduce(num_items_in_batch, op=dist.ReduceOp.SUM)
                loss = outputs.loss.sum() / num_items_in_batch
            else:
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)

            if self.model.model_info.is_moe_model and self.args.router_aux_loss_coef is not None:
                aux_loss = outputs.get('aux_loss')
                if aux_loss is not None:
                    if num_items_in_batch is not None:
                        aux_loss = aux_loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
                    loss = loss + self.args.router_aux_loss_coef * aux_loss.to(loss.device)

        if getattr(self.args, 'average_tokens_across_devices',
                   False) and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes
            if mode == 'eval' and self.template.sequence_parallel_size > 1:
                loss /= self.template.sequence_parallel_size

        if (outputs.logits is not None and labels is not None and self.args.tuner_backend != 'unsloth'):
            cu_seqlens = None
            if self.template.padding_free and self.args.acc_strategy == 'seq':
                cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
            # Liger does not have logits
            # Unsloth has a bug with output logits
            self._compute_acc(outputs, labels, cu_seqlens=cu_seqlens)
        # Auxiliary vision-grounding loss (lambda * ASL from the ML-Decoder head), stashed by the CT
        # template's _post_encode during the forward. Added to the LM loss so it backprops into the 3D
        # encoder. None when the head is off / no labeled volumes in the batch.
        aux_loss = getattr(self.template, '_aux_mldecoder_loss', None)
        if aux_loss is not None:
            loss = loss + aux_loss.to(loss.device)
            self.custom_metrics['train' if self.model.training else 'eval']['aux_mldecoder_loss'].update(aux_loss)
            self.template._aux_mldecoder_loss = None
        # Asymmetric presence-boolean weighting (cls-json channel only). Adds (w-1)*CE at presence value-token
        # positions so those tokens count w-times in the loss; w_true > w_false makes a false-for-true miss the
        # hardest. Scored on the gathered boolean positions only (outputs.hidden_states[-1] + lm_head), so
        # Liger stays on for the bulk and no full-vocab logits are materialised.
        if getattr(self, '_ch_log_now', False) and labels is not None:
            self._log_channel_losses(outputs, labels, channels, mode)
        self._maybe_setup_presence_weighting()
        if getattr(self, '_cls_auc_now', False) and labels is not None:
            self._cls_auc_accumulate(outputs, labels, channels)
        if getattr(self, '_train_acc_now', False) and labels is not None and outputs.logits is None:
            self._log_train_token_acc(outputs, labels, mode)
        if getattr(self, '_presence_now', False) and labels is not None:
            extra = self._presence_weight_extra_loss(outputs, labels, channels, num_items_in_batch)
            if extra is not None:
                loss = loss + extra.to(loss.device)
                self.custom_metrics[mode]['presence_ce_extra'].update(extra)
        # VICReg anti-collapse rank-reg on the 3D encoder's pre-projector tokens (stashed on the backbone
        # during the forward). Off (no compute) unless --vision_3d_rankreg_lambda>0. Targets the correlation
        # collapse the diagnostic found (cov term raises effective rank).
        if (self._rankreg_lambda > 0 or self._collapse_enabled or self._rankreg_fast_lambda > 0
                or self._proj_ortho_lambda > 0 or self._proj_fast_wd > 0) and self.model.training:
            base = model
            for _ in range(4):
                if hasattr(base, 'visual_3d'):
                    break
                base = getattr(base, 'module', base)
            bb = getattr(base, 'visual_3d', None)
            feats = getattr(bb, '_rankreg_feats', None) if bb is not None else None
            if feats is not None:
                mt = self.args.vision_3d_rankreg_max_tokens
                # collapse detector (no extra forward), only on logging steps to keep the SVD rare
                if self._collapse_enabled and self.state.global_step % max(1, self.args.logging_steps) == 0:
                    from swift.model.vision_3d.slowfast import collapse_metrics
                    # PRE-proj: the 3D-encoder (slow) tokens. (Historically mislogged as `postproj_*`; it has
                    # always measured the PRE-projector encoder rank -- renamed to be accurate.)
                    cm = collapse_metrics(feats, max_tokens=mt)
                    if cm is not None:
                        self.custom_metrics[mode]['preproj_eff_rank'].update(torch.tensor(cm[0]))
                        self.custom_metrics[mode]['preproj_sv0'].update(torch.tensor(cm[1]))
                    # POST-proj: the FAST <video> tokens actually spliced into the LLM (the channel that was
                    # observed to collapse to rank ~1). This is the TRUE post-proj rank.
                    ff0 = getattr(bb, '_rankreg_fast_feats', None)
                    if ff0 is not None:
                        cmf = collapse_metrics(ff0, max_tokens=mt)
                        if cmf is not None:
                            self.custom_metrics[mode]['postproj_fast_eff_rank'].update(torch.tensor(cmf[0]))
                            self.custom_metrics[mode]['postproj_fast_sv0'].update(torch.tensor(cmf[1]))
                if self._rankreg_lambda > 0:
                    from swift.model.vision_3d.slowfast import rankreg_loss
                    rr = rankreg_loss(feats, gamma=self.args.vision_3d_rankreg_gamma,
                                      var_w=self.args.vision_3d_rankreg_var_w,
                                      cov_w=self.args.vision_3d_rankreg_cov_w, max_tokens=mt)
                    loss = loss + self._rankreg_lambda * rr.to(loss.device)
                    self.custom_metrics[mode]['rankreg'].update(rr.detach())
                # Fast-channel anti-collapse (post-proj <video> tokens) -- the fast tokens collapse to rank ~1
                # while the pre-proj encoder stays high-rank; rank-reg them directly.
                if self._rankreg_fast_lambda > 0:
                    ff = getattr(bb, '_rankreg_fast_feats', None)
                    if ff is not None:
                        from swift.model.vision_3d.slowfast import rankreg_loss
                        rrf = rankreg_loss(ff, gamma=self.args.vision_3d_rankreg_gamma,
                                           var_w=self.args.vision_3d_rankreg_var_w,
                                           cov_w=self.args.vision_3d_rankreg_cov_w, max_tokens=mt)
                        loss = loss + self._rankreg_fast_lambda * rrf.to(loss.device)
                        self.custom_metrics[mode]['rankreg_fast'].update(rrf.detach())
                # Orthogonality + weight-decay on proj_fast weights: keep the projection full-rank (so it can't
                # squash the rich encoder features to a low-rank output) and bound its norm.
                pf = getattr(bb, 'proj_fast', None)
                if pf is not None and (self._proj_ortho_lambda > 0 or self._proj_fast_wd > 0):
                    import torch.nn as _nn
                    ws = [m.weight for m in pf.modules() if isinstance(m, _nn.Linear)]
                    if self._proj_ortho_lambda > 0 and ws:
                        from swift.model.vision_3d.slowfast import ortho_reg
                        orth = sum(ortho_reg(w) for w in ws)
                        loss = loss + self._proj_ortho_lambda * orth.to(loss.device)
                        self.custom_metrics[mode]['proj_fast_ortho'].update(orth.detach())
                    if self._proj_fast_wd > 0 and ws:
                        l2 = sum((w.float() ** 2).sum() for w in ws)
                        loss = loss + self._proj_fast_wd * l2.to(loss.device)
                bb._rankreg_fast_feats = None
                bb._rankreg_feats = None
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)
