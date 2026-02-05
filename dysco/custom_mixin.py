# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, List, Tuple, Dict

import torch
import torch.distributed as dist
from packaging import version
from torch import nn

from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    QuantizedCache,
    StaticCache,
)
from transformers.dynamic_module_utils import (
    check_python_requirements,
    get_cached_module_file,
    get_class_in_module,
    resolve_trust_remote_code,
)
from transformers.generation.utils import GenerationMixin
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.masking_utils import create_masks_for_generate
from transformers.pytorch_utils import isin_mps_friendly
from transformers.tokenization_utils import ExtensionsTrie
from transformers.utils import (
    ModelOutput,
    TransformersKwargs,
    is_accelerate_available,
    is_hqq_available,
    is_optimum_quanto_available,
    is_torchdynamo_exporting,
    logging,
)
from transformers.generation.candidate_generator import (
    AssistantVocabTranslatorCache,
    AssistedCandidateGenerator,
    AssistedCandidateGeneratorDifferentTokenizers,
    CandidateGenerator,
    EarlyExitCandidateGenerator,
    PromptLookupCandidateGenerator,
    UniversalSpeculativeDecodingGenerator,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
from transformers.generation.configuration_utils import (
    ALL_STATIC_CACHE_IMPLEMENTATIONS,
    DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS,
    STATIC_CACHE_IMPLEMENTATIONS,
    GenerationConfig,
    GenerationMode,
)
from transformers.generation.continuous_batching import ContinuousMixin
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    ConfidenceCriteria,
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

from transformers.generation.utils import (
    GenerateOutput,
    GenerateNonBeamOutput,
    GenerateBeamOutput,
    ALL_CACHE_NAMES,
    GENERATION_MODES_MAPPING,
)


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..tokenization_utils_base import PreTrainedTokenizerBase
    from .streamers import BaseStreamer

logger = logging.get_logger(__name__)

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module


@dataclass
class DyscoConfig:
    dysco_qrheads: list = None
    dysco_top_k: int = None
    dysco_top_p: float = None
    dysco_strength: float = None
    dysco_ctx_momentum: float = None
    dysco_ctx_warmup: int = 16
    dysco_interv_warmup: int = None
    dysco_rescale_template: bool = False
    dysco_static_rescaling: bool = False
    dysco_template_seqs: list = None

    # visualization
    return_attention_details: bool = False
    def __post_init__(self):
        # qrheads need to be list of tuples
        if self.dysco_template_seqs is not None:
            self.dysco_template_seqs = [torch.LongTensor(seq) for seq in self.dysco_template_seqs]

def _nucleus_mask(attn: torch.Tensor, p: float):
    """
    attn: [B, L] attention weights (assumed >= 0)
    p: float in (0, 1]
    returns: mask [B, L] (True = kept)
    """
    # normalize if needed
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # sort descending
    sorted_vals, sorted_idx = torch.sort(attn, dim=-1, descending=True)

    # cumulative probability
    cumvals = sorted_vals.cumsum(dim=-1)

    # keep tokens until cumulative prob >= p
    keep_sorted = cumvals <= p

    # always keep at least one token
    keep_sorted[..., 0] = True

    # scatter back to original order
    mask = torch.zeros_like(attn, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_idx, src=keep_sorted)

    return mask


def _aggregate_head_attention(attention_outputs, selected_heads):
    """Extract and average attention weights across selected (layer, head) pairs.

    Handles multi-GPU case where attention outputs from different layers may be on different devices.
    """
    # Get the first tensor to determine target device
    first_layer, first_head = selected_heads[0]
    target_device = attention_outputs[first_layer].device

    # Move all tensors to the same device before stacking
    per_head = [attention_outputs[layer][:, head].to(target_device) for layer, head in selected_heads]
    return torch.stack(per_head, dim=0).mean(dim=0).squeeze(1)


def _apply_importance_momentum(cur_importance, past_importance, momentum):
    """Exponential moving average of token importance scores.

    Handles multi-GPU case where tensors may be on different devices.
    """
    # Ensure past_importance is on the same device as cur_importance
    if past_importance.device != cur_importance.device:
        past_importance = past_importance.to(cur_importance.device)
    cur_importance = cur_importance * (1 - momentum)
    cur_importance[:, :-1] += past_importance * momentum
    return cur_importance


def _select_important_tokens(importance, generation_logging, top_tokens=None, top_percentile=None):
    """Select important tokens via top-k and/or nucleus (top-p) masks.

    Builds whichever masks correspond to non-None parameters, then ANDs them
    together.  When both are active, the intersection is equivalent to the
    former "hybrid" mode: nucleus selects the minimal set covering
    *top_percentile* of total mass, while top-k caps that set at *top_tokens*.

    Updates *generation_logging* in place.
    """
    if top_tokens is None and top_percentile is None:
        raise ValueError("At least one of top_tokens or top_percentile must be provided")

    top_k_mask = None
    nucleus_mask = None
    top_vals = None

    if top_tokens is not None:
        selected_k = min(top_tokens, importance.shape[1])
        top_vals, top_indices = torch.topk(importance, k=selected_k, dim=1)
        top_k_mask = torch.zeros_like(importance, dtype=torch.bool)
        top_k_mask[:, top_indices] = True

    if top_percentile is not None:
        nucleus_mask = _nucleus_mask(importance, top_percentile)

    # Combine masks
    if top_k_mask is not None and nucleus_mask is not None:
        selected_mask = top_k_mask & nucleus_mask
        # Determine which constraint was binding for logging (use mean across batch)
        num_nucleus = nucleus_mask.sum(dim=1).float().mean().item()
        if num_nucleus > top_tokens:
            # top-k was the tighter constraint
            generation_logging["avg_num_token"] += selected_k
            generation_logging["avg_nucleus_mass"] += torch.sum(top_vals).item() / importance.shape[0]  # average per sample
            generation_logging["scale_by_token"] += 1.0
        else:
            # nucleus was the tighter (or equal) constraint
            generation_logging["avg_num_token"] += num_nucleus
            generation_logging["avg_nucleus_mass"] += top_percentile
            generation_logging["scale_by_nucleus"] += 1.0
    elif top_k_mask is not None:
        selected_mask = top_k_mask
        generation_logging["avg_num_token"] += selected_k
        generation_logging["avg_nucleus_mass"] += torch.sum(top_vals).item() / importance.shape[0]  # average per sample
        generation_logging["scale_by_token"] += 1.0
    else:
        selected_mask = nucleus_mask
        generation_logging["avg_num_token"] += selected_mask.sum(dim=1).float().mean().item()
        generation_logging["avg_nucleus_mass"] += top_percentile
        generation_logging["scale_by_nucleus"] += 1.0

    return selected_mask


def _build_intervention_vector(selected_mask, reference_tensor, strength, non_template_mask, dtype=None):
    """Build attention logits intervention vector from selected mask and strength.

    dtype: explicit dtype for the output tensor. When None, inherits from reference_tensor.
    Handles multi-GPU case where tensors may be on different devices.
    """
    if non_template_mask is not None and 0 < strength < 99.0:
        # Ensure non_template_mask is on the same device as selected_mask
        if non_template_mask.device != selected_mask.device:
            non_template_mask = non_template_mask.to(selected_mask.device)
        if selected_mask.shape[1] < non_template_mask.shape[1]:
            selected_mask = selected_mask & non_template_mask[:, :selected_mask.shape[1]]
        else:
            selected_mask[:, :non_template_mask.shape[1]] = (
                selected_mask[:, :non_template_mask.shape[1]] & non_template_mask
            )

    ones_kwargs = {"dtype": dtype} if dtype is not None else {}
    if 0 < strength < 99.0:
        vec = torch.ones_like(reference_tensor, **ones_kwargs)
        vec[selected_mask] = strength
        vec[:, -1] = strength
        vec = torch.log(vec)
    elif strength >= 99.0:
        vec = torch.full_like(reference_tensor, float('-inf'), **ones_kwargs)
        vec[selected_mask] = 0.0
        vec[:, -1] = 0.0
    else:
        raise ValueError(f"Invalid strength: {strength}")
    return vec


def obtain_template_sequence_mask(input_ids, template_sequences: list):
    target_device = input_ids.device
    input_ids = input_ids.to(torch.device("cpu"))
    # input_ids: (batch_size, max_length)
    # return a mask of all template sequences
    batch_size, max_length = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for template in template_sequences:
        template_len = len(template)
        # For each possible starting position
        for i in range(max_length - template_len + 1):
            # Check if template matches at position i
            # Compare input_ids[:, i:i+template_len] with template
            matches = torch.all(input_ids[:, i:i+template_len] == template.unsqueeze(0), dim=1)
            # Mark positions where template matches
            for b in range(batch_size):
                if matches[b]:
                    mask[b, i:i+template_len] = True

    return mask.to(target_device)


class CustomGenerationMixin(GenerationMixin):
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in model classes.
    Inheriting from this class causes the model to have special generation-related behavior, such as loading a
    `GenerationConfig` at initialization time or ensuring `generate`-related tests are run in `transformers` CI.

    A model class should inherit from `GenerationMixin` to enable calling methods like `generate`, or when it
    has defined a custom `generate` method that relies on `GenerationMixin`, directly or indirectly, which
    approximately shares the same interface to public methods like `generate`. Three examples:
        - `LlamaForCausalLM` should inherit from `GenerationMixin` to enable calling `generate` and other public
            methods in the mixin;
        - `BlipForQuestionAnswering` has a custom `generate` method that approximately shares the same interface as
           `GenerationMixin.generate` (it has a few extra arguments, and the same output). That function also calls
           `GenerationMixin.generate` indirectly, through an inner model. As such, `BlipForQuestionAnswering` should
           inherit from `GenerationMixin` to benefit from all generation-related automation in our codebase;
        - `BarkModel` has a custom `generate` method and one of its inner models calls `GenerationMixin.generate`.
            However, its `generate` does not share the same interface as `GenerationMixin.generate`. In this case,
            `BarkModel` should NOT inherit from `GenerationMixin`, as it breaks the `generate` interface.

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`
        - *beam-search decoding* if `num_beams>1` and `do_sample=False`
        - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
        - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    def load_custom_generate(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        trust_remote_code: Optional[bool] = None,
        **kwargs,
    ) -> Callable:
        """
        Loads and returns a custom generate function, given a model repo.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                 Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
            trust_remote_code (`bool`, *optional*):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            **kwargs:
                Additional keyword arguments for remote code loading.

        Raises:
            OSError: If `pretrained_model_name_or_path` does not contain a `custom_generate` subdirectory.

        Returns:
            A callable that can be used to generate text.
        """
        # Fetches the generate.py file from the model repo. If it doesn't exist, a file in `.no_exist` cache directory
        # is created (preventing future hub requests), and an OSError is raised.
        try:
            module = get_cached_module_file(
                pretrained_model_name_or_path, module_file="custom_generate/generate.py", **kwargs
            )
        except OSError:
            raise OSError(
                f"`{pretrained_model_name_or_path}` does not contain a `custom_generate` subdirectory with a "
                "`generate.py` file, can't load the custom generate function."
            )

        # Handle opt-in `trust_remote_code` and related exceptions
        is_local_code = os.path.exists(pretrained_model_name_or_path)
        error_message = (
            f"The repository `{pretrained_model_name_or_path}` contains custom generation code that will override "
            "the default `generate` method."
        )
        resolve_trust_remote_code(
            trust_remote_code,
            pretrained_model_name_or_path,
            has_local_code=is_local_code,
            has_remote_code=not is_local_code,
            error_message=error_message,
        )

        # Load the custom generate function
        check_python_requirements(
            pretrained_model_name_or_path, requirements_file="custom_generate/requirements.txt", **kwargs
        )
        custom_generate_function = get_class_in_module("generate", module)
        return custom_generate_function

    @torch.no_grad()
    def dysco_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[Union[str, Callable]] = None,
        dysco_config: Optional[DyscoConfig] = None,
        # for baselines
        use_attnsharp: bool = False,
        attention_logits_temperature: Optional[float] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Dict[str, Any]]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://huggingface.co/papers/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
                to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
                deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            use_model_defaults (`bool`, *optional*):
                When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
                generation configuration (`model.generation_config`), as opposed to the global defaults
                (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
                `True`.
            custom_generate (`str` or `Callable`, *optional*):
                One of the following:
                - `str` (Hugging Face Hub repository name): runs the custom `generate` function defined at
                  `custom_generate/generate.py` in that repository instead of the standard `generate` method. The
                  repository fully replaces the generation logic, and the return type may differ.
                - `str` (local repository path): same as above but from a local path, `trust_remote_code` not required.
                - `Callable`: `generate` will perform the usual input preparation steps, then call the provided callable to
                  run the decoding loop.
                For more information, see [the docs](../../generation_strategies#custom-generation-methods).
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 0. If requested, load an arbitrary generation recipe from the Hub and run it instead
        trust_remote_code = kwargs.pop("trust_remote_code", None)

        if custom_generate is not None and isinstance(custom_generate, str):
            # Get all `generate` arguments in a single variable. Custom functions are responsible for handling them:
            # they receive the same inputs as `generate`, with `model` instead of `self` and excluding the arguments to
            # trigger the custom generation. They can access to methods from `GenerationMixin` through `model`.
            global_keys_to_exclude = {
                "self",
                "kwargs",
                "global_keys_to_exclude",
                "trust_remote_code",
                "custom_generate",
            }
            generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
            generate_arguments.update(kwargs)

            custom_generate_function = self.load_custom_generate(
                custom_generate, trust_remote_code=trust_remote_code, **kwargs
            )
            return custom_generate_function(model=self, **generate_arguments)

        # 1. Handle kwargs, `generation_config`, validate them and obtain generation mode
        generation_mode_kwargs = self._extract_generation_mode_kwargs(
            custom_generate,
            kwargs,
            synced_gpus,
            assistant_model,
            streamer,
        )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        generation_mode = generation_config.get_generation_mode(assistant_model)
        if isinstance(custom_generate, Callable):
            decoding_method = custom_generate
        else:
            # type() required to access the unbound class-level method
            decoding_method = getattr(type(self), GENERATION_MODES_MAPPING[generation_mode])

        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)

        # Deprecation-related step: set Hub repo for deprecated strategies.
        # NOTE: This must come after initializing generation_config, since we need it to determine if this is a deprecated mode.
        # It must also be before any preparation steps, since Hub repos expect to be loaded before preparation steps.
        # TODO joao, manuel: remove this in v4.62.0
        if deprecated_mode_repo := self._get_deprecated_gen_repo(generation_mode, trust_remote_code, custom_generate):
            return GenerationMixin.generate(
                self,
                inputs=inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                assistant_model=assistant_model,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                use_model_defaults=use_model_defaults,
                custom_generate=deprecated_mode_repo,
                trust_remote_code=trust_remote_code,
                **generation_mode_kwargs,
                **kwargs,
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        # Some generation modes (e.g. assisted) need `inputs_tensor` to rerun encoder.forward()
        if "inputs_tensor" in inspect.signature(decoding_method).parameters.keys():
            generation_mode_kwargs["inputs_tensor"] = inputs_tensor
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # Expand inputs depending on the generation mode
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, generation_mode_kwargs.get("tokenizer"))

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=generation_mode_kwargs.get("tokenizer"),
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        assert GENERATION_MODES_MAPPING[generation_mode] == "_sample"
        # 9. Call generation mode
        if dysco_config is not None:
            result = self._dysco_sample(
                input_ids=input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                dysco_config=dysco_config,
                **generation_mode_kwargs,
                **model_kwargs,
            )
        elif use_attnsharp and attention_logits_temperature is not None:
            result = self._sharp_sample(
                input_ids=input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                attention_logits_temperature=attention_logits_temperature,
                **generation_mode_kwargs,
                **model_kwargs,
            )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def _get_attn_weights(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(1)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        key_states = repeat_kv(key_states, num_key_value_groups)
    
        # Scale before multiplication to prevent overflow
        scale = 1.0 / math.sqrt(head_dim)
        scaled_queries = query_states * scale
        attn_weights = torch.matmul(scaled_queries, key_states.transpose(2,3))

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}")
        
        # make causal mask and add it to attention weights.
        causal_mask = self._get_causal_mask(attn_weights).to(attn_weights.device)
        attn_weights += causal_mask.unsqueeze(0)
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True) # Log-sum-exp of attention weights for numerical stability in softmax.
        attn_weights = torch.exp(attn_weights - attn_lses) # softmax
        return attn_weights

    def _dysco_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        dysco_config: Optional[DyscoConfig] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor, Dict[str, Any]]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample


        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        assert dysco_config is not None
        selected_heads = dysco_config.dysco_qrheads
        context_warmup_steps = dysco_config.dysco_ctx_warmup
        intervention_warmup_steps = dysco_config.dysco_interv_warmup
        ctx_momentum = dysco_config.dysco_ctx_momentum
        strength = dysco_config.dysco_strength
        top_tokens = dysco_config.dysco_top_k
        top_percentile = dysco_config.dysco_top_p
        if top_tokens is None and top_percentile is None:
            raise ValueError("Either top_tokens or top_percentile must be set")
        scale_template_tokens = dysco_config.dysco_rescale_template
        dynamic_rescale = not dysco_config.dysco_static_rescaling
        max_selected_layer = max(selected_heads, key=lambda x: x[0])[0]

        # segment the inputs 
        tf_ending_len = input_ids.shape[-1]
        total_truncate_length = intervention_warmup_steps + context_warmup_steps
        look_back_inputs = input_ids[:, -total_truncate_length:]
        input_ids = input_ids[:, :-total_truncate_length]
        look_back_start_len = tf_ending_len - total_truncate_length
        intervention_start_len = tf_ending_len - intervention_warmup_steps

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        assert is_prefill == True

        attention_details = []
        return_attention_details = dysco_config.return_attention_details
        static_mask_initialized = False
        past_token_importance = None
        if not scale_template_tokens:
            non_template_mask = obtain_template_sequence_mask(input_ids, dysco_config.dysco_template_seqs)
            non_template_mask = non_template_mask.to(input_ids.device)
            non_template_mask = ~non_template_mask # invert the mask to keep non-template tokens
        else:
            non_template_mask = None

        generation_logging = {"avg_num_token": 0.0, "avg_nucleus_mass": 0.0, "scale_by_token": 0.0, "scale_by_nucleus": 0.0, "num_generations": 0}
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            position_idx = cur_len - 1
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if is_prefill:
                outputs = self(**model_inputs, compute_logits=False, return_dict=True)
                is_prefill = False
                past_token_importance = torch.zeros_like(input_ids)
            # in context warmup stage, only get the attention weights, still reading the context
            elif cur_len < intervention_start_len:
                outputs = model_forward(
                        **model_inputs,
                        compute_logits=False,
                        output_attentions=True,
                        return_dict=True,
                    )
                attention_peak = outputs.attentions
                cur_token_importance = _aggregate_head_attention(attention_peak, selected_heads)
                past_token_importance = _apply_importance_momentum(cur_token_importance, past_token_importance, ctx_momentum)

                if return_attention_details:
                    # Only track first sample in batch to minimize memory
                    attention_details.append({
                        "position_idx": position_idx,
                        "attention_weights": cur_token_importance[0].cpu(),
                        "context_scores": past_token_importance[0].cpu(),
                    })
            else:
                if dynamic_rescale:
                    # SPECULATIVE PASS; skip updating the past key value
                    outputs = model_forward(
                        **model_inputs,
                        attention_logits_intervention_vector=None,
                        compute_logits=False,
                        output_attentions=True,
                        skip_update_past_key_value=True,
                        layer_early_stopping=max_selected_layer,
                        return_dict=True,
                    )
                    attention_peak = outputs.attentions
                    # attention_peak: tuple (num_layers) of [batch_size, num_heads, 1, sequence_length]  # since we only forward one token
                    cur_token_importance = _aggregate_head_attention(attention_peak, selected_heads)
                    past_token_importance = _apply_importance_momentum(cur_token_importance, past_token_importance, ctx_momentum)
                    if return_attention_details:
                        # Only track first sample in batch to minimize memory
                        attention_details.append({
                            "position_idx": position_idx,
                            "attention_weights": cur_token_importance[0].cpu(),
                            "context_scores": past_token_importance[0].cpu(),
                        })

                    generation_logging["num_generations"] += 1
                    selected_mask = _select_important_tokens(past_token_importance, generation_logging, top_tokens=top_tokens, top_percentile=top_percentile)
                    attention_logits_intervention_vector = _build_intervention_vector(selected_mask, past_token_importance, strength, non_template_mask)
                # statically rescale
                else:
                    # selected_mask,
                    if not static_mask_initialized:
                        static_mask_initialized = True
                        generation_logging["num_generations"] += 1
                        selected_mask = _select_important_tokens(past_token_importance, generation_logging, top_tokens=top_tokens, top_percentile=top_percentile)
                    # extend selected mask to the length of the input ids
                    # consider 1 always upweight generated tokens
                    selected_mask = torch.cat([selected_mask, torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device)], dim=1)
                    attention_logits_intervention_vector = _build_intervention_vector(selected_mask, input_ids, strength, non_template_mask, dtype=past_token_importance.dtype)

                # second pass, get logits;
                # REAL PASS
                outputs = model_forward(
                    **model_inputs,
                    attention_logits_intervention_vector=attention_logits_intervention_vector,
                    output_attentions=False,
                    return_dict=True,
                )
                # clear the logits if still doing teacher forcing
                if look_back_inputs is not None:
                    outputs.logits = None

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue


            if outputs.logits is not None:
                # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits and outputs.logits is not None:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            teacher_forcing = False
            if look_back_inputs is not None:
                next_tokens = look_back_inputs[:, 0]
                look_back_inputs = look_back_inputs[:, 1:]
                teacher_forcing = True
                if look_back_inputs.shape[1] == 0:
                    look_back_inputs = None
                    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            else:
                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if not teacher_forcing and has_eos_stopping_criteria:
                # if there are still look_back_inputs, we need to use them
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None and not teacher_forcing:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            if teacher_forcing:
                this_peer_finished = False
            else:
                this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        generation_logging["avg_num_token"] /= generation_logging["num_generations"]
        generation_logging["avg_nucleus_mass"] /= generation_logging["num_generations"]
        generation_logging["scale_by_token"] /= generation_logging["num_generations"]
        generation_logging["scale_by_nucleus"] /= generation_logging["num_generations"]
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            if return_attention_details:
                return input_ids, generation_logging, attention_details
            return input_ids, generation_logging

    def _sharp_sample( # for baselines, sharpenning attention or sharpenning system prompt
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        attention_logits_temperature = model_kwargs.pop("attention_logits_temperature", None)
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill:
                outputs = self(**model_inputs, compute_last_logits_only=True, attention_logits_temperature=attention_logits_temperature, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, attention_logits_temperature=attention_logits_temperature, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids, None
