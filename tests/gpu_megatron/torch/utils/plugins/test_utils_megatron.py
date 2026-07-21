# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from _test_utils.torch.megatron.models import get_mcore_qwen3_600m
from _test_utils.torch.megatron.utils import initialize_for_megatron
from transformers import AutoTokenizer

from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.plugins import megatron_generate, megatron_mmlu
from modelopt.torch.utils.plugins.megatron_generate import (
    _is_layerwise_capture_active,
    megatron_prefill,
)

SEED = 1234


class _LayerwiseState:
    def __init__(self, mode):
        self.mode = mode


def test_is_layerwise_capture_active():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    assert not _is_layerwise_capture_active(model)

    model[0]._layerwise_calib = _LayerwiseState("original")
    assert not _is_layerwise_capture_active(model)

    model[1]._layerwise_calib = _LayerwiseState("capture")
    assert _is_layerwise_capture_active(model)


# TODO: move to regression test folder


def _test_megatron_generate_and_mmlu(rank, size, parallelism):
    if parallelism == "tp":
        initialize_for_megatron(tensor_model_parallel_size=size, seed=SEED)
        model = get_mcore_qwen3_600m(tensor_model_parallel_size=size).cuda().eval()
    elif parallelism == "pp":
        initialize_for_megatron(pipeline_model_parallel_size=size, seed=SEED)
        model = get_mcore_qwen3_600m(pipeline_model_parallel_size=size).cuda().eval()
    elif parallelism == "cp":
        initialize_for_megatron(context_parallel_size=size, seed=SEED)
        model = get_mcore_qwen3_600m(context_parallel_size=size).cuda().eval()
    elif parallelism == "dp":
        # Data parallel is implicit: with all model-parallel sizes 1, DP == world size.
        initialize_for_megatron(seed=SEED)
        model = get_mcore_qwen3_600m().cuda().eval()
    else:
        raise ValueError(f"Invalid parallelism: {parallelism}")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # megatron_generate does not support CP (autoregressive decode is not sequence-partitioned).
    if parallelism != "cp":
        messages = [
            {"role": "user", "content": "Give me a short introduction to large language model."}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device="cuda")
        output_ids = megatron_generate(model, model_inputs["input_ids"])
        output_text = tokenizer.batch_decode(output_ids)
        print(rank, output_text)

    assert 0.36 < megatron_mmlu(model, tokenizer, fraction=0.1, batch_size=16) < 0.39


def _test_layerwise_megatron_prefill(rank, size):
    initialize_for_megatron(tensor_model_parallel_size=size, seed=SEED)
    model = get_mcore_qwen3_600m(tensor_model_parallel_size=size).cuda().eval()
    collector = LayerActivationCollector(model)
    decoder_layers = collector.get_decoder_layers(model)
    assert decoder_layers is not None

    collector._patch_all_layers(decoder_layers)
    try:
        input_ids = torch.randint(0, model.vocab_size, (1, 8), device="cuda")
        layer_inputs = collector.get_first_layer_inputs(
            start_layer=0,
            resumed_inputs=None,
            forward_loop=lambda patched_model: megatron_prefill(
                patched_model, input_ids, skip_return_logits=True
            ),
        )
        assert len(layer_inputs) == 1
    finally:
        collector._unpatch_all_layers()


def _test_layerwise_megatron_prefill_pp_rejected(rank, size):
    initialize_for_megatron(pipeline_model_parallel_size=size, seed=SEED)
    model = get_mcore_qwen3_600m(pipeline_model_parallel_size=size).cuda().eval()
    collector = LayerActivationCollector(model)
    decoder_layers = collector.get_decoder_layers(model)
    assert decoder_layers is not None

    collector._patch_all_layers(decoder_layers)
    try:
        input_ids = torch.randint(0, model.vocab_size, (1, 8), device="cuda")
        with pytest.raises(RuntimeError, match="pipeline_model_parallel_size=1"):
            collector.get_first_layer_inputs(
                start_layer=0,
                resumed_inputs=None,
                forward_loop=lambda patched_model: megatron_prefill(
                    patched_model, input_ids, skip_return_logits=True
                ),
            )
    finally:
        collector._unpatch_all_layers()


def test_layerwise_megatron_prefill(dist_workers):
    dist_workers.run(_test_layerwise_megatron_prefill)


def test_layerwise_megatron_prefill_pp_rejected(dist_workers, num_gpus):
    if num_gpus == 1:
        pytest.skip("Pipeline-parallel rejection requires at least 2 GPUs")
    dist_workers.run(_test_layerwise_megatron_prefill_pp_rejected)


@pytest.mark.parametrize("parallelism", ["tp", "pp", "cp", "dp"])
def test_megatron_generate_and_mmlu(dist_workers, parallelism, num_gpus):
    if num_gpus == 1 and parallelism != "tp":
        pytest.skip("Skipping as redundant test on 1 GPU")
    dist_workers.run(_test_megatron_generate_and_mmlu, parallelism=parallelism)
