# Single-GPU 70B Inference on AMD MI300X — vLLM / ROCm Benchmarks

Can a 70B-class open-weight model run *well* on a single GPU — no tensor
parallelism, no model sharding? On most hardware the answer is no: 70B in
BF16 needs ~140 GB of weights alone, far beyond an 80 GB H100 or a 15 GB
free-tier notebook GPU. The AMD Instinct MI300X ships 192 GB of HBM3 at
5.3 TB/s, which changes the question from "how do I split it?" to
"how fast does it go, and how long a context can I feed it?".

This repository is an open, reproducible benchmark of exactly that.

## What I will measure

- **Models**: 70B-class open weights (Qwen / Llama 70B families), BF16 and FP8
- **Engine**: vLLM on ROCm, single MI300X, tensor parallel = 1
- **Metrics**
  - Time to first token (TTFT) and inter-token latency
  - Output throughput (tokens/s) across batch sizes and concurrency
  - Long-context behaviour: 32k / 64k / 128k prompts, KV-cache headroom
  - Memory actually used vs. the 192 GB budget
- **Baselines**: published multi-GPU / sharded results for the same models,
  to quantify what zero sharding buys (and costs)

## Side experiment: looped transformers

Recurrent-depth ("looped") transformer inference unrolls the same block
many times, trading compute for parameters. I will prototype small
looped-transformer inference runs in the spare memory headroom and note
where single-device memory is the binding constraint.

## Environment

- AMD Developer Cloud, 1× Instinct MI300X (192 GB HBM3)
- ROCm software image, pre-installed PyTorch / vLLM containers
- Jupyter + SSH access

## Status

- [ ] AMD AI Developer Program cloud credits — application in preparation
- [ ] Instance setup notes + environment capture
- [ ] First benchmark run (BF16, short context)
- [ ] Long-context sweep
- [ ] Results report

## Why publish this

Most published MI300X numbers come from vendors. Independent,
reproducible single-GPU results for 70B-class models — including the
failure modes — help other developers decide whether ROCm is ready for
their workload. Everything here (configs, commands, raw numbers) is
meant to be rerun by anyone with the same instance.
