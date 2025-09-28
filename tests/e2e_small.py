import asyncio
import time
from pathlib import Path

import torch

from delphi.__main__ import run
from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.log.result_analysis import get_agg_metrics, load_data


async def test():
    cache_cfg = CacheConfig(
        dataset_repo="roneneldan/TinyStories", # using a very small dataset for testing in order to avoid downloading a lot of data
        dataset_split="train[:10%]",
        dataset_column="text",
        batch_size=8,
        cache_ctx_len=256,
        n_splits=5,
        n_tokens=10_000,
    )
    sampler_cfg = SamplerConfig(
        train_type="quantiles",
        test_type="quantiles",
        n_examples_train=40,
        n_examples_test=50,
        n_quantiles=10,
    )
    constructor_cfg = ConstructorConfig(
        min_examples=3, #Some absurdly low number to make sure the test runs quickly
        example_ctx_len=32,
        n_non_activating=10, # Make small 
        non_activating_source="random",
        faiss_embedding_cache_enabled=True,
        faiss_embedding_cache_dir=".embedding_cache",
    )
    run_cfg = RunConfig(
        name="test",
        #overwrite=["cache", "scores"], #Commented out so nothing is deleted
        model="EleutherAI/pythia-70m",
        sparse_model="EleutherAI/sae-pythia-70m-32k",
        hookpoints=["layers.3.mlp"],
        explainer_model="HuggingFaceTB/SmolLM-135M-Instruct", #Chat-compatible model
        explainer_model_max_len=1024, # to avoid too much memory usage
        max_latents=100, #max_latents means the max number of latents to explain per input
        seed=22,
        num_gpus=torch.cuda.device_count(),
        max_memory=0.3, #I added this to the config.py and __main__.py to avoid OOM errors
        filter_bos=True,
        verbose=False,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        cache_cfg=cache_cfg,
    )

    start_time = time.time()
    await run(run_cfg)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    scores_path = Path.cwd() / "results" / run_cfg.name / "scores"

    latent_df, counts = load_data(scores_path, run_cfg.hookpoints)
    processed_df = get_agg_metrics(latent_df, counts)

    # Performs better than random guessing
    for score_type, df in processed_df.groupby("score_type"):
        accuracy = df["accuracy"].mean()
        assert accuracy > 0.55, f"Score type {score_type} has an accuracy of {accuracy}"


if __name__ == "__main__":
    asyncio.run(test())