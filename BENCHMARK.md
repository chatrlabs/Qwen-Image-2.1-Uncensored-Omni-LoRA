# Omni LoRA Benchmark

This page reports how the two public Omni checkpoints compare with the original Qwen-Image-2.1 base model, and how we measured it. Both checkpoints come from the same training run:

| Checkpoint | Training step | Where to get it |
|---|---|---|
| Omni LoRA v0.1 | 32,000 (≈1.75 epochs) | [chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA-v0.1](https://huggingface.co/chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA-v0.1) |
| Omni Quantized | 90,000 (≈4.92 epochs) | [chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized](https://huggingface.co/chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized) |

Both are rank-128 adapters trained at 1024 px on the same data. Step 90,000 was evaluated as a BF16 LoRA fused into the base model at strength 1.0. The quantized files were not benchmarked separately.

## Summary

All scores are 0–100; higher is better. ± is one standard error.

| Benchmark | Base | Step 32,000 | Step 90,000 |
|---|---|---|---|
| Adult prompt adherence (960 images) | 85.1 ± 0.9 | 87.4 ± 0.8 | **89.7 ± 0.7** |
| Anatomy (960 images) | **70.3 ± 1.4** | 67.7 ± 1.4 | 69.4 ± 1.4 |
| General SFW prompts (300 PartiPrompts) | **79.8 ± 2.0** | 75.8 ± 2.2 | 79.6 ± 2.0 |

Paired differences on the same prompts and seed. \* marks a difference larger than two standard errors.

| Benchmark | Step 32,000 − Base | Step 90,000 − Base | Step 90,000 − Step 32,000 |
|---|---|---|---|
| Adult prompt adherence | +2.3 ± 0.8 \* | +4.6 ± 0.8 \* | +2.3 ± 0.8 \* |
| Anatomy | −2.6 ± 1.1 \* | −0.9 ± 1.1 | +1.7 ± 1.0 |
| General SFW prompts | −4.0 ± 1.8 \* | −0.2 ± 1.9 | +3.8 ± 1.5 \* |

What this shows:

- **Prompt adherence on adult content improves with training.** Step 90,000 follows prompts best, with a gain over the base model at every caption length.
- **The base model is already capable here.** It scores 85.1, so the adapter's gain is a few points, not a change from unusable to usable.
- **Step 32,000 costs some general ability; step 90,000 recovers it.** On ordinary SFW prompts, step 32,000 is 4 points below the base model, while step 90,000 is within noise of the base model.
- **Anatomy is not improved.** Neither checkpoint beats the base model on anatomy. Step 32,000 is slightly worse; step 90,000 is within noise of the base model.

## Detailed results

### Adult prompt adherence by caption length

Every test image has four captions of increasing detail. Each caption is rendered and judged separately (240 images per length).

| Caption length | Base | Step 32,000 | Step 90,000 | 32,000 − Base | 90,000 − Base |
|---|---|---|---|---|---|
| Tiny | 81.1 ± 1.9 | 83.2 ± 1.9 | 87.1 ± 1.6 | +2.1 ± 1.8 | +6.0 ± 1.7 \* |
| Short | 85.3 ± 1.7 | 87.8 ± 1.5 | 87.8 ± 1.4 | +2.5 ± 1.6 | +2.4 ± 1.5 |
| Medium | 85.1 ± 1.8 | 88.0 ± 1.6 | 90.0 ± 1.4 | +2.9 ± 1.7 | +4.9 ± 1.8 \* |
| Long | 88.9 ± 1.5 | 90.5 ± 1.3 | 93.9 ± 0.9 | +1.6 ± 1.6 | +5.0 ± 1.3 \* |
| **All** | 85.1 ± 0.9 | 87.4 ± 0.8 | 89.7 ± 0.7 | +2.3 ± 0.8 \* | +4.6 ± 0.8 \* |
| Images judged a match (score > 50) | 86.6% | 89.1% | 92.7% | | |

### Anatomy by caption length

| Caption length | Base | Step 32,000 | Step 90,000 |
|---|---|---|---|
| Tiny | 67.2 ± 2.9 | 66.0 ± 2.9 | 68.5 ± 2.9 |
| Short | 72.1 ± 2.8 | 68.1 ± 2.9 | 71.2 ± 2.8 |
| Medium | 72.1 ± 2.8 | 69.6 ± 2.8 | 71.7 ± 2.7 |
| Long | 69.9 ± 2.8 | 67.0 ± 2.9 | 66.3 ± 2.9 |
| **All** | 70.3 ± 1.4 | 67.7 ± 1.4 | 69.4 ± 1.4 |
| Images with a likely severe problem (score < 10) | 25.5% | 27.2% | 25.9% |

### General SFW prompts by PartiPrompts category

| Category | Prompts | Base | Step 32,000 | Step 90,000 |
|---|---|---|---|---|
| People | 120 | 82.2 | 76.1 | 80.4 |
| Animals | 18 | 53.5 | 57.2 | 66.7 |
| Artifacts | 18 | 69.2 | 59.0 | 63.1 |
| Arts | 18 | 80.8 | 67.7 | 83.7 |
| Food & Beverage | 18 | 70.0 | 72.0 | 69.4 |
| Illustrations | 18 | 83.3 | 91.5 | 82.1 |
| Indoor Scenes | 18 | 81.9 | 89.4 | 91.5 |
| Outdoor Scenes | 18 | 98.1 | 99.4 | 93.6 |
| Produce & Plants | 18 | 80.9 | 75.9 | 77.9 |
| Vehicles | 18 | 89.0 | 84.2 | 86.4 |
| World Knowledge | 18 | 75.7 | 59.8 | 76.2 |
| **All** | 300 | 79.8 ± 2.0 | 75.8 ± 2.2 | 79.6 ± 2.0 |

Categories other than People have only 18 prompts each, so their differences are mostly noise.

## Method

### Generation

All three models were rendered with the same pipeline and settings. The only difference is whether the adapter is loaded.

| Setting | Value |
|---|---|
| Base model | Qwen-Image-2.1 (non-distilled) |
| Pipeline | Diffusers `QwenImage21Pipeline` with its default dynamic-shift scheduler |
| Adapter | Fused into the base model at strength 1.0 (none for the base row) |
| Steps / guidance | 35 steps, true CFG 5.0 |
| Negative prompt | A single space (an empty string disables true CFG in this pipeline) |
| Seed | 42 for every prompt (CPU generator) |
| Resolution | About 1 megapixel, keeping each test image's aspect ratio; 1024 × 1024 for PartiPrompts |

The 35 steps / CFG 5.0 setting was chosen by a small sweep on step 90,000 (5 prompts, 10–40 steps, CFG 2–5) and then used unchanged for every model. This may give step 90,000 a slight advantage.

### Test sets

**Adult prompt set.** 240 adult photographs, 20 from each of 12 content categories, sampled with a fixed seed from a held-out split. Each image has four captions written by a vision-language model: tiny, short, medium, and the original long caption (median 5, 11, 31 and 58 words). This gives 960 prompts. We checked every test image by SHA256 against the full Qwen-Image-2.1 training pool (587,892 images): none of the 240 appear in it. The test images and captions are not released because they contain explicit content from a private dataset.

**General SFW set.** 300 prompts from Google's [PartiPrompts](https://github.com/google-research/parti): drawn once with a fixed seed, 120 from People and 18 from each of the 10 other categories. We left out the Abstract category, which has nothing concrete to check, and drew the other categories only from prompts that mention no person, so they test non-people content. The People category is weighted up because people are what the adapter is trained on. This set measures whether the adapter degrades ordinary, non-adult generation.

### Scoring

Each image is scored by a local vision-language model asked a single Yes/No question. The score is the model's probability of "Yes", normalized over Yes and No, read from the first answer token with thinking disabled and temperature 0. A score above 50 counts as a pass.

**Prompt adherence** (adult set and SFW set). The judge is [Qwen3.8-27B Uncensored (HauhauCS, Q8_K_P GGUF)](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF), served with llama.cpp, with the image downscaled to at most 1024 px. Adult set question:

> Does this image match the following description in every detail (people, poses, actions, clothing, setting, viewpoint)?
> Description: {caption}
> Answer with Yes or No only.

The SFW set uses the same question without the parenthetical list. In a pilot on hand-built mismatched pairs, this judge separated matching from non-matching image–caption pairs with an AUC of 0.94.

**Anatomy.** The judge is [Gemma 4 31B-it](https://huggingface.co/google/gemma-4-31B-it) in BF16. It is asked whether there is a problem, and the score is 1 − P(problem):

> This image was generated from the description below. Check the people in it: their anatomy (limbs, hands, feet, torso, breasts, genitals, where bodies meet) and whether the number of people, their sex, their body positions and the sexual act match the description. Is there any such problem in the image?
> Description: {caption}
> Answer with Yes or No only.

We picked this judge and question as the best of 7 vision-language models × 3 questions on 99 renders labelled by hand. It reached an AUC of 0.71 against the human labels, so treat this benchmark as a coarse signal: it is useful for comparing averages, not for judging single images.

### Statistics

- **Mean ± standard error** over prompts.
- **Paired differences** are computed per prompt (same caption, same seed) and then averaged, which removes prompt difficulty from the comparison. A difference larger than two standard errors is marked \*.

## Limitations

- One seed and one image per prompt. Differences of 1–2 points are within noise unless marked.
- Scores come from AI judges, not human ratings. Absolute values depend on the judge and question; comparisons between models under the same judge are more reliable than the numbers themselves.
- Image editing is not covered by this benchmark.
- The anatomy judge agrees with human labels only moderately (AUC 0.71).
- Results apply to the settings above. Other samplers, step counts or LoRA strengths may rank the checkpoints differently.
