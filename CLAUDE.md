# CLAUDE.md — Multilingual Prompt-Attack Classifier (RMIT Hackathon 2026, Day 1)

You are helping **Member B, the main implementer**. B writes and maintains all of the team's code. `PLAN.md` holds the team strategy and is the source of truth. Read it first. This file covers what to build and the rules the code must follow.

## The task in one paragraph
Classify each prompt (PT/FR/IT/EN) as `BENIGN`, `JAILBREAK` or `PROMPT_INJECTION`. `train.csv` has 14,000 rows with columns `id, text, language, label`; `test.csv` has 6,000 rows with no label. The metric is **macro-F1**. Classes are imbalanced (62/31/7), so the rare `PROMPT_INJECTION` class counts as much as the others. The submission is `id,label` with exactly 6,000 rows. The team has 6 submissions in total, and the best one counts.

## Hard rules (never break these)
- **The only backbone is `FacebookAI/xlm-roberta-large`.** Load it from the Kaggle model dataset that is attached to the notebook. The one exception: if A switches the team to the `FacebookAI/xlm-roberta-base` fallback (see PLAN.md), use that for every variant. Never suggest, download or load any other pretrained model, tokenizer or embedding. That includes translation models and sentence encoders.
- **No pseudo-labelling.** Train on `train.csv` only. Use `test.csv` only for prediction.
- **Offline.** Load with `local_files_only=True` from `/kaggle/input/...`. Don't `pip install` anything; use the packages Kaggle already has (torch, transformers, scikit-learn, pandas, numpy).
- **B never submits to the leaderboard.** A (the leader) does. The code writes CSVs, and a human decides what to upload.
- The final notebook must say on its first line: `Model: FacebookAI/xlm-roberta-large`.

## Environment
- Runs on Kaggle notebooks with a **T4×2** GPU (16GB each, **no bf16**, so use fp16 autocast + GradScaler). Each member has their own account and a limited weekly GPU quota, so **don't waste GPU time**. Make dry runs small.
- Paths are not known in advance. Make them CLI args with discoverable defaults, and print `ls /kaggle/input` once:
  - data: `--data_dir` (contains `train.csv`, `test.csv`)
  - model: `--model_dir`, usually something like `/kaggle/input/xlm-roberta-large/transformers/default/1`
  - outputs: `--out_dir` (default `/kaggle/working/outputs`). `/kaggle/working` is ~20GB, so **never save model checkpoints to disk**. Keep the best `state_dict` in CPU RAM instead.
- The code is developed in this repo, uploaded to Kaggle as a dataset or utility script, and run from a notebook cell with `!python src/train.py ...`.

## Repo layout to build
```
src/
  common.py              # LABELS, label map, paths, seeding, metric helpers
  make_folds.py          # -> folds.csv
  train.py               # one variant, one or more folds -> per-fold npz + json
  merge.py               # per-fold files -> oof_<variant>.npy / test_<variant>.npy
  blend.py               # weight search + class-bias search -> submission CSV + report
  validate_submission.py # format checks, used by blend.py and on its own
notebooks/
  run_variant.ipynb      # thin Kaggle runner: 2 folds in parallel, one per GPU
  final.ipynb            # clean end-to-end notebook for judging
PLAN.md
CLAUDE.md
```
Keep it small and flat. The deadline is 6 hours, so build no frameworks, config systems or abstractions beyond this.

## Fixed contract (other members' code and checks depend on it)
- **Labels:** `LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]`, `label2id = {BENIGN:0, JAILBREAK:1, PROMPT_INJECTION:2}`. Define this once in `common.py` and import it everywhere.
- **`folds.csv`:** columns `id, fold`. 5 folds from `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, stratified on `label + "_" + language`. Generate it **once** and commit it; never regenerate it mid-competition.
- **Per-fold outputs**, in `out_dir`, named by `--variant` (e.g. `v1`, `v2_len512_ht`):
  - `oof_<variant>_f<k>.npz`: `ids` (val fold ids, str) and `probs` (float32, N×3 softmax)
  - `test_<variant>_f<k>.npz`: `ids` (all 6000 test ids) and `probs` (6000×3)
  - `log_<variant>_f<k>.json`: args, seed, and per epoch: train loss, val macro-F1, per-class F1, per-language macro-F1, minutes elapsed; also the best epoch
  - Write all three as soon as a fold finishes.
- **Merging and alignment always go by `id`**, never by row position. `merge.py` builds `oof_<variant>.npy` (14000×3, train.csv order) and `test_<variant>.npy` (6000×3, fold mean, test.csv order). It fails loudly if any id is missing or duplicated.

## `train.py` spec
CLI args: `--variant --data_dir --model_dir --out_dir --folds 0,1 --max_len 384 --truncation head|headtail --head_frac 0.75 --lr 1e-5 --llrd 1.0 --epochs 3 --bs 8 --grad_accum 2 --warmup 0.1 --wd 0.01 --seed 42 --lang_prefix (flag) --grad_ckpt (flag) --smoke (flag)`.

- **Model:** `AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3, local_files_only=True)`.
- **Text input:** if `--lang_prefix` is set, use `f"[{language}] {text}"`.
- **Truncation:**
  - `head` is normal truncation.
  - `headtail` tokenizes without truncation. If the text is too long, keep the first `head_frac × budget` tokens and the last `(1 − head_frac) × budget` tokens, where budget is `max_len − 2`. Then re-add `<s>` and `</s>`.
  - Pre-tokenize once and cache in memory.
- **Batching:** dynamic padding. In training, group by length: shuffle the rows, cut them into chunks of 50 batches, sort each chunk by length, split it into batches, then shuffle the batches. Most prompts are under 128 tokens, so this makes 384 cost about the same as 256. At inference, sort by length so it runs fast.
- **Loss:** cross-entropy with class weights ∝ `1/sqrt(class_freq)` from the training fold, normalised to mean 1.
- **Optimiser:**
  - AdamW, with no weight decay on bias or LayerNorm.
  - If `--llrd < 1`: each encoder layer `i` (of 24) gets `lr × llrd^(24 − i)`, the embeddings get `lr × llrd^25`, and the classifier head gets `lr`.
  - Linear warmup, then linear decay.
- **Precision:** fp16 via `torch.autocast("cuda", dtype=torch.float16)` + `GradScaler`, with grad clip 1.0.
- **Per epoch:** evaluate on the val fold and compute macro-F1. Keep the best epoch's weights as a CPU `state_dict` copy.
- **Collapse guard:** if val macro-F1 < 0.40 after epoch 1, log it, re-seed (`seed + 1000`), and restart that fold once. If it collapses again, exit non-zero.
- **After the last epoch:** load the best state, predict val + test, and write the 3 files. Then free GPU memory before the next fold (`del model; gc.collect(); torch.cuda.empty_cache()`).
- **Parallel folds:** the notebook runs two processes, `CUDA_VISIBLE_DEVICES=0 python train.py --folds 0,2,4 …` and `CUDA_VISIBLE_DEVICES=1 python train.py --folds 1,3 …`, logging to separate files. The script must not assume it owns both GPUs.
- **`--smoke`:** 500 train rows, 200 test rows, 1 epoch, fold 0 only. It must finish in a few minutes and prove the whole path works, including file writing.
- **Logging:** print timing per epoch, and an ETA after the first 50 steps, so the team can plan GPU time.

## `blend.py` spec
- Input: a list of variants (with an optional fixed weight for each).
- **Weight search:** a simplex grid with step 0.05 over ≤4 variants, maximising OOF macro-F1 of the argmax.
- **Class-offset search:** `pred = argmax(log(p + 1e-9) + [0, b_jb, b_pi])`, with `b_jb, b_pi` ∈ [−1.5, 1.5] step 0.05, maximising OOF macro-F1. Loop over the grid in order of `abs(b_jb) + abs(b_pi)` and replace the best only on a strict improvement, so a tie keeps the smallest offset instead of the grid edge.
- Optional `--per_lang_bias`: a separate `(b_jb, b_pi)` for each language. It is off by default, and it should warn when a language × class cell has fewer than 150 rows.
- **Output:**
  - `submission_<name>.csv` (columns `id,label`, test.csv order). Run `validate_submission` on it before writing "OK".
  - `report_<name>.json`: weights, offsets, OOF macro-F1 raw vs tuned, per-class and per-language F1, confusion matrix, predicted label distribution on test.
- Print a one-line summary that A can paste into the submission log.

## `validate_submission.py`
Check that the header is exactly `id,label`, there are 6000 rows, the id set equals the `test.csv` id set with no duplicates, labels are only the 3 valid strings, and there are no NaNs. Also print the label distribution, and warn if any class is under 2% or over 85%. Exit non-zero on any failure.

## Build order and time targets (T = minutes from start)
1. **T+0–30:** `common.py`, `make_folds.py` → `folds.csv`, and data loading. Share `folds.csv` with the team right away.
2. **T+30–70:** `train.py` + `validate_submission.py`. Run `--smoke` locally on CPU (tiny) or on Kaggle. Then do a 500-row GPU dry run and confirm the predictions are not all one class.
3. **T+70–100:** start the **V1 5-fold** run (`--variant v1 --max_len 384 --truncation head`). Also give A the fold-0 test probs at `max_len 128, 2 epochs` for Submission #1.
4. **T+100–180:** `merge.py` + `blend.py`. Test them on the V1 folds as they finish.
5. **T+180–310:** support the blends, fix bugs, and add the variant flags A asks for.
6. **T+250–335:** `notebooks/final.ipynb`: model line at the top, then folds → training commands → merge → blend → submission, with seeds and variant settings written out. **Code freeze at T+310.**

## Variants the code must support (from PLAN.md)
| Variant | Flags |
|---|---|
| V1 baseline | `--max_len 384 --truncation head` |
| V2 long-context | `--max_len 512 --truncation headtail --head_frac 0.75` (add `--bs 4 --grad_accum 4 --grad_ckpt` if it runs out of memory) |
| V3 regularised | `--max_len 384 --truncation headtail --llrd 0.9 --lang_prefix` |
| V4 / V5 | the best of the above with `--epochs 4` (if its fold logs show the last epoch is best) or `--seed 43` / `--seed 44` |

## Working style for this session
- Write simple, readable, single-file scripts. Comment only where something is not obvious, such as the head+tail truncation or the LLRD grouping.
- Before you say something works, run the smoke test or a quick check. When a step is untested, say so.
- Never change the label map, the fold file or the output file names without B confirming. Teammates' code depends on them.
- If you spot something that would hurt the score (data leakage, row misalignment, a train/test distribution gap), raise it straight away, even when you weren't asked.
- Status of this repo: **planning is done and no code has been written yet.** Start with step 1 of the build order.
