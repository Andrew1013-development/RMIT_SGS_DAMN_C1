# RMIT Hackathon 2026 — Day 1 Strategy: Multilingual Prompt-Attack Classification

## Context
4-person team, 6 hours, 6 leaderboard submissions. Task: 3-class classification (BENIGN 62% / JAILBREAK 31% / PROMPT_INJECTION 7%) on PT/FR/IT/EN prompts, scored by **macro-F1** (the rare PROMPT_INJECTION class weighs as much as BENIGN). **Only one approved backbone may be fine-tuned.** Full Challenge 1 marks come from the first valid submission; after that, the aim is the best leaderboard score.

## Model choice: `FacebookAI/xlm-roberta-large`

### How the approved models compare for this task
| Model | Verdict | Reason |
|---|---|---|
| **xlm-roberta-large** (560M) | **Pick** | Strongest multilingual encoder on the list. Pretrained on 2.5TB CommonCrawl with lots of FR/PT/IT. Proven fine-tuning recipe. The organisers label it "best encoder". Fits on a T4 (16GB) in fp16 at max_len 256–512. |
| xlm-roberta-base | **Fallback** | Proven: another team reached ~0.93 LB with one 3-epoch run (lr 2e-5, max_len 256, ~8 min on a T4). Switch to it only if large keeps collapsing or we run out of time. |
| mmBERT-base | Runner-up | Newer, 8k context, fast. But it has a smaller encoder (~110M non-embedding params), so XLM-R-large usually wins on classification, and it has no proven score on this task. |
| rembert, distilbert-mBERT, mmBERT-small | No | Weaker versions of the above. |
| deberta-v3 PI-v2, wolf-defender | No | Their injection knowledge comes from **English** data, and deberta-v3-base is an English model, so the advantage mostly disappears on FR/PT/IT. They are also only binary-tuned (injection or not), with no JAILBREAK vs PROMPT_INJECTION split. |
| Mistral-7B (QLoRA) | No | Could be strong in theory. But 14k rows × 7B on a T4 takes ~2h per epoch, which leaves no room to iterate or recover in 6h. |
| Granite-3.3-2B (LoRA) | No | Feasible, but a riskier pipeline (padding, head, LoRA config). Expected gain over XLM-R-large is not worth that risk in one day. |

**Limits of XLM-R-large, and how we handle them:**
- **512-token limit, and jailbreak prompts are often long.** Use head+tail truncation.
- **Fine-tuning can collapse** (it predicts one class). Use lr 1e-5, 10% warmup, and restart with a new seed if fold F1 < 0.4 after epoch 1.

## Confirmed rules
1. **Ensembles of several XLM-R-large fine-tunes (folds and seeds) are allowed.**
2. **The best-scoring submission counts.** No submission has to be held back as a hedge, and later ones can take bigger risks.
3. **Every member has their own Kaggle account and GPU quota.** That gives 4 parallel GPU slots, each limited by Kaggle's weekly quota (~30 GPU-h).
4. **No pseudo-labelling.** Train on train.csv only, and use test.csv only for prediction.

**GPU quota budget:** at most ~5h of GPU per account today. That leaves ~25h per account for later in the week. Rules:
- Turn sessions off when idle.
- Don't leave notebooks running interactively while you debug; use "Save & Run All" commits for long runs.
- Dry runs go on CPU or on a 500-row subset.

## Reference point: a ~0.93 team
Another team scored ~0.93 on the LB with a plain recipe: xlm-roberta-base, one model trained on 90% of train, max_len 256, 3 epochs, lr 2e-5, 1/√freq class weights, length-grouped batches, and class offsets tuned on a 1,400-row holdout. Nothing in it goes beyond this plan, so we write our own code and don't copy theirs (sharing code between teams may break the rules). What their numbers tell us:
- **JAILBREAK is the hard class, not PROMPT_INJECTION.** Their F1: BENIGN 0.964, JAILBREAK 0.933, PROMPT_INJECTION 0.957 (precision 1.00). Most errors are BENIGN ↔ JAILBREAK in Portuguese, which matches C's TF-IDF results. Aim error analysis and offset tuning there.
- **Portuguese is their weakest language (0.933), and it has the long prompts.** C measured that 256 tokens cuts 14% of Portuguese rows and 384 cuts 3%, so V1 moves to 384.
- **They were still improving at epoch 3** (0.924 → 0.943 → 0.952), so V4 tries 4 epochs.
- **Their holdout said 0.955, the LB said 0.93.** They picked the epoch and the offsets on the same 1,400 rows (98 PI rows), which inflates the score. Our 5-fold OOF on 14,000 rows is the number to trust.

---

## Roles (all four work on XLM-R-large)
| Member | Role | Responsibilities |
|---|---|---|
| **A: Leader** | Guides the team, owns decisions and submissions, and runs the training variants that push accuracy | Runs the kickoff. Sets priorities at each checkpoint. Is the **only person who submits** and keeps the submission log. Once B's pipeline works, runs the improvement variants (long-context head+tail, LLRD, 4 epochs or a 2nd seed). Owns threshold tuning and blend decisions. Makes the final pick. |
| **B: Main implementer** | Writes and maintains all the code | `folds.csv`, `train.py`, `blend` (blend + class-bias search), `validate_submission()`, the final clean notebook. Runs the **baseline config** (max_len 384, head, lr 1e-5, weighted CE). Fixes bugs that anyone reports. Has the main AI coding-tool session (see `CLAUDE.md`). |
| **C: Generalist / devil's advocate** | Gives broad suggestions and keeps A and B from tunnel vision | EDA and reading raw examples (what makes PI different from JB, per language). Error analysis on each OOF file (confusion matrix, per-language F1, top mistakes). Independent sanity checks (label map, row order, CV↔LB gap). At each checkpoint, asks: "is this the biggest lever, or are we polishing?" Researches quick wins (truncation strategy, prefix, loss). Does not write the core pipeline. |
| **D: Backup** | Spare capacity and failover | Sets up their Kaggle account first, so their GPU is a spare compute slot. Runs extra fold jobs or seeds that A queues. Covers anyone who is blocked (Kaggle issues, OOM debugging, rerunning a crashed fold). Owns reproducibility checks and the README/write-up. Takes over a role if a member gets stuck. |

**Compute:** B's script runs on every account. On T4×2, run 2 folds at once (`CUDA_VISIBLE_DEVICES=0` / `1`). Rough cost: ~20 min per fold at 256, ~45 min per fold at 512. So 5 folds take ~1h at 256 and ~2h at 512 per account. With length-grouped batches, 384 costs about the same as 256, because most prompts are under 128 tokens. A and D are the main extra GPU slots for variants.

**Training variants (all XLM-R-large):**
| Variant | Settings | Run by |
|---|---|---|
| V1 baseline | max_len 384, head truncation, weighted CE, lr 1e-5, 3 epochs, length-grouped batches | B |
| V2 long-context | max_len 512, **head+tail** truncation (first ~380 + last ~130 tokens) | A |
| V3 regularised | max_len 384 head+tail, layer-wise LR decay 0.9, language-tag prefix (`"[fr] " + text`) | D (queued by A) |
| V4 longer or 2nd seed | The best of V1–V3 with 4 epochs if its fold logs show the last epoch is best; otherwise a 2nd seed | A or D |

**Checkpoints (5 min each, led by A, with C's critique):** T+30, T+100, T+180, T+250, T+310.

## Shared contract (agree on this before anyone trains)
- **`folds.csv`** (`id, fold`): 5 folds, `StratifiedKFold(seed=42)` on `label + "_" + language`. Built once by B and shared as a Kaggle dataset.
- **Fixed label map:** `{BENIGN:0, JAILBREAK:1, PROMPT_INJECTION:2}`.
- **Every fold run writes:**
  - `oof_<variant>_f<k>.npz` (the val fold's `ids` + softmax `probs`)
  - `test_<variant>_f<k>.npz` (all 6000 test `ids` + softmax `probs`)
  - `log_<variant>_f<k>.json` (per-epoch macro-F1, per-class F1, per-language macro-F1)
  - The merge step builds `oof_<variant>.npy` (14000×3, train.csv order) and `test_<variant>.npy` (6000×3, mean over folds, test.csv order). It aligns **by id**, never by position.
  - Save after **each fold** and publish a dataset version, so a crash loses at most one fold.
- **One `train.py`**: HF `AutoModelForSequenceClassification(num_labels=3)`, fp16, AdamW, linear schedule with 10% warmup, effective batch 16 (bs 8 × accum 2), class-weighted CE (∝ 1/√freq), keeps the best epoch by fold macro-F1. Arguments: `--variant --max_len --truncation head|headtail --lr --llrd --epochs --seed --folds --lang_prefix`.
- State the chosen model at the top of the notebook: `FacebookAI/xlm-roberta-large`.

## Threshold tuning and blending (B codes it, A decides; the biggest score gain)
1. Blend the variants' probs, with weights searched on the OOF.
2. **Class-offset search:** `pred = argmax(log p + [0, b_JB, b_PI])`, which is the same as multiplying the probs by `e^b`. Grid b ∈ [−1.5, 1.5] step 0.05 (weights ×0.22 to ×4.5), maximising OOF macro-F1. Apply the same offsets to test.
   - JAILBREAK may need a weight below ×0.5, which the old [0.5, 4] grid could not reach.
   - Check the grid in order of `|b_JB| + |b_PI|`, and replace the best only on a strict improvement. Then a tie keeps the smallest offset instead of the edge of the grid.
3. Only tune per-language thresholds if each language × class cell has enough rows. Otherwise skip it (overfit risk).
4. Apply offsets only if C's `check_bias_stability` (in `main.ipynb`) shows a positive mean gain on held-out rows.
5. Trust OOF over the public LB.

## Final model
The weighted mean of the fold models (5 each) from the best 2–3 variants, plus the class offsets tuned on the blended OOF. There is no full-data retrain: the fold-model average already uses all the data and is easier to verify.

---

## Timeline (T = minutes from start)
| Time | A (Leader) | B (Implementer) | C (Generalist) | D (Backup) |
|---|---|---|---|---|
| **0–30** | Kickoff, confirm roles, check every member's GPU quota. Set up Kaggle + GPU. | Build the repo/notebook skeleton, `folds.csv`, data loading | EDA: label × language table, token-length stats (texts > 256/512), duplicates train↔test, conflicting labels. Read 30 PI + 30 JB per language. | Set up Kaggle on their account, attach the XLM-R-large dataset, test GPU + fp16, then help C with the EDA |
| **T+30 checkpoint** | C presents the EDA. A decides max_len and truncation. | | | |
| **30–70** | Review B's code as it lands. Plan the variants V2–V4. | Write `train.py` + `validate_submission()`. Dry run on 500 rows. | Write an independent OOF-F1 / label-distribution checker | Dry run of `train.py` at 512 on their GPU (OOM check), report speeds |
| **70–100** | **Submission #1:** fold 0, max_len 128, 2 epochs (locks marks, tests the format) | Start the **V1 5-fold** run | Check Submission #1 file (row order, label map) | Start **V3 5-fold** (queued by A) |
| **T+100 checkpoint** | Confirm the pipeline works. C challenges the plan. | | | |
| **100–180** | Start the **V2 5-fold** run on their GPU | Write the blend (weight search + class-bias search) | Error analysis on the first finished folds: confusion matrix, per-language F1, top mistakes, starting with Portuguese BENIGN↔JAILBREAK. Suggest fixes. | Watch the runs, rerun any crashed fold, cover whoever is blocked |
| **~180** | **Submission #2:** V1 5-fold + tuned bias. Check that the CV↔LB gap is sane. | | | |
| **T+180 checkpoint** | Choose what goes into the blend | | | |
| **180–250** | Blend V1+V2(+V3) → **Submission #3** (~T+230). Queue V4 on the best config. | Support blending, fix bugs | Critique: is the next step the biggest lever? Check for overfitting to the LB | Run **V4** |
| **T+250 checkpoint** | Decide which variants go into the final blend, and whether a V5 is worth the quota | | | |
| **250–310** | **Submission #4:** best blend + V4 | Start the clean final notebook | Final error review of BENIGN↔JAILBREAK and PI misses. Propose one bold idea for Submission #6. | Reproducibility check of the notebook, start the README |
| **310–335** | **Code freeze.** **Submission #5:** best blend by OOF. | Finish the clean notebook (model name at the top) | Last sanity check of the final file | Finish the README / write-up |
| **335–360** | **Submission #6:** the bold variant. Since the best score counts, it carries no downside. Examples: per-language class offsets, a larger PI offset, or a blend weighted toward the long-context variant. | | | |

**Stretch (only if ahead at T+240 and there is quota left):** V5, another seed of the best config. More seeds in the blend give a small but reliable gain.

**Skip these:** other backbones (rule), pseudo-labelling (rule), translation augmentation (needs an unapproved model), big hyperparameter sweeps, TF-IDF as a blend member (adds little next to a ~0.95 transformer and may break the approved-models rule).

## Submission discipline
- Only A submits. Every file goes through `validate_submission()`: 6000 rows, ids equal to the test.csv set, labels in the 3 valid strings, header `id,label`.
- Shared log: `#, time, variant(s), OOF macro-F1 raw/tuned, per-language F1, LB, notes`.
- Submit only when OOF improves by ≥ 0.003, or to test a specific hypothesis.

## Risks and mitigations
- **GPU quota running out:** each member checks their quota at T+0 and T+180. Use the 256/384 variants on a low-quota account, and stop any variant that isn't beating V1 on its first 2 folds.
- **OOM at 512:** bs 4 × accum 4, gradient checkpointing.
- **Collapse:** lr 1e-5, warmup, fixed seeds. Restart a fold if F1 < 0.4 after epoch 1. If large keeps collapsing, switch the whole team to the xlm-roberta-base fallback (lr 2e-5, 3 epochs).
- **Behind schedule:** drop to 3 folds, or one 85/15 holdout plus a full-data retrain.
- **Bugs from AI-generated code** (row order, label map): C's independent checker runs on every output file. It checks the shape, checks that the argmax distribution is near 62/31/7, and recomputes OOF F1.
- **B becomes a bottleneck:** D takes over secondary code tasks (e.g. `validate_submission()`), and A reviews B's code early so problems surface before training starts.
- **Tunnel vision:** C has a standing slot at every checkpoint to challenge the current direction.

## Verification
- The 500-row dry run finishes and its predictions are not all one class.
- The OOF F1 recomputed in the blend matches the trainer's fold logs.
- The tuned OOF F1 is higher than raw, and `check_bias_stability` shows the gain holds on held-out rows.
- Submission #1 is accepted, which confirms the format end-to-end.
- The CV↔LB gap stays steady across submissions. If it jumps, check row order and the label map first.
