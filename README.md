# Multilingual Prompt-Attack Classification — RMIT Hackathon 2026
## Team: Database Analysis & Management Network (DAMN)

Our team's solution and write-up for Challenge 1: label each prompt (Portuguese, French, Italian or English) as **BENIGN**, **JAILBREAK** or **PROMPT_INJECTION**, scored by macro-F1.

**In one line:** we fine-tuned one approved model, `patronus-studio/wolf-defender-prompt-injection` (a safety-tuned ModernBERT), with 5-fold cross-validation, and combined three runs of it by majority vote.

| Submission | Model | Our validation | Leaderboard |
|---|---|---|---|
| Another team's recipe, backbone swapped to XLM-R-large | `FacebookAI/xlm-roberta-large` | 0.9695 (one 1,400-row holdout) | 0.95267 |
| **main-v2**: 5-fold, 384 tokens | wolf-defender | **0.9674** (out-of-fold, all 14,000 rows) | **0.97069** |
| main-v3: 1,024 tokens + label smoothing | wolf-defender | 0.9649 (mean of fold scores) | 0.97039 |
| **Final** (`main-final.ipynb`): seed 43 + vote with main-v2 and main-v3 | wolf-defender | — | Public (30% of dataset): 0.97218, (Private 70% of dataset): 0.97741 |

---

## 1. The challenge
- **Data:** `train.csv` has 14,000 labelled prompts; `test.csv` has 6,000. Every language has the same split: 62% BENIGN, 31% JAILBREAK, 7% PROMPT_INJECTION.
- **Metric:** macro-F1, so the rare PROMPT_INJECTION class counts as much as BENIGN.
- **Rules:** fine-tune one model from the approved list; 6 leaderboard submissions; 6 hours of competing time on Day 1; AI coding tools allowed.

## 2. How we worked
Four members on separate Kaggle accounts, which gave us four GPU quotas to run experiments in parallel.

| Member | Role |
|---|---|
| A — Leader | Set priorities at checkpoints, ran extra training variants, owned every leaderboard submission |
| B — Implementer | Wrote and maintained the training pipeline |
| C — Generalist | Data checks, baseline, error analysis, file checkers; challenged the plan to avoid tunnel vision |
| D — Backup | Spare GPU slot, reruns, reproducibility checks |

`PLAN.md` holds the strategy we agreed at the start (roles, timeline, submission rules) and `CLAUDE.md` the brief for our AI coding assistant. Both describe the **initial** plan (XLM-R-large); section 3 explains why we changed course.

## 3. What we did, in order
1. **Data checks and a baseline** (`old/main-v0.ipynb`). Test has the same language mix as train (pt 45%, fr 25%, it 20%, en 10%), only 8 duplicate prompts, and almost no train/test overlap. A character TF-IDF + logistic regression baseline scored **0.9155** out-of-fold. It wasn't submitted (it isn't an approved model) but served as a sanity floor. Portuguese was clearly the weakest language, with most errors between BENIGN and JAILBREAK.
2. **Model choice.** We compared the approved list and planned around XLM-R-large, the strongest general multilingual encoder. `test.ipynb` was a 10-minute dry run of it on a 3,000-row sample.
3. **Learning from other teams.** We studied notebooks shared by teams scoring about 0.93, 0.94 and 0.97 (not included in this repo). The decisive finding: the 0.97 team used **wolf-defender**, a model already trained to detect prompt injection, and it beat XLM-R-base by a wide margin on their validation. The others contributed smaller ideas: keeping the start and end of long prompts, length-grouped batches, and class-threshold tuning. We rewrote these as our own pipeline (`old/main-v1.ipynb`, then `old/main-v2.ipynb`).
4. **main-v2 → 0.97069.** Wolf-defender with 5-fold cross-validation on both T4 GPUs. This gave an out-of-fold prediction for every training row, and the test prediction is the mean of the 5 fold models.
5. **XLM-R-large check → 0.95267.** Swapping XLM-R-large into the 0.93 team's recipe lifted it from 0.93, but it stayed well behind wolf-defender.
6. **main-v3 → 0.97039.** Two ideas at once: 1,024 tokens instead of 384, and label smoothing 0.05. No gain. Only 3.8% of prompts are longer than 384 tokens, so the longer context had little to work on. Label smoothing also left the model under-trained in 4 epochs: its best epoch was the last one in 4 of 5 folds.
7. **Final submission** (`main-final.ipynb`). main-v2 retrained with seed 43, then a 2-of-3 majority vote with main-v2's and main-v3's submissions. All three are fine-tunes of the same model.

## 4. The final pipeline (`main-final.ipynb`)
| Step | What it does |
|---|---|
| 1. Training script | Loads wolf-defender and replaces its 2-class head with a 3-class one. Prompts get up to 384 tokens; longer ones keep their first 2/3 and last 1/3, since attacks often sit at the end. lr 3e-5, effective batch 64, 4 epochs, keeping each fold's best epoch. Class weights ∝ 1/√frequency; fp16; length-grouped batches |
| 2. Train 5 folds | Folds stratified by label + language. Folds 0, 2, 4 on one T4, folds 1, 3 on the other; about 35 min |
| 3. Out-of-fold score | Joins the folds' validation predictions: our estimate of test performance |
| 4. Class offsets | Tries small shifts to the JAILBREAK / PROMPT_INJECTION scores, kept only if they also help on held-out rows. For main-v2 they didn't, so the offsets stayed 0. Writes `submission.csv` |
| 5. Vote | 2-of-3 majority with main-v2 and main-v3; where all three disagree, this run decides. A guard stops the vote if the attached files aren't those two runs. Writes `submission_vote.csv` |

**main-v2 in detail:** out-of-fold macro-F1 0.9674. By class: BENIGN 0.978, JAILBREAK 0.963, PROMPT_INJECTION 0.961. By language: it 0.980, fr 0.977, pt 0.960, en 0.953.

## 5. What we learned
- **The backbone mattered most.** A model already trained for prompt-injection detection beat the strongest general-purpose encoder we tried (0.971 vs 0.953). Reading other teams' notebooks found it faster than our own planning did.
- **Trust the 14,000-row out-of-fold score, not a small holdout.** XLM-R-large looked like 0.97 on its 1,400-row holdout and scored 0.953 on the leaderboard. Our 5-fold estimates landed within 0.004 of the leaderboard.
- **Compare runs fold by fold, and change one thing per run.** We first compared main-v2 and main-v3 on fold 4's line alone. Across all folds the gap was within run-to-run noise, and with three changes in one run we couldn't tell which one mattered.
- **Leaderboard noise is about ±0.003,** so 0.97069 vs 0.97039 is a tie.
- **Name output files by run, not by order.** Two of our submission files ended up with swapped names, which briefly sent the vote and part of our analysis to the wrong run. We caught it by matching each file's label shares against the shares each notebook printed. The final notebook now refuses to vote if the files don't look like the same model.
- **Ensembles of the same model are cheap insurance.** Averaging 5 fold models already lifted the leaderboard score above the individual fold scores. Separate runs disagreed on only about 1% of test rows (65 rows between main-v2 and main-v3).

## 6. Repository layout
```
main-final.ipynb        final submission notebook (seed 43 + vote)
old/main-v2.ipynb       best single run (0.97069), with its Kaggle outputs
old/main-v3.ipynb       1,024 tokens + label smoothing (0.97039)
old/main-v1.ipynb       first wolf-defender version: one holdout + full-data retrain (not submitted)
old/main-v2.1.ipynb     main-v2 plus a per-language 62/31/7 relabelling cell (not used for the final)
old/main-v0.ipynb       data checks, TF-IDF baseline, file checkers
test.ipynb              XLM-R-large dry run
submissions/            our leaderboard files: submission-1.csv = XLM-R-large run,
                        submission-xlm.csv = main-v2 (misnamed), submission-3.csv = main-v3
datasets/               competition data
PLAN.md, CLAUDE.md      initial strategy and AI-assistant brief
```

## 7. Reproducing the final submission
1. On Kaggle, open `main-final.ipynb` with GPU T4×2 and Internet off.
2. Attach the competition data, the `wolf-defender-prompt-injection` model, and a dataset with main-v2's and main-v3's submission files. Their names are set in step 5 (`EARLIER`).
3. Run All (about 35–40 min). `submission_vote.csv` is the final file.

## 8. Rules and tools
- One approved model in every submitted solution; the XLM-R-large run was a separate experiment and is not part of the final vote.
- Trained on `train.csv` only, with `test.csv` used only for prediction; no pseudo-labelling; everything runs offline.
- We used Claude Code, an AI coding assistant, to plan, write the notebooks, smoke-test them on CPU before using GPU time, and analyse logs and submissions.
