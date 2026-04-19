---
description: Get fully up to speed on the Silence project from cold start.
---

You are working on **Silence**, a personal silent-speech recognition system (DIY AlterEgo). You have no memory of prior sessions. Before doing anything, load full context by reading, in this order:

1. **`README.md`** at the repo root — mission, hardware, technical approach, MVP target.
2. **`journal/`** — session logs, chronological. Read the **most recent file** (`ls -t journal/` then read the top entry). It tells you what state the project was in at last session.
3. **`ml_backend/README.md`** — ML pipeline architecture, dataset choice (Gaddy 2020), current baseline result, stack decisions.
4. **`recorder/README.md`** and **`recorder/DATA_COLLECTION.md`** — the Flask recording harness and the data-collection strategy for hitting MVP vs AlterEgo tiers.
5. **`git log --oneline -20`** — recent commits tell you what has actually changed vs what the docs claim.
6. **`git status`** — anything in flight? uncommitted work?

Key things to internalize:

- **Single user**: the system is trained on Roman specifically, no generalization required for MVP.
- **Hardware track is often blocked**: the OpenBCI V3 clone board needs custom firmware from the eBay seller. If the latest journal entry still says "awaiting firmware," assume hardware is still down and ML track continues on the Gaddy public dataset as a proxy.
- **ML track uses raw PyTorch** (not Lightning), 20-word closed-vocab classifier for MVP, with a planned Model-1 (overt mouthing) → Model-2 (subvocal self-distillation) cascade.
- **Vocab files**: `ml_backend/vocab/mvp_20words.txt` is Roman's target; `ml_backend/vocab/gaddy_closed_top20.txt` is the pipeline-validation vocab matched to Gaddy's actual tokens.
- **Data lives outside git**: `ml_backend/data/` (Gaddy, ~5 GB) and `recorder/data/` (personal recordings) are gitignored.
- **Venv**: `ml_backend/.venv/` is the shared environment used by both `ml_backend` and `recorder`. PyTorch 2.11 + CUDA confirmed working.

Once you've read the above, summarize back to the user in ≤5 bullets:
- where the project is
- what's blocked
- what the next move options are

Do not start writing code until the user confirms the next move.
