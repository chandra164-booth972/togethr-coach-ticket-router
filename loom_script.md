# Loom Walkthrough Script — Togethr Coach Ticket Router (Week 5)

**Target length: 4–5 minutes.** Suggested screen shown in parentheses.

---

## 0:00–0:30 — Intro (talking head or title slide)

"Hi, I'm Abhi. For this project I forked the Gen Academy Week 5 fine-tuning notebook and retargeted it at a real problem for Togethr, my relationship-navigation startup. Togethr employs a growing roster of coaches, counselors, and therapists, and right now every internal complaint or issue they submit — payroll, scheduling, a platform bug, a safety concern — lands in one shared inbox with no automatic routing. I fine-tuned a small Qwen3 model to route these tickets into 8 internal queues, and I'll also walk through a real, documented dead end I hit trying to do the same fine-tune on Fireworks AI."

## 0:30–1:15 — The task and the taxonomy (screen: README Section 1)

"The idea: a lightweight classifier reads a practitioner's ticket and predicts one of 8 queues — things like Client Safety and Escalation, Compensation and Payroll, Scheduling and Caseload, Platform bugs, Credentialing, Training gaps, HR conflicts, and a general catch-all. The one that matters most is Client Safety — that's disclosures, boundary violations, crisis language — and it's deliberately the rarest class in my dataset, because real safety escalations should be rare, but it's the one queue where a routing miss actually matters."

## 1:15–1:45 — Dataset (screen: togethr_coach_tickets.csv or the README's distribution table)

"I generated a 590-row synthetic dataset matching the original project's exact schema — a category and a ticket text column, with the same PII-placeholder convention. It's intentionally imbalanced: Scheduling is the most common ticket type at 130 examples, Client Safety is the rarest at 35. Split 80/20, stratified, same random seed throughout so every downstream comparison uses the identical held-out set."

## 1:45–2:30 — Training (screen: the loss curve / the notebook's LLaMA Board cell)

"I forked the original notebook cell-for-cell — same LLaMA Factory pipeline, same LLaMA Board UI, just swapped in my label set and system prompts. Trained a LoRA adapter on Qwen3-1.7B-Base on a free Colab T4 GPU. Here's the loss curve — starting loss around 5.3, final loss down to 0.017, a clean convergence with no spikes."

## 2:30–3:15 — Results (screen: the baseline-vs-fine-tuned bar chart)

"This is the core result. The untrained base model scored 39% accuracy on my held-out validation tickets — and it completely whiffed on two categories, Scheduling and Platform bugs, zero F1 on both, because it has no idea these categories exist as a fixed taxonomy going in. After fine-tuning: 94.1% accuracy, and Client Safety and Escalation — the category that matters most — hit perfect precision and recall. Every safety-critical ticket in my validation set got routed correctly, and nothing else got mistaken for one. The confusion matrix shows the only real confusion is between Credentialing, Training, and Payroll — which makes sense, they share a lot of documentation-and-forms vocabulary."

## 3:15–4:15 — The Fireworks dead end (screen: terminal screenshots / README Section 5)

"Per my guidance for this project, I also tried to run the same fine-tune on Fireworks AI, to get a second training platform to compare against. This is the part I actually think is the most interesting finding of the whole project. Fireworks' Model Library shows Qwen3 1.7B as tunable, and a dry-run of my training job validated cleanly with no errors — but the real submission was flat-out rejected: 'model is not supported for fine-tuning.' So I tried Fireworks' own documented example model instead, Qwen3 4B — and that one got blocked by a completely different wall: it said B200 and B300 training requires a Tier 2 account, and offered to unlock it for fifty dollars in credits. A 4-billion-parameter model has no real technical reason to need Fireworks' newest flagship GPUs. Rather than pay to unlock that for a course project, I documented it as the finding: Fireworks' advertised tunable-model list and its dry-run validation don't reliably predict what the actual training backend will accept, and my free-tier account was blocked from managed fine-tuning entirely, independent of which small model I picked."

## 4:15–4:45 — Wrap-up (talking head)

"Everything's in the GitHub repo — the dataset, the forked notebook, the Fireworks scripts and error logs, and the full write-up. The core deliverable, the LLaMA Factory fine-tune, is a clean success: 39% to 94% accuracy, perfect on the highest-stakes category. And the Fireworks side turned into a real platform-limitations finding rather than a working second data point, which I think is still a useful thing to know if Togethr ever wants to fine-tune models for production. Thanks for watching."

---

### Notes for recording
- Have the bar chart and confusion matrix images open and ready before recording so you're not scrolling live.
- If asked how to find the two Fireworks failures, the exact error messages and commands are in README.md Section 5 — no need to re-derive them live.
- Consider showing the terminal screenshots of the two `firectl` failures directly if you want to prove they're real rather than paraphrased — adds credibility to the "documented finding" framing.
