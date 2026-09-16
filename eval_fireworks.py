"""
Evaluate a Fireworks-fine-tuned Togethr ticket router against the SAME held-out
validation split used in the Colab notebook (val_split_reference.csv), so the
result is directly comparable to the LLaMA Board / merged-model numbers.

Usage:
    export FIREWORKS_API_KEY=...
    python eval_fireworks.py --model accounts/<your-account>/models/<your-ft-model-id>

Requires: pip install fireworks-ai pandas scikit-learn
"""
import argparse
import os
import time

import pandas as pd
from fireworks.client import Fireworks
from sklearn.metrics import classification_report, confusion_matrix

LABEL_TOKENS = [
    "Client Safety / Escalation",
    "Compensation & Payroll",
    "Scheduling & Caseload",
    "Platform / Technical",
    "Credentialing & Compliance",
    "Training & Resources",
    "HR / Interpersonal",
    "General / Other",
]

SYSTEM_PROMPT = (
    "You are Togethr's internal ticket routing assistant. Togethr employs coaches, counselors, and therapists "
    "who submit complaints or issues about their work. Given a ticket, respond with exactly one of the following "
    "categories: Client Safety / Escalation, Compensation & Payroll, Scheduling & Caseload, Platform / Technical, "
    "Credentialing & Compliance, Training & Resources, HR / Interpersonal, General / Other."
)


def classify(client: Fireworks, model: str, ticket_text: str, retries: int = 3) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ticket: {ticket_text}"},
    ]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=12,
                temperature=0,
            )
            generated = resp.choices[0].message.content.strip()
            matched = next(
                (l for l in LABEL_TOKENS if generated.lower().startswith(l.lower())),
                None,
            )
            return matched or "General / Other"
        except Exception as e:  # noqa: BLE001 - keep the run alive across transient API errors
            if attempt == retries - 1:
                print(f"  ! failed after {retries} attempts: {e}")
                return "General / Other"
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Fireworks model id, e.g. accounts/me/models/togethr-router-v1")
    parser.add_argument("--val-csv", default="val_split_reference.csv")
    parser.add_argument("--out-csv", default="fireworks_predictions.csv")
    args = parser.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("Set FIREWORKS_API_KEY in your environment first.")

    client = Fireworks(api_key=api_key)
    df_val = pd.read_csv(args.val_csv)

    y_true, y_pred = [], []
    for i, row in df_val.iterrows():
        pred = classify(client, args.model, row["text"])
        y_true.append(row["label"])
        y_pred.append(pred)
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(df_val)}")

    df_val["fireworks_pred"] = y_pred
    df_val.to_csv(args.out_csv, index=False)

    print("\n=== Fireworks SFT model — validation results ===")
    print(classification_report(y_true, y_pred, labels=LABEL_TOKENS, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_TOKENS)
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=LABEL_TOKENS, columns=LABEL_TOKENS))

    print(f"\nSaved predictions → {args.out_csv}")


if __name__ == "__main__":
    main()
