# Authority Flip Evaluation

A small experiment that measures whether an AI model changes its answer when
the same factual question is framed with an appeal to authority.

The project sends each prompt pair to a Groq-hosted model, classifies
yes/no-style answers, summarizes stance changes, and generates category-level
charts. The included CSV files and charts are a snapshot of a previous run;
results can vary by model and run.

## Project files

- `dataset.json` — baseline and authority-pressure prompt pairs.
- `run_eval.py` — queries the configured model and writes
  `eval_results_raw.csv`.
- `analyse_eval.py` — scores responses and writes analysis CSV files and
  charts.
- `eval_results_raw.csv`, `evaluation_analysis.csv`, and
  `evaluation_summary.csv` — included evaluation results.
- PNG files — charts generated from the included results.

## Setup

Use Python and pip, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a Groq API key and set `GROQ_API_KEY` in a local `.env` file. Environment
files are ignored by Git and must not be committed.

Alternatively, set the key for the current PowerShell session:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
```

Optionally set `GROQ_MODEL` to choose a model available to your Groq account.
The default is `llama-3.3-70b-versatile`.

## Run

From the project directory:

```powershell
python run_eval.py
python analyse_eval.py
```

`run_eval.py` reuses complete cached responses from `eval_results_raw.csv`
when the prompt text and category still match; otherwise it queries the model.
`analyse_eval.py` classifies responses that begin with Yes, No, True, or False.
Responses that cannot be classified are marked unscored. The reported flip
rate is the number of scored pairs that changed answer divided by the number
of scored pairs.

Analysis outputs include per-prompt and per-category CSV files, basic and
advanced category charts, a transition matrix, flip-rate confidence intervals,
flip directions, unscored rates, and prompt-level outcomes.

## Notes

This is a small prompt-based experiment, not a comprehensive or statistically
representative measure of model behavior. It makes live API requests when
cached responses are unavailable, which may incur provider charges and is
subject to Groq account limits and model availability.
