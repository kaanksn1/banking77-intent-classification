# Scope and ownership

The repository owner is responsible for **Naive Bayes and GitHub/integration**.
When assisting this owner, limit implementation to that scope unless the user
explicitly assigns additional work.

- `src/banking77/naive_bayes.py` and `train_naive_bayes.py`: owner's model work.
- Data analysis, preprocessing/feature experiments: teammate 2.
- Logistic Regression implementation and experiments: teammate 3.
- Linear SVM implementation and experiments: teammate 4.
- Cross-model evaluation, comparison plots and presentation: teammate 5.

Shared setup, data downloads and agreed split/output contracts already exist.
Maintain the minimum shared infrastructure needed to run Naive Bayes, but do not
complete another teammate's implementation, experiments or reports on their behalf.
Describe their pending tasks accurately in documentation. Each teammate contributes
their own work through a separate branch and pull request.

Use validation for model/parameter selection. Keep the official test set reserved
until settings are fixed. Respect the duplicate/overlap reporting in docs/EXPERIMENTS.md.

Verification: `python -m unittest discover -s tests -v` and a Naive Bayes validation run.

## Explicitly assigned extension (2026-10-10)

The user assigned the newly required lecture baselines after the professor
clarified that RNN/CNN/LSTM and Transformer methods must also be included.
The owner may implement `neural_models.py`, `train_neural.py`,
`train_embeddings.py`, embedding preparation, GPU setup/preflight, and the
owner's NB + CNN voting experiment in `train_ensemble.py`, with their tests
and documentation. Existing teammate data/LR/SVM/evaluation implementations
and personal contribution reports remain owned by those teammates.
Use the unchanged BANKING77 splits. CLINC and JEV additions remain canceled.
Commit frozen new-model protocols before evaluating their official test.
