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
