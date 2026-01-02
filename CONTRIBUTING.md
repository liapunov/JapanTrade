# Contributing to JapanTrade

Thank you for considering a contribution to JapanTrade. This project provides tools for acquiring, normalizing, and exploring Japanese customs trade data. Contributions that improve data acquisition, normalization, analysis utilities, documentation, and the Streamlit explorer are all welcome.

## Ways to contribute

- **Bug reports**: File an issue with clear steps to reproduce, expected behavior, and environment details (OS, Python version, package versions, and sample data if possible).
- **Feature requests**: Open an issue describing the problem to solve, the proposed approach, and any alternatives considered.
- **Pull requests**: Improve code, tests, docs, or tooling by opening a pull request linked to an issue when applicable.

## Development setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. For notebook work, install Jupyter (`pip install jupyter`).
3. For the Streamlit app, run:
   ```bash
   streamlit run src/japantrade/app.py
   ```

## Pull request expectations

- Keep changes scoped and focused on the repository's purpose: downloading customs data, normalizing it for analysis, and supporting exploration/reporting tools.
- Add or update tests when changing behavior. Place tests under `tests/` and prefer deterministic fixtures.
- Follow existing code style and avoid introducing unnecessary dependencies.
- Update documentation or examples when user-visible behavior changes.
- Describe the change, rationale, and testing performed in the pull request template.

## Coding guidelines

- Prefer clear, readable Python with type hints where they add clarity.
- Avoid broad exception handling; catch specific exceptions when needed.
- Keep functions and modules cohesive: downloading logic lives in `customsgrabber.py`, normalization in `tradefile.py`, analysis in `tradeanalysis.py`, and app concerns in `app.py`.
- Do not place `try/except` blocks around imports.

## Commit and branch hygiene

- Write clear commit messages summarizing the change.
- Rebase or merge main as needed to keep branches up to date before submitting.

## Reporting security issues

Please do not open public issues for security concerns. Instead, email **security@japantrade.dev** with details and reproduction steps. See `SECURITY.md` for more.
