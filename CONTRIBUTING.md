# Contributing to Text Miners

Thank you for contributing! This guide explains how team collaborators work with the repository.

## Prerequisites

- You have been added as a collaborator on `dominiclau2002/text-miners` (Settings → Collaborators).
- Git is installed on your machine.
- You have run the one-time notebook setup described in `README.md` (nbstripout + pre-commit).

## One-time clone

Clone the repository directly — no fork needed:

```bash
git clone https://github.com/dominiclau2002/text-miners.git
cd text-miners
```

## Daily workflow

### 1. Keep your local `main` up to date

```bash
git checkout main
git pull origin main
```

### 2. Create a feature branch

Use a short, descriptive branch name:

```bash
git checkout -b your-username/short-description
# example:
git checkout -b mirfanmn/task2-classification
```

### 3. Make your changes

Edit notebooks, scripts, or documentation as needed.

### 4. Stage and commit

```bash
git add <files>
git commit -m "Short description of what you changed"
```

Keep commit messages concise and in the present tense (e.g. `Add classification notebook scaffold`).

### 5. Push your branch

```bash
git push origin your-username/short-description
```

Because you are a collaborator you push directly to `dominiclau2002/text-miners` — no fork is required.

### 6. Open a Pull Request

1. Go to <https://github.com/dominiclau2002/text-miners>.
2. GitHub will show a **"Compare & pull request"** banner for your recently pushed branch — click it.
3. Fill in the PR title and description.
4. Request a review from a team member if needed.
5. Once approved, merge using **Squash and merge** or **Merge commit** as agreed by the team.

### 7. Clean up after merge

```bash
git checkout main
git pull origin main
git branch -d your-username/short-description
```

## Branch naming conventions

| Prefix | Use |
|---|---|
| `username/feature-name` | New feature or notebook |
| `username/fix-name` | Bug fix |
| `username/docs-name` | Documentation only |

## Notebook hygiene

This repository uses `nbstripout` to strip notebook outputs before commit. Run the one-time setup once per clone:

```bash
python3 -m pip install nbstripout pre-commit
python3 -m nbstripout --install
python3 -m pre_commit install
```

You can verify the hook is working at any time:

```bash
python3 -m pre_commit run --all-files
```

## Questions

Open a GitHub Issue or contact the repository owner (@dominiclau2002).
