## 🧩 **`CONTRIBUTING.md` for Data Preprocessing Templates**


# 🤝 Contributing to Data Preprocessing Templates

Thank you for your interest in contributing to this project!
We welcome all contributions — from small documentation fixes to new preprocessing techniques.

To maintain a clean and consistent Git history, we follow the **Conventional Commits** standard.
Please read the following guidelines before making a pull request.

---

## 🧾 Commit Message Format

Each commit message consists of a **header**, an optional **body**, and an optional **footer**:

```

<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

```

### Example
```

feat(tabular): add KNN imputer for missing value handling

* Implement KNNImputer function in tabular_preprocessing.ipynb
* Add example usage and explanation in markdown
* Update README with new preprocessing technique

Closes #23

```

---

## 🧱 Commit Message Components

### **1. type**
Specifies the kind of change you’re making.

| Type | Description | Example |
|------|--------------|----------|
| **feat** | New feature (e.g., new preprocessing technique) | `feat(text): add TF-IDF vectorization` |
| **fix** | Bug fix | `fix(tabular): correct missing value threshold logic` |
| **docs** | Documentation changes only | `docs(readme): update feature selection explanation` |
| **style** | Code style or formatting (no logic changes) | `style(image): reformat image resize section` |
| **refactor** | Code restructuring without new features or fixes | `refactor(tabular): modularize outlier detection` |
| **perf** | Performance optimization | `perf(tabular): optimize correlation heatmap generation` |
| **test** | Add or modify tests | `test(tabular): add test for imputation functions` |
| **chore** | Maintenance tasks (dependencies, configs, etc.) | `chore: add requirements.txt for sklearn and pandas` |
| **ci** | CI/CD or automation changes | `ci(github): add workflow for linting notebooks` |
| **build** | Build system or packaging changes | `build: add Dockerfile for reproducible notebooks` |
| **revert** | Revert a previous commit | `revert: undo feat(image) due to conflict` |

---

### **2. scope**
The affected part of the project. Examples:
- `tabular`
- `text`
- `image`
- `docs`
- `utils`
- `notebook`

Example:
```

feat(tabular): add frequency encoding method

```

---

### **3. description**
A short, imperative summary (max 50 characters).
Use the **present tense** and **imperative mood** (e.g., *add*, *fix*, *update*).

✅ Good:
> `feat(image): add histogram equalization function`

❌ Bad:
> `added histogram equalization function`

---

## 🧩 Body (Optional)
Used for detailed explanations — what and why (max 72 characters per line).
Include context for complex changes, new dependencies, or impacts.

Example:
```

* Introduce OrdinalEncoder for ordered categorical columns
* Update encoding section in tabular_preprocessing.ipynb
* Add small dataset example for demonstration

```

---

## 📜 Footer (Optional)
For referencing issues, PRs, or noting breaking changes.

Examples:
```

Closes #45
BREAKING CHANGE: ColumnTransformer replaced with Pipeline

```

Use `!` in the header to indicate a breaking change:
```

feat(tabular)!: rename main preprocessing function

````

---

## 🧠 Development Guidelines

### 1️⃣ Code Style
- Use **PEP8** conventions.
- Keep code cells clean — avoid redundant prints or debug code.
- Use **clear English comments** for educational readability.

### 2️⃣ Notebook Structure
- Keep a consistent layout:
  - Title and Description
  - Imports
  - Data Loading
  - EDA
  - Preprocessing Sections
  - Model Training
- Each step should include both **Markdown explanations** and **executable code cells**.

### 3️⃣ Documentation
- Update `README.md` when adding or modifying a preprocessing technique.
- Add clear comments inside notebooks to explain the reasoning behind code choices.
- Use headings (`##`, `###`) for new sections.

### 4️⃣ Data
- Do **not** upload private or copyrighted datasets.
- Use public datasets (e.g., from Kaggle, UCI, or sklearn).
- Keep any sample CSV files small (<5MB).

### 5️⃣ Testing
If you add a new preprocessing function, please:
- Include a minimal example in the notebook.
- Ensure no runtime errors on sample data.

---

## 💬 Pull Request Process

1. Fork this repository
2. Create a new branch:
```bash
   git checkout -b feat/add-new-technique
```

3. Commit using the **Conventional Commits** format.
4. Push your branch and open a pull request.
5. Include:

   * A brief description of the changes
   * Screenshots (if visualization-related)
   * Issue references (if applicable)

---


### 💡 Example Commit History

```
feat(tabular): add frequency encoding
fix(tabular): correct correlation heatmap labels
docs(readme): add feature selection section
refactor(image): simplify resize function
chore: update dependencies in requirements.txt
```

---

## ❤️ Thank You

Every contribution helps make this project a better learning resource for the community.
We truly appreciate your time and effort in improving it!
