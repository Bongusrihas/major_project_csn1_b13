# 🧬 Causal Representation Learning for Autoimmune Disease Progression

### Major project: improving longitudinal prediction with temporal residual boosting

Based on **“Causal Representation Learning for Predicting Autoimmune Disease Progression From Longitudinal Multimodal Clinical Data”** — Harshit Sharma & Simran Kaur, *IEEE Access*, Vol. 13 (2025).
[Read the paper · DOI 10.1109/ACCESS.2025.3598276](https://doi.org/10.1109/ACCESS.2025.3598276)

The project predicts **next-visit disease flare** and **disease severity**, compares the paper implementation with **one improvement**, and presents the results in Streamlit.

**New user?** Follow [Download, install and run](#-download-install-and-run) first, then [Use the application](#-use-the-application-step-by-step). Only **Paper framework** and **Proposed framework** appear in the active comparison.

## 📚 Paper, architecture and improvement documentation

| Guide | What it explains |
|---|---|
| [Methodology](docs/METHODOLOGY.md) | What the paper studies, its architecture and training, what TRB adds, correction equations and paired evaluation. |
| [Architecture](docs/CAUSAL_MODEL.md) | Active blocks, tensor dimensions, code locations and differences from the publication. |
| [Measured results](docs/MEASURED_RESULTS.md) | Every current score, absolute gain, error reduction, unfavorable change and confidence interval. Generated from verified runs. |
| [Architecture sheet](docs/architecture_sheet.html) | Visual summary of the two frameworks; also available as [PDF](docs/architecture_sheet.pdf) and [PNG](docs/architecture_sheet.png). |
| [Implementation audit](docs/PROJECT_AUDIT.md) | Leakage controls, evidence limits and remaining validation work. |

The detailed result document and [JSON snapshot](docs/results/current_comparison.json) update with completed preparation, together with the result blocks below and in the methodology. To refresh these documents from existing artifacts without training or opening the app, run `uv run python -m src.project.reporting`.

The architecture sheet's editable source is [`scripts/build_architecture_sheet.py`](scripts/build_architecture_sheet.py). Its optional authoring command is `uv run --with reportlab python scripts/build_architecture_sheet.py`, which rebuilds HTML and PDF. Render the PDF preview with Poppler: `pdftoppm -scale-to 2200 -singlefile -png docs/architecture_sheet.pdf docs/architecture_sheet`. ReportLab and Poppler are documentation-authoring tools; they are not needed for a normal app launch.

---

## ⚖️ The two methods

| Method | What it does |
|---|---|
| **1. Paper framework** | Structural R-GCN → multi-resolution temporal fusion → temporal causal transformer → flare and severity heads. |
| **2. Proposed framework** | Keeps the paper network and adds a missingness-aware temporal correction, with short- and long-term history, irregular-visit trends and validation-only tuning. |

```text
Longitudinal patient visits
           │
           ▼
  Paper graph + temporal network ───────────► Paper prediction
           │
           ▼
  Temporal embedding + observed history
           │
           ▼
  Residual boosting correction ────────────► Improved prediction
```

The proposed framework uses temporal residual boosting and replaces LECG in the current workflow. It learns a correction to the paper prediction rather than relying on edge gates that showed little benefit in our earlier tests. The correction uses only current and earlier visits. Validation patients determine its stopping point, correction strength and one fixed configuration. The internal tuning choices are configurations of the proposed framework, not additional frameworks in the comparison.

---

## 🚀 Download, install and run

Follow these steps in order. The recommended setup uses **uv** to manage Python and install dependencies. You do not need to activate a virtual environment manually.

### Before you start

- Use **PowerShell** on Windows, or **Terminal** on macOS / Linux.
- Keep an internet connection for the first dependency installation and public-dataset download.
- Install Git if you choose the clone method below. The ZIP method does not require Git.
- Use **Python 3.12** for the documented setup; uv can install it for you. The repository's `.python-version` selects 3.12.
- A GPU is optional. Training works on CPU and can take longer; download and training time depend on your connection and hardware.

### Step 1 — Download the project from GitHub

Choose **one** of these methods.

**Option A: clone with Git**

1. Open this project's repository on GitHub.
2. Click the green **Code** button, select **HTTPS**, and copy the clone URL.
3. Open PowerShell / Terminal in the folder where you want the project.
4. Replace the placeholder URL below with the URL you copied, then run:

```shell
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git longitudinal_health_predictor
cd longitudinal_health_predictor
```

`YOUR_USERNAME/YOUR_REPOSITORY` is a placeholder, not a working download address. The final argument to `git clone` gives your local project folder the name `longitudinal_health_predictor`, regardless of the repository's name.

**Option B: download a ZIP without Git**

1. Open the project's GitHub repository.
2. Select **Code → Download ZIP**.
3. Extract the ZIP completely; do not run the project from inside the archive.
4. Open the extracted project folder in PowerShell / Terminal. If extraction creates nested folders, open the inner folder containing **`run.py`, `pyproject.toml` and `uv.lock`**.

For example, on Windows, replace this example path with your extracted folder:

```powershell
cd "C:\Users\YOUR_NAME\Downloads\longitudinal_health_predictor"
```

All commands below must run from that project folder.

### Step 2 — Install uv

If uv is already installed, skip the installer and check `uv --version`.

**Windows — PowerShell**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux — Terminal**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen your terminal after installation so it picks up uv on your PATH. Return to the project folder with `cd`, then verify:

```shell
uv --version
```

These commands use Astral's official installer. Alternative installation methods are in the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Step 3 — Install Python 3.12 with uv

```shell
uv python install 3.12
```

uv uses the project's `.python-version` when preparing its environment. You can use this setup even if the system `python` command is unavailable.

### Step 4 — Install the project dependencies

```shell
uv sync
```

This creates **`.venv`** and installs the dependencies declared in **`pyproject.toml`**, using the versions recorded in **`uv.lock`**. It includes PyTorch, Streamlit, XGBoost and the plotting libraries. The first installation can download large packages; allow it to finish before starting the app.

To require the checked-in lockfile without allowing uv to update it, use this variant instead:

```shell
uv sync --locked
```

Keep `pyproject.toml`, `uv.lock` and `.python-version` with the project when publishing or sharing it. See the [official uv locking and syncing guide](https://docs.astral.sh/uv/concepts/projects/sync/) for how dependency synchronization works.

#### GPU setup and verification

On **64-bit x86 Windows and Linux**, `uv sync` installs **PyTorch 2.6.0 with CUDA 12.4** from the official PyTorch CUDA index. The source is recorded in `pyproject.toml` and `uv.lock`, so later `uv run` commands retain the CUDA build. Other platforms use the default PyTorch package source.

To repair an older clone that installed a CPU-only build, update its project files from GitHub and run:

```shell
uv sync --locked
uv run python run.py --check-device
```

If training or Streamlit is running from that environment, stop it with **Ctrl+C** before replacing PyTorch, then restart it after synchronization. On a supported NVIDIA GPU with a compatible driver, the check prints the GPU name, `Training device: cuda` and `CUDA forward/backward check: passed`. This command checks the device without downloading datasets or starting training.

For the tested RTX 3050 setup, the output includes:

```text
PyTorch: 2.6.0+cu124
CUDA runtime in PyTorch: 12.4
Training device: cuda — NVIDIA GeForce RTX 3050 ...
CUDA forward/backward check: passed
```

The CUDA wheel includes its runtime libraries; this setup does not require a separate CUDA Toolkit installation. A compatible **NVIDIA driver** and GPU are still required. `nvidia-smi` reports driver support; `torch.version.cuda` reports the runtime bundled with PyTorch, so their version numbers can differ. See the [official PyTorch 2.6.0 CUDA wheel instructions](https://pytorch.org/get-started/previous-versions/#v260) and [uv's PyTorch source configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).

The neural network automatically selects CUDA when available and otherwise uses CPU. Data preparation, plots and the proposed framework's XGBoost residual head run on CPU, so GPU use is intermittent. Reading cached results does not start GPU training. The change of PyTorch build can invalidate older training caches; the next normal launch will rebuild affected work.

### Step 5 — Run the complete project

```shell
uv run python run.py
```

After setup, **this is the single command you normally need**. It automatically:

1. Checks the cached datasets and completed runs.
2. Downloads the public SLE data if missing and prepares the real and simulated cohorts.
3. Trains **Paper framework** and **Proposed framework** using the fixed defaults when valid cached models are unavailable.
4. Generates measured scores, prediction tables, charts and 3D projections.
5. Prepares three random-dataset comparisons when their cached results are unavailable.
6. Starts Streamlit and opens the dashboard in your browser.

Keep the terminal open while the server runs. The first run can spend substantial time preparing and training before the dashboard opens. Training progress appears in the terminal; no epochs, thresholds or model settings need to be entered in the app.

### Step 6 — Open the dashboard

Use the **URL printed in your terminal**. The launcher first tries:

```text
http://127.0.0.1:8501/
```

If that port is occupied, it chooses another available port through 8520, such as `http://127.0.0.1:8502/`. If the browser does not open automatically, copy the printed URL into your browser. These addresses work on the computer running the project.

For the dataset-testing page, choose **Dataset tests** in the sidebar. Its path is `/Random_dataset_tests` on the port printed by your launcher.

### Step 7 — Stop and reopen the app

To stop the server, press **Ctrl+C** in the terminal that is running it.

To open it again later, return to the same project folder and run:

```shell
uv run python run.py
```

**Valid cached datasets, models, predictions and figures are reused.** Changing an ordinary dashboard widget does not retrain the models. The **Random datasets → Generate and test both methods** action can train new random-cohort comparisons if its requested settings are not cached.

### Optional commands — use only when needed

| Task | Command |
|---|---|
| Normal launch / reuse completed work | `uv run python run.py` |
| Rebuild results and then open Streamlit | `uv run python run.py --update` |
| Prepare datasets, training and figures without opening Streamlit | `uv run python run.py --prepare-only` |
| Rebuild results without opening Streamlit | `uv run python run.py --update --prepare-only` |
| Explicitly repeat validation seed selection | `uv run python run.py --reselect-seed` |
| Run the tests | `uv run python -m pytest -q` |
| Show the selected device and test GPU computation | `uv run python run.py --check-device` |

`--update` rebuilds results with the same fixed defaults and frozen model seed; previous run folders are preserved. Changes to training code, dataset content or numerical-library versions can also invalidate the relevant training cache. Ordinary launches keep the frozen seed.

If Python and the project dependencies are already installed, the original `python run.py` launcher remains available. The uv workflow above is recommended for new installations.

---

## 🧭 Use the application step by step

### Compare the two frameworks

1. Open **Paper vs proposed** in the sidebar.
2. Choose the **Public SLE cohort** or **Simulated SLE / RA cohort**.
3. Read the two rows of measured scores and the charts generated from that completed run.
4. Open **Proposed framework** to see the method explanation, additional curves and patient predictions.
5. Open **3D explorer** to rotate / zoom the embeddings, select a patient and move through visit history.

The main comparison uses the two local implementations on identical held-out patients. Original published figures are available as a historical reference on the proposed-framework page; they are separate from the current local evaluation.

### Try the actual public dataset first

1. Open **Dataset tests**.
2. Leave **Dataset source → Actual public dataset** selected.
3. Review the held-out clinical visits. Optionally select **Download actual dataset with known outcomes** to inspect the CSV.
4. Click **Predict with Paper and Proposed**.
5. Compare the calculated scores and the chart showing which framework was correct.
6. Under **Which framework was right?**, choose a patient or view all patients.
7. Read the actual flare, both Yes/No predictions, predicted probabilities and correctness result.
8. Download the combined predictions and comparison scores using the buttons below the results.

The built-in example uses public GSE65391 test patients. The authors' original private paper cohort is not included. A patient's final visit has no known next-visit outcome and therefore remains unscored.

### Upload your own actual patient dataset

1. Open **Dataset tests → Upload CSV**.
2. Choose **Public SLE model · actual clinical inputs** for records that match the public SLE input schema. Choose the simulation format only for records matching that separate schema.
3. Expand **Input format, column details and CSV templates**.
4. Download the **blank input CSV** and **column guide**. You can also download a filled example containing actual public test-patient records.
5. Fill the CSV using one row per visit. Include patient identifiers, visit order, time gaps and your available clinical measurements; keep their units and category coding consistent with the guide.
6. If known, enter **Yes / No** in `actual_next_flare`. Leave unknown outcomes blank. Optional `actual_next_severity` accepts categories 0–4.
7. Upload the CSV using the upload control.
8. Select your actual flare and severity columns if their names were not detected automatically. Select **No outcome column** for prediction-only use.
9. Choose whether each outcome describes the **next visit** or the **current visit**. Current-visit mode shifts labels within each patient to check the preceding prediction.
10. Fix any validation errors displayed by the app, then click **Predict with Paper and Proposed**.
11. Review both frameworks' predictions, measured scores and per-visit correctness. Download the comparison if needed.

The page uses the existing models, saved training scaler and validation thresholds. Your supplied outcomes check predictions; they do not train a model or select a new threshold. Uploaded patient files and their results are held in the current session rather than saved by this page. Download results before closing the session if you want to keep them.

### Enter visits directly without creating a CSV

1. Open **Dataset tests → Enter patient data**.
2. Choose the matching model input format.
3. Select the clinical input columns you have measurements for.
4. Fill the editable table. Repeat the patient ID across that patient's visits, increase the visit number and enter the elapsed days.
5. Use the table's **+ row** control to add visits or another patient.
6. Select actual flare **Yes / No** when known; leave it blank otherwise.
7. Choose next-visit or current-visit outcome timing, then click **Predict with Paper and Proposed**.
8. Review and download the results as described above.

### Generate and compare random datasets

1. Open **Dataset tests → Random datasets**.
2. Keep the default dataset count and seed to reuse the prepared comparisons, or enter different values for a new experiment.
3. Click **Generate and test both methods**.
4. Allow any uncached training to finish.
5. Review the paired scores, choose a dataset and patient, then download the predictions.

### Understand the scores and correctness results

| Score | Better direction | Meaning |
|---|---|---|
| AUROC | Higher | Ability to rank flare visits above non-flare visits across thresholds. Requires both outcome classes. |
| AUPRC | Higher | Precision–recall performance; interpret it alongside how common flares are in the evaluated data. |
| F1 | Higher | Balance between precision and recall at the saved prediction threshold. |
| Accuracy | Higher | Fraction of scored visits whose Yes/No prediction matches the actual flare. |
| Precision / Recall | Higher | How often predicted flares are correct / how many actual flares are detected. |
| Severity MAE / RMSE | Lower | Error between predicted and actual next-visit severity categories. |
| Brier | Lower | Squared error of flare probabilities against actual Yes/No outcomes. |

| “Who was right?” | Meaning |
|---|---|
| **Paper only** | Only the Paper prediction matched the supplied actual flare. |
| **Proposed only** | Only the Proposed prediction matched the supplied actual flare. |
| **Both** | Both predictions matched the actual flare. |
| **Neither** | Both predictions were wrong for that visit. |
| **Not scored** | The actual outcome is unknown. |

Scores come from the selected run or supplied outcomes, not hardcoded values. A blank metric means it cannot be calculated from those labels. Accuracy alone can be misleading when most visits have no flare; compare multiple metrics on the same patients.

### Troubleshooting

| Problem | What to do |
|---|---|
| `git` is not recognized | Use **Code → Download ZIP**, or install Git and reopen the terminal before cloning. |
| `uv` is not recognized | Close and reopen the terminal after installing uv, then run `uv --version` again. |
| `pyproject.toml` or `run.py` cannot be found | Use `cd` to open the extracted / cloned project root. Both files must be in that folder. Quote paths containing spaces. |
| System `python` is unavailable | Run `uv python install 3.12`, then use `uv run python run.py`. |
| Dependency download fails | Check the connection and available disk space, then retry `uv sync`. |
| Setup changes cause an out-of-date lockfile error | For an unchanged checkout, obtain its matching `pyproject.toml` and `uv.lock`. If you intentionally edited dependencies, run `uv sync` to update the lock. |
| The browser does not open, or 8501 shows another app | Open the exact URL printed by this project's launcher; it may use another port such as 8502. |
| The first launch is still training | Keep the terminal open and check its progress. First preparation can be slow on CPU; subsequent valid runs use the cache. |
| An NVIDIA GPU is present but training uses CPU | Run `uv run python run.py --check-device`. If PyTorch has no CUDA runtime, update the checkout and run `uv sync --locked` after stopping its running Python processes. Check that the source and lockfile contain the CUDA wheel. |
| CUDA is installed but the GPU check fails | Check `nvidia-smi` and the NVIDIA driver. The default CUDA 12.4 build must support your GPU; errors from the computation check should be resolved before full training. |
| GPU utilization falls to zero during a run | Data preparation, XGBoost correction, charts and cached-result viewing use CPU. GPU utilization should rise during the neural training batches, rather than stay high throughout the whole workflow. |
| A cached run is incomplete or fails its integrity check | Run `uv run python run.py` to let preparation repair missing work. Use `--update` when you intentionally want a rebuild. |
| CSV measurements or labels are rejected | Use the selected model's template. Supply numeric clinical values, unique patient / visit pairs and valid Yes/No outcomes; leave unknown values blank. |
| AUROC or other scores are blank | Supply actual outcomes. AUROC requires both Yes and No labels; recognized training / validation patients are excluded from summary scores by default. |

---

## 🖥️ Streamlit sections

### 1. Paper vs proposed

All metrics, bars and curves read the latest completed run. The page automatically refreshes after a new run is published; it also shows the run ID, completion time and frozen model seed.

- Exactly two comparison rows and chart bars: **Paper framework** and **Proposed framework**, evaluated on the same patients.
- The original publication is available as an optional historical reference on the second page; its fixed scores are excluded from the main run comparison.
- The LSTM, Transformer, VRNN and clinical-baseline tables are excluded from the active comparison.

### 2. Proposed framework

- A short explanation of TRB and the change to the paper pipeline.
- Both methods’ measured scores on the same held-out patients.
- ROC, precision–recall, calibration, confusion matrices, training curves and severity errors.
- Patient timelines and actual outcomes beside both predictions.
- Historical-source expander with original paper figures, source-paper download and attribution; these are separate from measured local curves.

### 3. 3D embedding explorer

- Google Embedding Projector-style **3D t-SNE / PCA** views of real model embeddings.
- Rotate, zoom, search a patient and compare prediction colors.
- Reveal successive visits and animate a highlighted patient trajectory.

### 4. Dataset tests

- **Actual public dataset:** load the held-out GSE65391 patient visits, download the actual inputs with known outcomes, and run both saved frameworks.
- **Upload CSV:** supply your own actual patient visits. Download a blank template, a column guide or a filled public-data example from the page.
- **Enter patient data:** type measurements and optional actual flare Yes/No outcomes directly into the editable table.
- Each labeled visit shows both predictions, correctness and **Paper only / Proposed only / Both / Neither**. Unknown outcomes stay unscored.
- Summary scores and graphs are computed from the supplied outcomes. Download both predictions and scores; uploaded patient files are not saved by this page.
- Both saved models, training-fitted scalers and validation thresholds are reused. Uploading data never starts training or changes a threshold.

The **Random datasets** option also supports:

- Generate a few new longitudinal datasets with a chosen random seed.
- Train and test **both methods on identical patient splits**.
- See side-by-side scores and each method’s predicted risk, flare label and severity against the actual next-visit outcome.
- Download the paired predictions as CSV; repeated settings use the cache.

#### Entering your own actual dataset

Open **Dataset tests → Upload CSV** or **Enter patient data**. Choose the model matching your input format; use the public SLE model for actual clinical records with the public cohort's measurement units and category coding.

Required columns are `patient_id`, `visit`, `gap_days`, plus clinical measurements from the selected model's template. Use one row per visit, repeat the patient ID across visits, and include all available earlier visits. The first visit has a gap of 0 days. Missing clinical values stay blank; every row needs at least one measured input. `subject` / `visit_no` aliases and `cumulative_time` in days are also accepted.

Add **`actual_next_flare`** with **Yes/No** (also accepts 1/0 or True/False) when you know the actual following-visit result. Leave it blank if unknown. Optional **`actual_next_severity`** accepts categories 0–4. Current `sledai` is a measurement, not the future severity outcome.

By default, each row's outcome describes the **following visit**. If your outcome column describes the current visit, select **Current visit**; the app shifts it within each patient to check the preceding prediction. The final visit then stays unscored. You can map differently named outcome columns after uploading.

Files without outcomes still produce predictions. AUROC requires both flare classes; unavailable metrics remain blank. Training and validation patients recognized by their original IDs are excluded from summary scores by default, while their row-level predictions remain visible. Renaming previously seen patient IDs does not create an independent test set.

CSV uploads are limited to 10 MB / 10,000 visits and 128 visits per patient. The schema guide lists the exact expected clinical columns and actual source examples; do not replace medication category codes with doses or mix laboratory units.

---

## 📊 Published paper statistics

These values are transcribed from the supplied published PDF. They describe the authors’ clinical cohort, not our rerun.

| Published paper framework | Score |
|---|---:|
| AUROC | **0.867** |
| AUPRC | **0.726** |
| F1 | **0.702** |
| Severity MAE | **0.74** |
| Severity RMSE | **0.95** |
| Calibration ECE | **0.032** |

Original graphics and tables are cropped from `docs/paper/reference.pdf` and attributed to Sharma & Kaur (2025), **CC BY 4.0**. They are not recreated from invented patient predictions.

## 📈 Our measured results

<!-- RUN_RESULTS:START -->

Generated from the latest completed run. Retraining updates this section automatically.

Frozen model seed: **73**. Patient split seed: **42**. Maximum neural epochs: **30**.

| Evaluation cohort | Paper framework AUROC | Proposed framework AUROC | Change |
|---|---:|---:|---:|
| Public pediatric SLE cohort, GSE65391 | 0.5203 | 0.7201 | +19.98 points |
| Simulated SLE / RA cohort | 0.5885 | 0.6544 | +6.59 points |

Public SLE holdout: AUPRC **0.1744 → 0.3637**, F1 **0.2817 → 0.3438**, severity MAE **0.826 → 0.586**.

Public pediatric SLE cohort, GSE65391: **24 held-out patients**; paired patient bootstrap 95% interval for AUROC change: **-0.55 to +41.25 points**.

Simulated SLE / RA cohort: **300 held-out patients**; paired patient bootstrap 95% interval for AUROC change: **+4.69 to +8.51 points**.

The seed was chosen using validation scores, not test scores. These are exploratory local comparisons; performance can vary across datasets and the original paper cohort is unavailable.

<!-- RUN_RESULTS:END -->

**The 4–5 point improvement is a research target, not a guarantee.** The gains above compare our two local methods. They do **not** establish improvement over the paper’s published 0.867 score, because the authors’ original clinical cohort is unavailable.

---

## 📦 Datasets

| Dataset | Source and use |
|---|---|
| **GSE65391** | Public NCBI GEO pediatric SLE cohort; 157 patients with repeated visits after filtering. Used for a real-data paired evaluation. |
| **Simulated SLE / RA cohort** | Generated locally: 2,000 simulated patients. The repository’s “54-dim” name includes 52 input variables and two outcomes; it is not the paper’s original patient data. |
| **Random disease cohorts** | Three generated cohorts by default, each with 120 patients and six visits. Used for controlled side-by-side stress tests. |

[NCBI GEO: GSE65391](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65391)

The paper implementation is local; the authors’ original code, checkpoint and clinical cohort are not bundled. Public-cohort inputs and research severity bins differ from the paper.

Train/validation/test patients never overlap. Scalers fit training data only, features use available history only, and the final visit is excluded when it has no next-visit target. Paper scores and local evaluation scores are labeled separately in the app.

## 🎲 One-time seed selection

The first preparation compares model seeds **7, 21, 42, 73 and 123** using the same training and validation patients. Selection uses proposed-framework **validation AUROC**, with AUPRC, Brier and severity MAE as tie breakers. Test scores never select the seed.

The selected seed is frozen in **`config/frozen_seed.json`** and reused for both frameworks, including normal runs and `--update`. Dataset generation and patient splitting keep seed **42**. Random cohorts remain reproducible with their recorded generation seeds.

To explicitly repeat seed selection after changing the study protocol:

```shell
uv run python run.py --reselect-seed
```

This is optional; it does not run during ordinary launches. The validation search is an exploratory tuning step, not evidence of improved performance on an independent clinical cohort.

## ⚙️ Fixed defaults

| Setting | Default |
|---|---:|
| Maximum neural training epochs | 30 |
| Batch size | 64 |
| Neural learning rate | 0.0001 |
| Early-stopping patience | 6 epochs |
| Random cohorts | 3 |
| Patients / visits per random cohort | 120 / 6 |
| Added relation-edge noise | 20% |
| Dataset / patient-split seed | 42 |
| Model seed | Loaded from `config/frozen_seed.json` after the one-time validation search |

Defaults live in `src/project/settings.py`; no training settings need to be entered in the app. CUDA is used when available, otherwise the models run on CPU.

## 💾 Caching and safe recovery

- Downloads use temporary files and gzip verification before the final filename is committed.
- Runs use content fingerprints and SHA-256 artifact checks.
- A run is ready only after its completion manifest is written.
- Missing figures regenerate from cached predictions without retraining the models.
- Missing or modified model artifacts trigger a new run; existing outputs are preserved.
- A preparation lock prevents concurrent runs from writing the same cache.
- Streamlit binds to `127.0.0.1` and disables usage telemetry.

Generated artifacts are saved under **`reports/project/`**: each completed run contains two model artifacts, measured scores, patient predictions, figures and projections. The current dashboard index is `reports/project/dashboard.json`.

## 🗂️ Project structure

```text
run.py                         One-command launcher
pyproject.toml                 Project metadata and uv dependency declarations
uv.lock                        Locked dependency versions for uv sync
.python-version                Python 3.12 selection
app/streamlit_app.py            Four-section Streamlit navigation
app/dashboard.py               Shared dashboard views
app/dataset_testing.py         Actual data, CSV upload and manual-entry views
app/pages/                     Paper, improvement, 3D and random tests
src/project/settings.py        Fixed defaults
src/project/data.py            Downloading and preprocessing
src/project/pair.py            Paired training and evaluation
src/project/improvement.py     Our single TRB improvement
src/project/paper.py           Published statistics and original figures
src/project/plots.py           Measured charts and 3D projections
src/project/pipeline.py        Cached preparation and launch
src/project/custom_data.py     Frozen-model inference and supplied-outcome checks
src/models/                    Paper graph / temporal network
src/causal/                     Training, datasets and inference
src/synthetic/                 Random cohort generation
docs/paper/                    Reference PDF, attribution and extracted figures
reports/project/               Cached runs and dashboard artifacts
tests/                         Correctness and cache-recovery tests
```

## ✅ Tests

```shell
uv run python -m pytest -q
```

The dataset-testing suite checks actual-data prediction round trips, future-information and outcome leakage, label alignment, input validation, frozen thresholds, unknown outcomes, patient exposure and the Streamlit source options.

*Academic research prototype. Treatment sensitivity and embedding visualizations do not establish clinical causal effects.*
