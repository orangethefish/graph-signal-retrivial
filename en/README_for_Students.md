# Graph Signal Recovery — Student Package

1. Keep `Student_Graph_Signal_Assignment_MSSV_MODULO.ipynb` and `public_cases.json` in the same folder.
2. Open the notebook and read **Section I and all instructions in Section II** before starting.
3. Enter **all group members' student IDs (MSSVs)** in Section III; the representative's ID is the first entry.
4. Run all notebook cells in order. Sections IV and V load and display the graph; Section VI explains the implemented model; Section VII compares methods and computes the predictions; Section VIII exports the CSV. Review the explanation and results before submitting.
5. You **must submit** the completed notebook and `submission_<representative_mssv>.csv`, with exactly the **four columns** specified in Section VIII: `representative_mssv,group_mssvs,node_id,x_hat`. The submission instructions request the notebook filename `Student_Graph_Signal_Assignment.ipynb`; rename your completed copy accordingly when submitting.
6. Include a separate explanation of your method if the notebook does not explain it fully. Without a valid CSV file, the submission cannot be graded automatically.

The student IDs `20260001` through `20260100` are currently **examples**; the instructor will replace the data when the actual class list is available. The graph assigned to a student ID is fixed, and the package does not contain the ground-truth signal `x_true`.

**Terminology:** MSSV is the Vietnamese abbreviation for student ID. The original code identifiers are retained: `NHOM_MSSV` is the list of all group members' IDs, and `MSSV_DAI_DIEN` is the representative's ID. Required CSV column names are also unchanged. `public_cases.json` is copied unchanged from the Vietnamese package.

## Implemented solution

The notebook contains the entire implementation, so it can be submitted without a separate solver module. It compares weighted neighbor averaging, quadratic graph Laplacian smoothing, and Huber-loss smoothing. Leave-one-node-out validation selects the smoothing strength, extending the search grid when needed. Every withheld reading is excluded from fitting and Huber threshold estimation. Huber must improve validation RMSE by more than 1% to replace the best quadratic or neighbor model. The chosen method is then fitted using all observations.

Section VI explains the assumptions, objectives, parameter selection, optimization, convergence checks, and limitations. Section VII displays scores and plots and records the supporting evidence. Section VIII validates the predictions and student IDs before exporting the four required columns.

## Run the notebook

From the repository root, the prepared local environment can execute the notebook without opening an editor:

```sh
.venv/bin/python en/run_assignment.py
```

The runner uses the IDs currently entered in Section III, executes every cell, saves the notebook with its outputs, and exports the prediction CSV. Alternatively, open the notebook in a compatible editor, select the Python environment with the dependencies installed, and run all cells.

For a fresh environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r en/requirements.txt
.venv/bin/python en/run_assignment.py
```

If creating the environment fails because `ensurepip` is unavailable, an existing system installation of pip can provision it instead:

```sh
python3 -m venv --without-pip .venv
python3 -m pip --python .venv install -r en/requirements.txt
.venv/bin/python en/run_assignment.py
```

The numerical regression tests run against the actual notebook code:

```sh
.venv/bin/python -m unittest discover -s en/tests -v
```

They check an analytical solution, node ordering, preservation of constant signals and the quadratic mean, resistance to a synthetic outlier, withheld-reading isolation, search expansion, the documented example, input rejection, and CSV formatting with leading-zero IDs.

## Example results and generated files

The saved notebook and `submission_20260001.csv` were generated using the **example IDs**, not your actual group IDs. Replace the list in Section III and rerun all cells before submitting. Do not try different representatives to select an easier case.

For example case `20260001`, the selected method is **quadratic Laplacian smoothing with `lambda = 0.1`**. Validation RMSE is approximately **0.730948**, compared with **0.735080** for neighbor averaging. The best tested Huber variant ties the quadratic result, so the simpler quadratic model is retained. These scores use withheld noisy observations; the instructor's hidden-ground-truth RMSE remains unknown.

The required prediction file is `submission_<representative ID>.csv`, with one row per node. Optional supporting files are stored separately in `results_<representative ID>/`:

- `validation_scores.csv`: every tested model and parameter setting, including convergence counts.
- `heldout_predictions.csv`: the selected model's validation predictions and residuals.
- `diagnostics.json`: chosen parameters, scores, convergence, dataset checksum, and software versions.
- `signal_recovery.png`: observations and estimates on a shared color scale, corrections, and readings by node ID.
- `validation_curves.png`: validation RMSE and MAE across smoothing strengths.

The completed notebook contains the method explanation; review and adapt it to your group's work. Rename the notebook to `Student_Graph_Signal_Assignment.ipynb` when preparing the required submission.
