# Graph Signal Recovery

Estimate the true readings of a sensor network from noisy observations and a weighted graph. This course project requires a mathematical model, an explanation of the solution method, and one prediction for every sensor.

**Recommended starting point:** weighted graph Laplacian smoothing, with its smoothing strength selected by leave-one-node-out validation. Compare it with neighbor averaging and a Huber-loss variant before choosing the final method.

The repository currently contains the assignment templates and public data. The recovery algorithm and parameter-selection workflow described below still need to be implemented.

## Repository layout

```text
.
├── README.md
├── en/
│   ├── README_for_Students.md
│   ├── Student_Graph_Signal_Assignment_MSSV_MODULO.ipynb
│   └── public_cases.json
└── vi/
    ├── README_for_Students.md
    ├── Student_Graph_Signal_Assignment_MSSV_MODULO.ipynb
    └── public_cases.json
```

The `en/` and `vi/` folders provide English and Vietnamese instructions. Their JSON datasets are identical. Keep the notebook and its JSON file in the same folder.

Read the official instructions in the chosen notebook, especially Sections I and II. This README supplements them with a proposed solution strategy.

## Problem and input

Each sensor is a node in an undirected, connected, weighted graph. Every node has a noisy observation; some observations may be outliers. The goal is to estimate the unknown true signal at all nodes.

| Quantity | Meaning |
|---|---|
| `G` | Graph containing nodes, edges, and edge weights |
| `positions` | Dictionary mapping node IDs to 2D sensor coordinates |
| `y` | Vector of noisy observations, ordered by node ID |
| `x_hat` | Estimated true signal to be computed, in the same order as `y` |
| `x_true` | Hidden ground truth used by the instructor for grading |

Each case in `public_cases.json` stores `student_id`, `nodes`, `positions`, `edges`, and `y`. Edges are triples `[u, v, weight]`.

The supplied dataset has 100 example cases, identified by `20260001` through `20260100`. Every current case has 32 nodes, numbered `0` through `31`, and between 73 and 86 edges. Implement the method for `n = len(y)` rather than hard-coding 32; the instructor may replace the example data.

The physical quantity, measurement units, noise distribution, and outlier locations are unspecified. Assumptions about them must be explained rather than treated as known facts.

## Obtain the assigned case

1. Open the notebook in the preferred language using Jupyter or a compatible notebook editor. Its existing code requires NumPy, NetworkX, and Matplotlib.
2. In Section III, replace `NHOM_MSSV` with all actual group members' student IDs. Store IDs as strings and preserve leading zeros.
3. The first entry is the representative: `MSSV_DAI_DIEN = NHOM_MSSV[0]`.
4. Run Section IV to load the data and Section V to visualize it.

MSSV means student ID. An ID matching an existing case selects that case directly. Other numerical IDs are mapped to a case using modulo; alphanumeric IDs use a stable hash. The assignment is deterministic. Follow the instructor's rule to select one representative without trying multiple IDs to find an easier case.

## Recommended model: graph Laplacian smoothing

Assume connected sensors tend to have similar true values, with larger edge weights indicating a stronger relationship. Allow the signal to vary across the network.

Estimate the signal by minimizing:

$$
\hat{x} = \operatorname*{arg\,min}_{x\in\mathbb{R}^n}
\left[
\frac{1}{2}\sum_{i=0}^{n-1}(x_i-y_i)^2
+ \frac{\lambda}{2}\sum_{\{i,j\}\in E}w_{ij}(x_i-x_j)^2
\right].
$$

Each undirected edge appears once in the edge sum.

- **Measurement agreement:** the first term discourages unnecessary changes to observed readings.
- **Graph smoothness:** the second discourages large differences between connected sensors, especially across strongly weighted edges.
- **Smoothing strength:** a positive `lambda` balances the terms. Too little smoothing preserves noise; too much can erase real variation.

Construct the symmetric weighted adjacency matrix `W`, the diagonal weighted degree matrix `D`, and the unnormalized Laplacian:

$$
D_{ii}=\sum_j W_{ij}, \qquad L=D-W.
$$

Because the edge penalty equals $x^\top Lx$, the solution satisfies:

$$
\boxed{(I+\lambda L)\hat{x}=y.}
$$

Compute the estimate by solving this linear system. Explicit matrix inversion is unnecessary. For the current data, the system has only 32 unknowns.

## Select parameters without the hidden answers

Use **leave-one-node-out cross-validation** on the assigned case:

1. Choose a small grid of positive smoothing strengths, initially `0.003, 0.01, 0.03, 0.1, 0.3, 1, 3`.
2. For each candidate, withhold one node's observation while retaining the full graph and all coordinates.
3. Fit using the remaining observations and predict the withheld value.
4. Repeat for every node and compute validation RMSE. Record MAE as an additional diagnostic.
5. Inspect the score curve and nearby parameter choices. Expand the grid if its minimum lies at an endpoint.
6. Select the method and parameters, then **refit using all observations** to produce the final `x_hat`.

The withheld observation must be excluded from fitting. For the quadratic model, define a diagonal mask `M` with zero at the withheld node and one at every other node, then solve:

$$
(M+\lambda L)x=My.
$$

Any preprocessing or noise-scale estimation used during validation must also exclude the withheld reading.

Validation compares predictions with noisy observations. Under independent, zero-mean measurement noise, it provides a useful proxy for recovery performance. Outliers, correlated noise, and the small sample size can distort that proxy. Selecting the minimum on the same validation data also makes the reported score optimistic. It is not the instructor's hidden-ground-truth RMSE.

## Robust alternative: Huber loss

To limit the influence of unusually large measurement errors, compare the quadratic model with:

$$
\hat{x} = \operatorname*{arg\,min}_x
\left[
\sum_i\rho_\delta(x_i-y_i)
+\frac{\lambda}{2}x^\top Lx
\right],
$$

where:

$$
\rho_\delta(r)=
\begin{cases}
\frac{1}{2}r^2,& |r|\leq\delta,\\
\delta|r|-\frac{1}{2}\delta^2,& |r|>\delta.
\end{cases}
$$

Huber loss is quadratic for small residuals and linear for large ones. The threshold `delta` controls when an observation's influence is reduced. Thresholds depend on measurement scale; choose a small candidate set informed by training residuals and evaluate it using the same validation procedure.

One solution method is iteratively reweighted least squares:

1. Initialize with ordinary graph smoothing.
2. Calculate residuals `r_i = x_i - y_i`.
3. Set `q_i = 1` for `abs(r_i) <= delta`, otherwise `q_i = delta / abs(r_i)`.
4. With `R = diag(q)`, solve `(R + lambda * L) x_new = R y`.
5. Repeat until relative changes are small, subject to a maximum iteration count. Record convergence diagnostics.

During cross-validation, set the withheld node's weight to zero throughout fitting. A local discrepancy is evidence to investigate, not proof of a faulty sensor. Prefer the simpler quadratic model if the robust alternative offers no convincing benefit.

## Exploratory findings on example case `20260001`

This example has 32 nodes and 80 edges. The following calculations informed the recommendation:

| Observation | Value |
|---|---:|
| Weighted mean squared difference across edges | 1.106 |
| Mean squared difference across all distinct node pairs | 4.087 |
| Observed reading at node 28 | 17.549 |
| Weighted average of node 28's four neighbors | 19.550 |

Connected observations agree more closely than arbitrary pairs, supporting the smoothness assumption. Node 28 is a candidate for closer inspection, but its true value is unknown.

An exploratory leave-one-node-out comparison gave:

| Method | Validation RMSE |
|---|---:|
| Linear plane fitted to the 2D coordinates | 0.842 |
| Quadratic surface fitted to the 2D coordinates | 0.868 |
| Weighted average of the withheld node's neighbors | 0.735 |
| Graph Laplacian smoothing, best tested `lambda = 0.1` | 0.731 |

Huber thresholds of `0.5`, `1.0`, and `1.5` tied the quadratic model's best score on the tested grid. Increasing the quadratic model's `lambda` to `3` worsened its validation RMSE to approximately `0.964`.

These are preliminary results from the initial investigation; a reproduction script is not currently included. The small improvement over neighbor averaging should not be treated as a demonstrated advantage on hidden truth. Reproduce the comparison on the group's assigned case, save the evaluation code, and do not assume `lambda = 0.1` transfers to another case.

## Implementation and report workflow

1. **Inspect the data:** verify dimensions, connectivity, finite readings, and positive edge weights. Visualize observations and local discrepancies.
2. **Implement baselines:** neighbor averaging, quadratic Laplacian smoothing, and the Huber variant.
3. **Evaluate:** record validation RMSE, MAE, parameter curves, and solver convergence where applicable. Avoid selecting among a large number of methods using only 32 readings.
4. **Refit and inspect:** use all observations for the final estimate. Plot observed and estimated signals on the same color scale, plus the correction `x_hat - y`.
5. **Explain:** document assumptions, objective terms, parameter selection, optimization, checks, and limitations in Section VI or a separate report.
6. **Export:** verify that `x_hat` has the same length and order as `y`, then use the notebook's existing `save_submission(x_hat)` helper.

Section VII is the intended location for solution code. Including reproducible code is recommended, although the assignment permits computing predictions elsewhere if the method is explained fully.

## Expected output and submission

The numerical output is `x_hat`, containing exactly one finite real-valued estimate per node. Entry `i` corresponds to node `i` and observation `y[i]`. Do not submit the unprocessed observation vector as the answer.

Submit both required files:

- `Student_Graph_Signal_Assignment.ipynb`: the completed original notebook, renamed from the supplied filename, containing all group members' IDs and the required work.
- `submission_<representative_mssv>.csv`: predictions with exactly these four columns:

```csv
representative_mssv,group_mssvs,node_id,x_hat
```

| Column | Required content |
|---|---|
| `representative_mssv` | Representative's ID, repeated identically on every row |
| `group_mssvs` | All group members' IDs joined with `|`, repeated on every row |
| `node_id` | Every integer from `0` to `n-1`, appearing exactly once |
| `x_hat` | Finite numerical estimate for that node |

Use UTF-8 encoding, commas between columns, and periods for decimals. The current dataset requires 32 data rows plus the header. Preserve student IDs exactly, including leading zeros.

A separate report is required if the notebook does not fully explain the method. A PDF cannot replace the notebook, and plots or report tables cannot replace the CSV.

The instructor evaluates recovery using:

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=0}^{n-1}
(\hat{x}_i-x_i^{\mathrm{true}})^2}.
$$

Lower is better. The repository contains no `x_true`, so the actual grading RMSE cannot be calculated locally.

## References

- [English assignment instructions](en/README_for_Students.md) and [notebook](en/Student_Graph_Signal_Assignment_MSSV_MODULO.ipynb).
- [Vietnamese assignment instructions](vi/README_for_Students.md) and [notebook](vi/Student_Graph_Signal_Assignment_MSSV_MODULO.ipynb).
- Shuman et al., [The Emerging Field of Signal Processing on Graphs](https://arxiv.org/abs/1211.0053), for graph signal smoothness and Laplacian methods.
- Zhou et al., [A New Perspective on Robust M-Estimation](https://doi.org/10.1214/17-AOS1606), for Huber loss and the robustness tradeoff.
