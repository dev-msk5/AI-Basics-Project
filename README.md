# AI-Basics-Project

Solution for the *MI alapjai* circle-fitting project.

## Project goal

Given a noisy point set, fit the smallest enclosing circle to it. The input
may contain measurement noise, shape errors, outliers, a complete circle, or
only an arc of a circle. The project compares methods introduced in the
lectures:

- genetic algorithm;
- local search (hill climbing with random restarts);
- nature-inspired search (simulated annealing).

The methods optimize the same three parameters, `(center_x, center_y, radius)`,
and use the same cost function. The final radius is tightened to the farthest
point, so the returned circle encloses every input point.

## Current implementation status

### Completed

- Circle model with distance calculation and smallest-valid-radius tightening.
- Batched NumPy cost evaluation for comparing populations/chains efficiently.
- Hill climbing with multiple random restarts and adaptive step-size reduction.
- Simulated annealing with multiple chains, probabilistic uphill moves, and
  geometric cooling.
- Genetic algorithm with tournament selection, arithmetic crossover, Gaussian
  mutation, and elitism.
- Reproducible runs through a fixed random seed.
- Point generators for:
  - complete circles and circle arcs;
  - ellipses and ellipse arcs;
  - regular polygons;
  - uniformly random points;
  - Gaussian noise and randomly displaced outliers.
- Plotting helpers for fitted circles, overlaid solutions, and cost histories.
- Runtime measurement for each method in the example runner.

### Partially completed / still needed for the assignment

- The example runner currently executes one selected data set at a time. The
  available data-set lines can be switched in `src/main.py`, but they are not
  yet organized into an automated experiment matrix.
- The project does not yet produce a saved table or report comparing accuracy
  and runtime over repeated runs.
- The effect of increasing iteration count on precision and computation time
  still needs a systematic benchmark and visualization.
- Results for known shapes (regular circle, ellipse, polygon, arcs, and
  outlier-contaminated data) still need to be collected and evaluated against
  known reference values.
- A final assessment of efficiency, accuracy, stability, and limitations is
  therefore still outstanding.

In short, the optimization core and the tooling needed for the investigation
are implemented; the reproducible experiment campaign and its conclusions are
the remaining work.

## Running the current example

Install the project dependencies and run:

```powershell
pip install -e .
python src/main.py
```

The runner uses seed `42`, prints the fitted center, radius, cost, and elapsed
time for all three methods, then displays a cost-history comparison and an
overlay of the fitted circles. Edit the point-generation section in
`src/main.py` to select a circle arc, ellipse, polygon, outlier data set, or
random point set.

## Repository structure

```text
src/
  algorithms/   Hill climbing, simulated annealing, and genetic algorithm
  generator/    Synthetic point-set generators and noise/outlier helpers
  model/        Circle representation and batched cost function
  plot/         Matplotlib plotting helpers
  main.py       Example runner and timing/plotting entry point
```
