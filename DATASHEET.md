Datasheet for BBO Capstone Query History

## Motivation
* **Why was this dataset created?**
    This dataset was created to document the iterative query history and function evaluations for the "Black-Box Optimization (BBO) Capstone Challenge." Its primary purpose is to serve as a training and validation set for surrogate models (specifically Gaussian Processes and Deep Ensembles) attempting to predict the global optima of 8 unknown functions.
* **What task does it support?**
    It supports the task of Bayesian Optimization (BO). Specifically, it allows researchers to analyze the efficiency of different acquisition functions (like UCB) and surrogate architectures (like GPs vs. NNs) in high-dimensional (2D to 8D) search spaces with limited data budgets.

## Composition
* **What does it contain?**
    The dataset contains 18 samples for each of 8 separate black-box functions (Function 1 through Function 8). Each sample consists of an input vector $X$ (coordinates in the range [0, 1]) and a scalar output $Y$ (the function value).
* **What is the size and format?**
    * **Size:** 144 total data points (8 functions × 18 queries each).
    * **Format:** The data is stored as NumPy `.npy` arrays.
    * **Dimensionality:** Inputs range from 2D (Function 1) to 8D (Function 8).
* **Are there any gaps?**
    Yes. Due to the "black-box" nature and the limited query budget (only 18 points per function), the dataset is extremely sparse. There are significant gaps in coverage, particularly near the boundaries (corners) of the 8D hypercube, leading to potential "corner bias" in trained models.

## Collection Process
* **How were the queries generated?**
    1.  **Initial Phase (Points 1-5):** Generated using Random Search and Latin Hypercube Sampling (LHS) to establish a baseline coverage of the search space.
    2.  **Intermediate Phase (Points 6-12):** Generated using a Gaussian Process (GP) surrogate model with a UCB acquisition function.
    3.  **Advanced Phase (Points 13-18):** Generated using a **Deep Ensemble** surrogate model (10 neural networks with bagging) to better handle high-dimensional feature interactions, guided by an LLM "scout" for qualitative pattern recognition.
* **Over what time frame?**
    The data was collected sequentially over a 10-week period, with one new batch of queries submitted weekly.

## Preprocessing and Uses
* **Have you applied any transformations?**
    * **Inputs ($X$):** No transformation is stored in the raw data, but the surrogate models internally apply `StandardScaler` (z-score normalization) before training.
    * **Outputs ($Y$):** Similarly, outputs are standardized during model training to ensure stable gradient descent for the neural networks.
* **Intended Uses:**
    * Benchmarking optimization algorithms.
    * Testing the robustness of surrogate models on small datasets.
    * Analyzing the "curse of dimensionality" in BBO tasks.
* **Inappropriate Uses:**
    * Training large-scale deep learning models (e.g., Transformers) from scratch, as the dataset is far too small (n=18) and would lead to immediate overfitting.
    * Inferring causal relationships, as the data is purely observational/interventional optimization data.

## Distribution and Maintenance
* **Where is the dataset available?**
    The dataset is hosted publicly in this GitHub repository under the `data/` directory.
* **Terms of Use:**
    The data is provided under the MIT License, allowing for open use and reproduction of the optimization experiments.
* **Who maintains it?**
    The dataset is maintained by the repository owner (SBrian Hogan) as part of the Capstone Project requirements.
