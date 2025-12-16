# Model Card for Deep Ensemble BBO Strategy

## Overview
* **Name:** Hybrid Deep Ensemble with UCB Acquisition
* **Type:** Bayesian Optimization Surrogate Model
* **Version:** 2.1 (Transitioned from Gaussian Process to Deep Ensemble in Week 6)

## Intended Use
* **Suitable Tasks:**
    * Optimization of expensive black-box functions where gradients are unavailable.
    * High-dimensional search spaces (up to 8D) where standard Gaussian Processes struggle with computational scaling and kernel selection.
    * Scenarios requiring robust uncertainty estimation to balance exploration and exploitation.
* **Use Cases to Avoid:**
    * Real-time applications requiring sub-millisecond inference (the ensemble is computationally heavy).
    * Convex optimization problems where gradient descent would be exponentially faster.
    * Problems with thousands of available data points (where standard Deep Learning would outperform this specialized ensemble approach).

## Details of the Approach
* **Strategy Evolution:**
    * **Rounds 1-5:** Utilized `sklearn.GaussianProcessRegressor` with an RBF kernel. This worked well for low-D functions but failed to capture complex interactions in 8D.
    * **Rounds 6-10:** Transitioned to a **Deep Ensemble**.
        * **Architecture:** 10 independent Feed-Forward Neural Networks.
        * **Structure:** Each network has 2 hidden layers of 128 neurons (`Dense(128, activation='relu')`).
        * **Uncertainty:** Aleatoric and Epistemic uncertainty is approximated by the **variance** of the predictions across the 10 models.
        * **Training:** Uses **Bagging** (bootstrapped sampling) and `EarlyStopping` to prevent overfitting on the small dataset.
        * **Acquisition:** Upper Confidence Bound (UCB) with a dynamic $\beta$ parameter to guide the search.

## Performance
* **Metrics:**
    * **Max $Y$ Found:** The primary metric is the highest function value discovered so far.
    * **Query Efficiency:** The improvement in Max $Y$ per new query.
    * **Robustness:** The ability of the ensemble to avoid "collapse" (zero variance).
* **Summary of Results:**
    * **Functions 1-3 (Low-D):** The ensemble matched the performance of the GP, finding stable optima.
    * **Functions 4-8 (High-D):** The Deep Ensemble significantly outperformed the GP. It successfully identified non-linear interactions in Function 6 and adapted to a sudden "spike" in Function 5, leveraging its hierarchical feature learning capabilities.

## Assumptions and Limitations
* **Assumptions:**
    * **Smoothness:** Assumes the underlying black-box functions are continuous enough to be approximated by ReLU networks. Discontinuous "Dirac delta" spikes may be missed.
    * **Stationarity:** Assumes the function landscape does not change over time (though our strategy adapts if new data reveals non-stationarity).
* **Limitations:**
    * **Data Starvation:** With only ~18 points, the risk of overfitting is high. The model relies heavily on `EarlyStopping` and Bagging to mitigate this.
    * **Corner Bias:** The Latin Hypercube initialization under-sampled the extreme corners of the 8D space, potentially leaving global optima in the "corners" undiscovered.

## Ethical Considerations
* **Transparency & Reproducibility:**
    * This approach prioritizes transparency by using standard libraries (TensorFlow/Keras) and explicitly documenting hyperparameters (learning rate, batch size) in the repository.
    * By publishing this Model Card and the Datasheet, we ensure that other researchers can reproduce the specific "pivot" from GP to Deep Learning, understanding *why* that architectural choice was made (scalability vs. data efficiency).
* **Real-World Adaptation:**
    * The "Ensemble" approach mimics real-world decision-making committees. Understanding how to aggregate conflicting model predictions (high variance) is a key skill for deploying reliable AI in safety-critical domains like autonomous driving or medical diagnosis.
