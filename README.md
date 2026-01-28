# Black-Box Optimisation Capstone Project

## Project Overview (Non-Technical Write-Up)
In the world of engineering and science, we often face "Black Box" problems: systems where we can adjust the inputs and measure the outputs, but we don't know the mathematical formula inside. Optimising these systems—finding the best possible inputs—is usually slow and expensive.

This project developed an Artificial Intelligence solution to solve this problem efficiently. I moved beyond standard statistical guesses by building a "Deep Ensemble"—a committee of neural networks that vote on the best next step. For highly complex problems (8 dimensions), I integrated dimensionality reduction (PCA) to simplify the search space. This approach allowed the system to learn from limited data, identifying optimal solutions faster and more reliably than traditional methods.

## Repository Contents
* **`BBO_Capstone_Project.ipynb`**: The main Jupyter Notebook containing the full code, methodology, and results.
* **[Datasheet for BBO Dataset](DATASHEET.md)**: Documentation detailing the creation, composition, and intended use of the dataset.
* **[Model Card for Optimization Strategy](MODEL_CARD.md)**: A summary of the machine learning models used (Deep Ensemble + PCA), their limitations, and performance metrics.
* **`requirements.txt`**: A list of Python libraries required to run the code.
* **`Capstone_Data/`**: Folder containing the input/output data used to train the models.

## How to Run the Code
1.  Clone this repository.
2.  Install requirements: `pip install -r requirements.txt`
3.  Open `BBO_Capstone_Project.ipynb` in Jupyter Notebook.
4.  Ensure the data paths in Cell 2 point to the `Capstone_Data` folder.

## Methodology Highlights
* **Deep Ensembles:** Used to quantify uncertainty in high-dimensional spaces.
* **PCA (Principal Component Analysis):** Applied to 8D functions to reduce noise and focus the search.
* **Bagging:** Implemented to prevent model overfitting on small datasets.
