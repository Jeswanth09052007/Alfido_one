# Iris Classification Project

## Goal
Build a classification model to predict iris species using classic iris flower features:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Dataset
The dataset contains **150 rows** and **5 columns**.

Target column:

```text
species
```

Feature columns:

```text
sepal_length, sepal_width, petal_length, petal_width
```

## Work Completed

### 1. Exploratory Data Analysis
The notebook and script include:

- Dataset shape
- Missing value check
- Class distribution
- Feature summary
- Scatter plots for class separability

Generated plots are available in the `plots/` folder:

- `class_distribution.png`
- `petal_scatter.png`
- `sepal_scatter.png`

### 2. Models Trained and Compared

The following algorithms were trained:

- k-Nearest Neighbors
- Logistic Regression
- Decision Tree

Metrics reported:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- 5-fold cross-validation accuracy

Final comparison is saved in:

```text
metrics_comparison.csv
```

### 3. Best Model

Best model selected based on test accuracy, F1-score, and cross-validation score:

```text
Logistic Regression
```

Saved model files:

```text
models/best_iris_model.joblib
models/best_iris_model.pkl
```

## How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train models and generate outputs

```bash
python train_model.py
```

### Step 3: Run example inference

```bash
python inference.py
```

## Example Inference Code

```python
import joblib
import pandas as pd

model = joblib.load("models/best_iris_model.joblib")

sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

prediction = model.predict(sample)[0]
print("Predicted iris species:", prediction)
```

## Submission Checklist

Submit these items:

- `notebooks/Iris_Classification.ipynb`
- `models/best_iris_model.joblib`
- `models/best_iris_model.pkl`
- `README.md`
- `inference.py`
- `requirements.txt`
- `plots/` folder
- `metrics_comparison.csv`

Recommended submission format:

```text
iris_classification_project.zip
```

## Notes
The Iris dataset is small and clean, so high accuracy is expected. Petal length and petal width provide strong class separation, especially for Iris-setosa.

## Submission Instructions as per Task Rules

For this task, submit the work using the following structure:

1. Push this complete project folder to GitHub.
2. Upload large artifacts, if any, to Google Drive and set sharing to `Anyone with the link can view`.
3. Use `docs/Iris_Classification_Submission_Document.docx` or `docs/Iris_Classification_Submission_Document.pdf` as the required separate document.
4. Fill the document with your GitHub repository link, notebook link, model file link, screenshots/plots link, and task number.
5. On the Task Submission page, paste all links clearly and mention the task number.

Recommended final links to submit:

- GitHub Repository Link
- Notebook Link: `notebooks/Iris_Classification.ipynb`
- Model File Link: `models/best_iris_model.joblib` or `models/best_iris_model.pkl`
- Screenshots/Plots Link: `plots/` folder
- Submission Document: `docs/Iris_Classification_Submission_Document.pdf`

