import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report

DATA_PATH = Path("data/iris.csv")
PLOTS_DIR = Path("plots")
MODELS_DIR = Path("models")
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target = "species"

X = df[features]
y = df[target]

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nClass distribution:\n", df[target].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "k-NN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cv_accuracy = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

    results.append({
        "Model": name,
        "Test Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "5-Fold CV Accuracy": cv_accuracy
    })

    print(f"\n{name}")
    print("-" * 50)
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results).sort_values(
    ["Test Accuracy", "F1 Score", "5-Fold CV Accuracy"], ascending=False
)
results_df.to_csv("metrics_comparison.csv", index=False)

best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]
best_model.fit(X_train, y_train)

joblib.dump(best_model, MODELS_DIR / "best_iris_model.joblib")
with open(MODELS_DIR / "best_iris_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nModel comparison:")
print(results_df)
print(f"\nBest model saved: {best_name}")

# Plots
counts = df[target].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(counts.index, counts.values)
plt.title("Class Distribution")
plt.xlabel("Species")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 5))
for species, group in df.groupby(target):
    plt.scatter(group["petal_length"], group["petal_width"], label=species)
plt.title("Class Separability: Petal Length vs Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "petal_scatter.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 5))
for species, group in df.groupby(target):
    plt.scatter(group["sepal_length"], group["sepal_width"], label=species)
plt.title("Class Separability: Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "sepal_scatter.png", dpi=180)
plt.close()

x = np.arange(len(results_df["Model"]))
width = 0.2
plt.figure(figsize=(8, 5))
plt.bar(x - width * 1.5, results_df["Test Accuracy"], width, label="Accuracy")
plt.bar(x - width / 2, results_df["Precision"], width, label="Precision")
plt.bar(x + width / 2, results_df["Recall"], width, label="Recall")
plt.bar(x + width * 1.5, results_df["F1 Score"], width, label="F1")
plt.xticks(x, results_df["Model"], rotation=15)
plt.ylim(0, 1.1)
plt.title("Model Metrics Comparison")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "metrics_comparison.png", dpi=180)
plt.close()

y_pred = best_model.predict(X_test)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(f"Confusion Matrix - {best_name}")
plt.colorbar()
plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
plt.yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_best_model.png", dpi=180)
plt.close()
