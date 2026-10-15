import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, confusion_matrix,
    classification_report
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def generate_dataset(n_samples: int = 500) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset for catalytic activity prediction.
    Features are correlated with the target in physically plausible ways.
    """
    surface_area   = np.random.uniform(50, 500, n_samples)      # m²g/
    band_gap       = np.random.uniform(0.5, 5.0, n_samples)     # eV
    metal_loading  = np.random.uniform(0.1, 20.0, n_samples)    # wt%
    temperature    = np.random.uniform(200, 800, n_samples)     # °C
    pressure       = np.random.uniform(1, 50, n_samples)        # bar

    activity = (
        0.40 * surface_area
        - 5.00 * band_gap
        + 3.00 * metal_loading
        + 0.10 * temperature
        + 0.50 * pressure
        + np.random.normal(0, 15, n_samples)   # measurement noise
    )

    df = pd.DataFrame({
        "surface_area":  surface_area,
        "band_gap":      band_gap,
        "metal_loading": metal_loading,
        "temperature":   temperature,
        "pressure":      pressure,
        "activity":      activity
    })

    for col in ["surface_area", "band_gap", "metal_loading"]:
        mask = np.random.rand(n_samples) < 0.05
        df.loc[mask, col] = np.nan

    return df

def preprocess(df: pd.DataFrame):
    """
    Handles missing values, separates features/target,
    scales features, and splits into train/test sets.
    """
    print("\n── PREPROCESSING ──────────────────────────────────────────")
    print(f"  Original shape : {df.shape}")
    print(f"  Missing values :\n{df.isnull().sum()}")

    df = df.fillna(df.median(numeric_only=True))
    print(f"  Missing after imputation: {df.isnull().sum().sum()}")

    feature_cols = ["surface_area", "band_gap", "metal_loading", "temperature", "pressure"]
    X = df[feature_cols].values
    y = df["activity"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"  Train samples  : {X_train.shape[0]}")
    print(f"  Test  samples  : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, feature_cols, scaler

def run_eda(df: pd.DataFrame, feature_cols: list):
    """
    Prints summary statistics and saves two EDA figures:
      - Correlation heatmap
      - Feature distribution plots
    """
    print("\n── EDA ─────────────────────────────────────────────────────")
    print(df[feature_cols + ["activity"]].describe().round(2))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
 
    corr = df[feature_cols + ["activity"]].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

    df[feature_cols].plot(
        kind="hist", bins=25, subplots=False, alpha=0.55, ax=axes[1]
    )
    axes[1].set_title("Feature Distributions", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Value")
    axes[1].legend(feature_cols, fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/eda_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  EDA figure saved → eda_plots.png")

def run_regression(X_train, X_test, y_train, y_test, feature_cols):
    """
    Trains six regression models and returns a results DataFrame
    along with the trained models dict.
    """
    models = {
        "Linear Regression":         LinearRegression(),
        "Ridge Regression":          Ridge(alpha=1.0),
        "Lasso Regression":          Lasso(alpha=0.1, max_iter=5000),
        "Decision Tree":             DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest":             RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_STATE),
        "KNN Regressor":             KNeighborsRegressor(n_neighbors=5),
    }

    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results.append({"Model": name, "R²": round(r2, 4), "MSE": round(mse, 2)})
        trained[name] = (model, y_pred)

    results_df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
    return results_df, trained

def run_classification(X_train, X_test, y_train, y_test):
    """
    Splits target at median to create binary classes:
      1 = high activity,  0 = low activity
    Trains four classifiers and returns a results DataFrame.
    """
    # Use TRAIN median as threshold (no data leakage from test set)
    threshold = np.median(y_train)
    y_train_cls = (y_train >= threshold).astype(int)
    y_test_cls  = (y_test  >= threshold).astype(int)

    models = {
        "Logistic Regression":         LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN Classifier":              KNeighborsClassifier(n_neighbors=5),
        "Decision Tree Classifier":    DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest Classifier":    RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_STATE),
    }

    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train_cls)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test_cls, y_pred)
        report = classification_report(y_test_cls, y_pred, output_dict=True)
        results.append({
            "Model":     name,
            "Accuracy":  round(acc, 4),
            "Precision": round(report["weighted avg"]["precision"], 4),
            "Recall":    round(report["weighted avg"]["recall"], 4),
            "F1-Score":  round(report["weighted avg"]["f1-score"], 4),
        })
        trained[name] = (model, y_pred, y_test_cls)

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return results_df, trained

def plot_regression_results(trained_reg, y_test):
    """
    Actual vs Predicted scatter plots for all regression models.
    """
    n_models = len(trained_reg)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (name, (model, y_pred)) in enumerate(trained_reg.items()):
        ax = axes[idx]
        ax.scatter(y_test, y_pred, alpha=0.5, color="steelblue", s=18, edgecolors="none")
        # Perfect prediction line
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual Activity")
        ax.set_ylabel("Predicted Activity")
        r2 = r2_score(y_test, y_pred)
        ax.legend(fontsize=8)
        ax.annotate(f"R²={r2:.3f}", xy=(0.05, 0.90), xycoords="axes fraction", fontsize=9)

    plt.suptitle("Actual vs Predicted — All Regression Models", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/regression_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Regression plots saved → regression_actual_vs_predicted.png")


def plot_feature_importance(trained_reg, feature_cols):
    """
    Feature importance bar charts for Decision Tree and Random Forest.
    """
    tree_models = {k: v for k, v in trained_reg.items() if "Tree" in k or "Forest" in k}
    fig, axes = plt.subplots(1, len(tree_models), figsize=(14, 5))

    for ax, (name, (model, _)) in zip(axes, tree_models.items()):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        ax.barh(
            [feature_cols[i] for i in idx],
            importances[idx],
            color="darkorange", edgecolor="black"
        )
        ax.set_title(f"Feature Importance\n{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Feature importance saved → feature_importance.png")


def plot_confusion_matrices(trained_cls):
    """
    Confusion matrix heatmaps for all classifiers.
    """
    n_models = len(trained_cls)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    for ax, (name, (model, y_pred, y_true)) in zip(axes, trained_cls.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Low", "High"], yticklabels=["Low", "High"]
        )
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — Classification Models", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Confusion matrices saved → confusion_matrices.png")


def plot_model_comparison(reg_results, cls_results):
    """
    Side-by-side bar charts comparing all models on key metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Regression — R² comparison
    colors_r = ["gold" if i == 0 else "steelblue" for i in range(len(reg_results))]
    axes[0].barh(reg_results["Model"], reg_results["R²"], color=colors_r, edgecolor="black")
    axes[0].set_title("Regression: R² Score", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("R²")
    axes[0].invert_yaxis()
    for i, v in enumerate(reg_results["R²"]):
        axes[0].text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)

    # Classification — Accuracy comparison
    colors_c = ["gold" if i == 0 else "teal" for i in range(len(cls_results))]
    axes[1].barh(cls_results["Model"], cls_results["Accuracy"], color=colors_c, edgecolor="black")
    axes[1].set_title("Classification: Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Accuracy")
    axes[1].invert_yaxis()
    for i, v in enumerate(cls_results["Accuracy"]):
        axes[1].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Model comparison chart saved → model_comparison.png")

def main():
    print("=" * 62)
    print("  CATALYTIC ACTIVITY ML PIPELINE")
    print("=" * 62)

    df = generate_dataset(n_samples=500)
    X_train, X_test, y_train, y_test, feature_cols, scaler = preprocess(df)

    run_eda(df, feature_cols)

    print("\n── REGRESSION RESULTS ──────────────────────────────────────")
    reg_results, trained_reg = run_regression(X_train, X_test, y_train, y_test, feature_cols)
    print(reg_results.to_string(index=False))

    print("\n── CLASSIFICATION RESULTS ──────────────────────────────────")
    cls_results, trained_cls = run_classification(X_train, X_test, y_train, y_test)
    print(cls_results.to_string(index=False))

    print("\n── SAVING VISUALIZATIONS ───────────────────────────────────")
    plot_regression_results(trained_reg, y_test)
    plot_feature_importance(trained_reg, feature_cols)
    plot_confusion_matrices(trained_cls)
    plot_model_comparison(reg_results, cls_results)

    best_reg = reg_results.iloc[0]
    best_cls = cls_results.iloc[0]
    print("\n── SUMMARY ─────────────────────────────────────────────────")
    print(f"  Best Regressor    : {best_reg['Model']}  (R²={best_reg['R²']}, MSE={best_reg['MSE']})")
    print(f"  Best Classifier   : {best_cls['Model']}  (Accuracy={best_cls['Accuracy']}, F1={best_cls['F1-Score']})")
    print("=" * 62)


if __name__ == "__main__":
    main()
