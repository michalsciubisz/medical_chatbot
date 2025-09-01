# import argparse
# import json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib

# from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer

# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import BernoulliNB

# def plot_and_save_confusion_matrix(y_true, y_pred, classes, title, path):
#     cm = confusion_matrix(y_true, y_pred, labels=classes)
#     fig, ax = plt.subplots(figsize=(6, 5))
#     im = ax.imshow(cm, aspect='auto')
#     ax.set_title(title)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_xticks(range(len(classes)))
#     ax.set_yticks(range(len(classes)))
#     ax.set_xticklabels(classes, rotation=45, ha='right')
#     ax.set_yticklabels(classes)
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, cm[i, j], ha="center", va="center")
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     plt.tight_layout()
#     fig.savefig(path)
#     plt.close(fig)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-i", "--input", default="text_data.csv", help="Wejściowy CSV z preprocessu")
#     ap.add_argument("-o", "--output", default="best_model.joblib", help="Ścieżka do zapisu najlepszego modelu")
#     ap.add_argument("--keep-missing", action="store_true", help="Nie usuwaj wierszy z severity_grade=='missing'")
#     ap.add_argument("--test-size", type=float, default=0.2, help="Udział zbioru testowego (domyślnie 0.2)")
#     ap.add_argument("--min-test-per-class", type=int, default=1,
#                     help="Minimalna liczba obserwacji każdej klasy w zbiorze testowym (domyślnie 1)")
#     args = ap.parse_args()

#     df = pd.read_csv(args.input)
#     assert "severity_grade" in df.columns, "CSV musi zawierać kolumnę 'severity_grade'!"

#     # Opcjonalnie usuń missing
#     if not args.keep_missing:
#         before = len(df)
#         df = df[df["severity_grade"].astype(str) != "missing"].copy()
#         print(f"[i] Usunięto {before - len(df)} wierszy z missing target")
#         if df["severity_grade"].nunique() < 2:
#             raise SystemExit("Po usunięciu 'missing' zostało < 2 klas. Użyj --keep-missing lub popraw preprocess.")

#     # Podział X/y
#     y = df["severity_grade"].astype(str)
#     X = df.drop(columns=["severity_grade", "severity_score"], errors="ignore")

#     # Rozpoznanie typów
#     num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
#     cat_cols_all = [c for c in X.columns if c not in num_cols]

#     # Preprocessing: impute + scale numeryczne, impute + one-hot kategoryczne
#     num_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler(with_mean=False))  # with_mean=False – bezpieczniej dla sparse
#     ])
#     cat_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True))
#     ])
#     pre = ColumnTransformer([
#         ("num", num_pipe, num_cols),
#         ("cat", cat_pipe, cat_cols_all)
#     ])

#     # Modele (z class_weight tam, gdzie ma sens)
#     models = {
#         'Random Forest': RandomForestClassifier(random_state=42, class_weight="balanced"),
#         'Logistic Regression': LogisticRegression(max_iter=1000, class_weight="balanced"),
#         'KNN': KNeighborsClassifier(n_neighbors=5),
#         'SVM': SVC(class_weight="balanced", probability=False),
#         'Gradient Boosting': GradientBoostingClassifier(random_state=42),  # GB nie ma class_weight
#         'Naive Bayes': BernoulliNB(),
#     }

#     pipelines = {name: Pipeline([("pre", pre), ("clf", clf)]) for name, clf in models.items()}

#     # Train/test split – ważne: STRATYFIKACJA
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     Path("artifacts/conf_matrices").mkdir(parents=True, exist_ok=True)

#     results = {}
#     reports = {}
#     classes_sorted = sorted(y_train.unique().tolist())

#     # Holdout + raporty
#     for name, pipe in pipelines.items():
#         pipe.fit(X_train, y_train)
#         preds = pipe.predict(X_test)
#         results[name] = {
#             "holdout_accuracy": float(accuracy_score(y_test, preds)),
#             "holdout_bal_acc": float(balanced_accuracy_score(y_test, preds)),
#             "holdout_f1_macro": float(f1_score(y_test, preds, average="macro")),
#         }
#         reports[name] = classification_report(y_test, preds, output_dict=True, zero_division=0)
#         plot_and_save_confusion_matrix(
#             y_test, preds, classes_sorted,
#             f"{name} – Confusion Matrix",
#             f"artifacts/conf_matrices/cm_{name.replace(' ','_')}.png"
#         )

#     # CV – ważne przy niezbalansowanych klasach: użyjemy balanced_accuracy
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
#     for name, pipe in pipelines.items():
#         acc = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
#         bal = cross_val_score(pipe, X, y, cv=cv, scoring='balanced_accuracy')
#         f1m = cross_val_score(pipe, X, y, cv=cv, scoring='f1_macro')
#         results[name].update({
#             "cv_acc_mean": float(acc.mean()), "cv_acc_std": float(acc.std()),
#             "cv_balacc_mean": float(bal.mean()), "cv_balacc_std": float(bal.std()),
#             "cv_f1_macro_mean": float(f1m.mean()), "cv_f1_macro_std": float(f1m.std()),
#         })

#     # Wybór najlepszego – wg balanced accuracy w CV (najbardziej sensowne przy niezbalansowaniu)
#     best_name = max(results.keys(), key=lambda n: results[n]["cv_balacc_mean"])
#     best_pipe = pipelines[best_name]

#     # Fit na całości i zapis
#     best_pipe.fit(X, y)
#     joblib.dump(best_pipe, args.output)

#     # Zapis listy cech po OneHot (dla replikacji w produkcji)
#     # Uwaga: pobranie nazw kolumn po transformacji wymaga dopasowanego preprocesora:
#     pre_fitted = best_pipe.named_steps["pre"]
#     feat_names_num = num_cols
#     # kategorie dla OneHot:
#     ohe = pre_fitted.named_transformers_["cat"].named_steps["onehot"]
#     feat_names_cat = []
#     if len(cat_cols_all) > 0:
#         cats = ohe.categories_
#         for col, vals in zip(cat_cols_all, cats):
#             feat_names_cat += [f"{col}={v}" for v in vals]
#     feature_columns = feat_names_num + feat_names_cat

#     Path("artifacts").mkdir(exist_ok=True)
#     with open("artifacts/feature_columns.json", "w", encoding="utf-8") as f:
#         json.dump(feature_columns, f, ensure_ascii=False, indent=2)

#     # Podsumowanie i zapis wyników
#     res_df = pd.DataFrame(results).T.sort_values(
#         by=["cv_balacc_mean", "cv_f1_macro_mean", "cv_acc_mean"], ascending=False
#     )
#     Path("artifacts").mkdir(exist_ok=True)
#     res_df.to_csv("artifacts/results_scores.csv")

#     print("[OK] Najlepszy model:", best_name)
#     print(res_df)
#     summary = {
#         "best_model": best_name,
#         "scores": results[best_name],
#         "paths": {
#             "results_scores": "artifacts/results_scores.csv",
#             "best_model_joblib": args.output,
#             "confusion_matrices_dir": "artifacts/conf_matrices",
#             "feature_columns": "artifacts/feature_columns.json"
#         }
#     }
#     with open("artifacts/summary.json", "w", encoding="utf-8") as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     main()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB


def plot_and_save_confusion_matrix(y_true, y_pred, classes, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="text_data.csv", help="Wejściowy CSV z preprocessu")
    ap.add_argument("-o", "--output", default="best_model.joblib", help="Ścieżka do zapisu najlepszego modelu")
    ap.add_argument("--keep-missing", action="store_true", help="Nie usuwaj wierszy z severity_grade=='missing'")
    ap.add_argument("--test-size", type=float, default=0.2, help="Udział zbioru testowego (domyślnie 0.2)")
    ap.add_argument("--min-test-per-class", type=int, default=1,
                    help="Minimalna liczba obserwacji każdej klasy w teście (domyślnie 1)")
    args = ap.parse_args()

    # ====== Wczytanie ======
    df = pd.read_csv(args.input)
    assert "severity_grade" in df.columns, "CSV musi zawierać kolumnę 'severity_grade'!"

    # Opcjonalnie usuń missing
    if not args.keep_missing:
        before = len(df)
        df = df[df["severity_grade"].astype(str) != "missing"].copy()
        print(f"[i] Usunięto {before - len(df)} wierszy z missing target")
        if df["severity_grade"].nunique() < 2:
            raise SystemExit("Po usunięciu 'missing' zostało < 2 klas. Użyj --keep-missing lub popraw preprocess.")

    # Podział X/y
    y = df["severity_grade"].astype(str)
    X = df.drop(columns=["severity_grade", "severity_score"], errors="ignore")

    # Podgląd rozkładu klas przed podziałem
    print("\n[i] Rozkład klas (całość):")
    print(y.value_counts().sort_index())

    # ====== Rozpoznanie typów ======
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_all = [c for c in X.columns if c not in num_cols]

    # ====== Preprocessing ======
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))  # bezpiecznie dla zbiorów sparse
    ])
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
    # Jeśli są kategorie – dodaj OneHot
    if len(cat_cols_all) > 0:
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)))
    cat_pipe = Pipeline(cat_steps)

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", num_pipe, num_cols))
    if len(cat_cols_all) > 0:
        transformers.append(("cat", cat_pipe, cat_cols_all))
    pre = ColumnTransformer(transformers)

    # ====== Modele ======
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, class_weight="balanced"),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight="balanced"),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(class_weight="balanced", probability=False),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),  # GB nie ma class_weight
        'Naive Bayes': BernoulliNB(),
    }
    pipelines = {name: Pipeline([("pre", pre), ("clf", clf)]) for name, clf in models.items()}

    # ====== Train/Test split z gwarancją min. liczby próbek klasy w teście ======
    requested_ts = float(args.test_size)
    vc = y.value_counts()
    needed_ts = (args.min_test_per_class / vc).max()  # minimalny udział, by każda klasa miała >= min-test-per-class
    test_size = max(requested_ts, float(needed_ts))

    if test_size >= 0.5:
        print(f"[!] Uwaga: test_size wyliczony na {test_size:.3f} (>= 0.5). "
              f"To sugeruje bardzo małą klasę względem wymaganego minimum. "
              f"Rozważ zmniejszenie --min-test-per-class lub zebranie większej liczby obserwacji tej klasy.")
        test_size = min(test_size, 0.49)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Raport rozkładu klas po podziale
    def _counts(lbls, name):
        c = pd.Series(lbls).value_counts().sort_index()
        p = (c / c.sum() * 100).round(2)
        rep = pd.DataFrame({"count": c, "percent": p})
        print(f"\n[i] Rozkład klas w {name}:")
        print(rep.to_string())

    _counts(y_train, "TRAIN")
    _counts(y_test, "TEST")
    print(f"[i] Ostateczny test_size: {test_size:.3f} (żądany {requested_ts:.3f}, wymagany >= {float(needed_ts):.3f})")

    Path("artifacts/conf_matrices").mkdir(parents=True, exist_ok=True)

    # ====== Holdout + raporty ======
    results = {}
    reports = {}
    classes_sorted = sorted(y_train.unique().tolist())

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "holdout_accuracy": float(accuracy_score(y_test, preds)),
            "holdout_bal_acc": float(balanced_accuracy_score(y_test, preds)),
            "holdout_f1_macro": float(f1_score(y_test, preds, average="macro")),
        }
        reports[name] = classification_report(y_test, preds, output_dict=True, zero_division=0)
        plot_and_save_confusion_matrix(
            y_test, preds, classes_sorted,
            f"{name} – Confusion Matrix",
            f"artifacts/conf_matrices/cm_{name.replace(' ', '_')}.png"
        )

    # ====== Cross-Validation (zbalansowana metryka) ======
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    for name, pipe in pipelines.items():
        acc = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
        bal = cross_val_score(pipe, X, y, cv=cv, scoring='balanced_accuracy')
        f1m = cross_val_score(pipe, X, y, cv=cv, scoring='f1_macro')
        results[name].update({
            "cv_acc_mean": float(acc.mean()), "cv_acc_std": float(acc.std()),
            "cv_balacc_mean": float(bal.mean()), "cv_balacc_std": float(bal.std()),
            "cv_f1_macro_mean": float(f1m.mean()), "cv_f1_macro_std": float(f1m.std()),
        })

    # ====== Wybór najlepszego modelu wg balanced accuracy w CV ======
    best_name = max(results.keys(), key=lambda n: results[n]["cv_balacc_mean"])
    best_pipe = pipelines[best_name]

    # Fit na CAŁOŚCI i zapis modelu
    best_pipe.fit(X, y)
    joblib.dump(best_pipe, args.output)

    # ====== Zapis listy cech po OneHot (dla replikacji) ======
    feature_columns = []
    if len(num_cols) > 0:
        feature_columns += list(num_cols)

    # jeśli były kategorie – wyciągnij ich kategorie z dopasowanego OHE
    if len(cat_cols_all) > 0:
        pre_fitted = best_pipe.named_steps["pre"]
        ohe = pre_fitted.named_transformers_["cat"].named_steps.get("onehot", None)
        if ohe is not None:
            cats = ohe.categories_
            for col, vals in zip(cat_cols_all, cats):
                feature_columns += [f"{col}={v}" for v in vals]

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)

    # ====== Podsumowanie i zapis wyników ======
    res_df = pd.DataFrame(results).T.sort_values(
        by=["cv_balacc_mean", "cv_f1_macro_mean", "cv_acc_mean"], ascending=False
    )
    res_df.to_csv("artifacts/results_scores.csv")

    print("[OK] Najlepszy model:", best_name)
    print(res_df)
    summary = {
        "best_model": best_name,
        "scores": results[best_name],
        "paths": {
            "results_scores": "artifacts/results_scores.csv",
            "best_model_joblib": args.output,
            "confusion_matrices_dir": "artifacts/conf_matrices",
            "feature_columns": "artifacts/feature_columns.json"
        }
    }
    with open("artifacts/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
