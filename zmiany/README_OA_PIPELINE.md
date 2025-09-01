# Pipeline do porównania modeli (OA severity)

Dwa kroki – analogicznie do Twojego podejścia:

## 1) Preprocessing pliku `.sav` → `text_data.csv`

Wymagania:
```bash
pip install pyreadstat pandas numpy
```

Uruchomienie (w folderze z plikiem `.sav`):
```bash
python preprocess_sav_to_csv.py -i CHECK_T0_DANS_ENG_20161128.sav -o text_data.csv
```

Skrypt:
- wczytuje SPSS `.sav`,
- standaryzuje nazwy kolumn,
- **automatycznie wybiera zmienną celu** jeśli znajdzie coś typu `severity_grade`, `kl_grade`, `kellgren`, `grade` itp. (mała liczba unikalnych wartości),
- jeśli nie znajdzie, **buduje `severity_score` na bazie WOMAC total** (albo subskal pain/stiff/function / innych kolumn z tymi frazami) i **dzieli na 3 klasy**: `mild`, `moderate`, `severe` (tercyle),
- robi prostą imputację braków,
- zapisuje `text_data.csv` z dwiema ostatnimi kolumnami: `severity_score`, `severity_grade`,
- zapisuje też plik `text_data.preprocessing_info.json` z metadanymi podejścia.

## 2) Trenowanie i porównanie modeli

Wymagania:
```bash
pip install scikit-learn pandas numpy matplotlib joblib
```

Uruchomienie:
```bash
python train_compare_models.py -i text_data.csv -o best_model.joblib
```

Skrypt:
- używa modeli: RandomForest, LogisticRegression, KNN, SVM, GradientBoosting, BernoulliNB,
- dzieli dane 80/20 (stratyfikacja), robi `get_dummies`,
- liczy **accuracy na holdoucie**, **5-fold CV**, **RepeatedStratifiedKFold (10×3)**,
- rysuje **macierze pomyłek** (matplotlib),
- wskazuje **najlepszy model** po `rep_cv_mean`,
- **zapisuje**: `best_model.joblib`, `artifacts/results_scores.csv`, macierze w `artifacts/conf_matrices/`, oraz `artifacts/feature_columns.json`.

## Uwaga dot. odtwarzania predykcji
Jeśli po czasie będziesz ładować `best_model.joblib`, pamiętaj aby na nowych danych wykonać **te same `get_dummies`** co podczas treningu (zgodnie z listą w `artifacts/feature_columns.json`) – brakujące kolumny dopełnić zerami, nadmiarowe wyrzucić.
