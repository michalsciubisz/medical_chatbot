import argparse
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- helpers ----------

def normalize_col(col: str) -> str:
    c = str(col).strip().lower()
    c = re.sub(r"\s+", "_", c)
    c = c.replace("-", "_").replace("/", "_")
    c = re.sub(r"[^a-z0-9_]", "", c)
    return c

def is_good_target(s: pd.Series) -> bool:
    nun = s.dropna().nunique()
    return 2 <= nun <= 12

def try_read_sav(path: str, encodings=None):
    try:
        import pyreadstat
    except ImportError as e:
        raise SystemExit("Brak pakietu pyreadstat (wymagane dla .sav etykiet). Zainstaluj: pip install pyreadstat") from e

    if encodings is None:
        encodings = ["utf-8", "cp1252", "latin1", "cp1250"]

    last_err = None
    for enc in encodings:
        try:
            df, meta = pyreadstat.read_sav(path, encoding=enc)
            return df, meta, enc
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Nieznany błąd dekodowania")

def try_read_dta(path: str):
    """
    Czyta .dta z pyreadstat, żeby mieć etykiety. Jeśli pyreadstat brak, spada do pandas.read_stata (bez etykiet).
    """
    try:
        import pyreadstat
        df, meta = pyreadstat.read_dta(path, apply_value_formats=False)
        return df, meta, "pyreadstat.read_dta"
    except Exception:
        # fallback: bez etykiet
        df = pd.read_stata(path, convert_categoricals=False)
        meta = None
        return df, meta, "pandas.read_stata"

from collections.abc import Mapping

def build_readable_names(columns, meta):
    """
    Tworzy mapę 'techniczna_nazwa' -> 'czytelna_nazwa' z etykiet zmiennych.
    Działa zarówno gdy meta.column_labels jest dict-em, jak i gdy jest listą (pyreadstat dla .dta/.sav).
    """
    mapping = {}
    labels_map = {}

    if meta is not None and hasattr(meta, "column_labels") and meta.column_labels:
        # 1) Jeśli to słownik – super prosto
        if isinstance(meta.column_labels, Mapping):
            labels_map = dict(meta.column_labels)
        # 2) Jeśli to lista/tupla – sparuj z column_names albo z aktualną listą 'columns'
        elif isinstance(meta.column_labels, (list, tuple)):
            meta_names = getattr(meta, "column_names", None)
            labels_list = list(meta.column_labels)

            if meta_names and len(meta_names) == len(labels_list):
                labels_map = {str(meta_names[i]): labels_list[i] for i in range(len(labels_list))}
            elif len(columns) == len(labels_list):
                # fallback pozycją względem już odczytanych kolumn
                labels_map = {str(columns[i]): labels_list[i] for i in range(len(labels_list))}
            else:
                labels_map = {}  # format niepasujący – trudno, użyjemy nazw technicznych

    # Zbuduj mapę: oryg -> ładna_nazwa (albo z etykiety, albo z nazwy technicznej)
    for orig in columns:
        label = None
        if labels_map:
            label = labels_map.get(str(orig))
        if label and str(label).strip():
            pretty = normalize_col(str(label))
            if not pretty:
                pretty = normalize_col(orig)
        else:
            pretty = normalize_col(orig)
        mapping[orig] = pretty

    # Zapewnij unikalność nazw (doklej sufiksy _2, _3, ...)
    seen = {}
    unique_map = {}
    for k, v in mapping.items():
        if v not in seen:
            seen[v] = 1
            unique_map[k] = v
        else:
            seen[v] += 1
            unique_map[k] = f"{v}_{seen[v]}"

    return unique_map

def minmax01(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return s.fillna(np.nan)  # nie normalizujemy gdy brak zakresu
    return (s - mn) / (mx - mn)

def _parse_kl_value(x):
    """
    Zamienia wartość na int 0..4 jeśli to możliwe.
    Obsługuje napisy typu 'KL 2', 'grade 3', 'kellgren 4', itp.
    Zwraca np.nan jeśli nie da się sparsować.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    # wyłuskaj pierwszą cyfrę 0..4
    m = re.search(r"\b([0-4])\b", s)
    if m:
        return int(m.group(1))
    # czasem wartości są floatami 0.0..4.0
    try:
        v = float(s.replace(",", "."))
        if 0 <= v <= 4:
            return int(round(v))
    except Exception:
        pass
    return np.nan

def _pick_kl_across_cols(df, kl_cols, strategy="max"):
    """
    Zbiera KL z wielu kolumn i agreguje wg strategii:
    - max: maksimum po wierszu
    - left/right: preferuj kolumny zawierające 'left|l' lub 'right|r' w nazwie
    - first_nonnull: pierwsza nie-NaN w zadanej kolejności kolumn
    Zwraca pd.Series z int 0..4 lub NaN.
    """
    # sparsuj każdą kolumnę do 0..4
    parsed = []
    for c in kl_cols:
        parsed.append(df[c].apply(_parse_kl_value))
    if not parsed:
        return pd.Series(np.nan, index=df.index)

    M = pd.concat(parsed, axis=1)
    M.columns = kl_cols

    strategy = strategy.lower()
    if strategy == "max":
        return M.max(axis=1, skipna=True)
    elif strategy in {"left", "right"}:
        # heuristic: wybierz kolumny zawierające słowo/skrót po stronie
        key = "left" if strategy == "left" else "right"
        # preferowane kolumny
        pref = [c for c in kl_cols if re.search(rf"\b{key}\b|_{key}\b|\b{key[0]}\b|_{key[0]}\b", c)]
        # jeśli brak dopasowania, spadnij do first_nonnull
        order = pref + [c for c in kl_cols if c not in pref]
        for c in order:
            s = M[c]
            if s.notna().any():
                # wybierz po wierszu (pierwsza nie-NaN)
                out = s.copy()
                # wiersze NaN uzupełnij kolejnymi kolumnami z order
                na_mask = out.isna()
                for c2 in order:
                    if c2 == c:
                        continue
                    out[na_mask] = out[na_mask].fillna(M[c2][na_mask])
                    na_mask = out.isna()
                    if not na_mask.any():
                        break
                return out
        return M.max(axis=1, skipna=True)  # awaryjnie
    else:  # first_nonnull
        out = pd.Series(np.nan, index=df.index)
        for c in kl_cols:
            v = M[c]
            na_mask = out.isna()
            out[na_mask] = v[na_mask]
            if not out.isna().any():
                break
        return out

def robust_binning(series: pd.Series):
    """
    Binning z qcut (3-koszyki) z fallbackiem do 2-koszyków, w ostateczności median split.
    Zwraca: (labels_series(str), bin_edges(list|None), bin_labels(list))
    """
    s = series.astype(float)
    # delikatny jitter gdy bardzo mało unikatów
    if s.dropna().nunique() < 3:
        s = s + np.random.normal(0, 1e-6, size=len(s))

    labels3 = ["mild", "moderate", "severe"]
    try:
        cats = pd.qcut(s, q=3, labels=labels3, duplicates="drop")
        if cats.isna().all() or cats.nunique(dropna=True) < 3:
            raise ValueError
        _, edges = pd.qcut(s, q=3, retbins=True, duplicates="drop")
        return cats.astype(str), list(np.unique(edges).tolist()), labels3
    except Exception:
        labels2 = ["low", "high"]
        try:
            cats = pd.qcut(s, q=2, labels=labels2, duplicates="drop")
            if cats.isna().all() or cats.nunique(dropna=True) < 2:
                raise ValueError
            _, edges = pd.qcut(s, q=2, retbins=True, duplicates="drop")
            return cats.astype(str), list(np.unique(edges).tolist()), labels2
        except Exception:
            med = float(np.nanmedian(s))
            cats = np.where(s <= med, "low", "high")
            return pd.Series(cats, index=series.index).astype(str), [-np.inf, med, np.inf], ["low", "high"]

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Ścieżka do pliku .sav lub .dta")
    ap.add_argument("-o", "--output", default="text_data.csv", help="Wyjściowy CSV (domyślnie text_data.csv)")
    ap.add_argument("--encoding", default=None, help="(dla .sav) Wymuś kodowanie, np. cp1252/cp1250/latin1")
    ap.add_argument("--score-cols", default=None,
                    help="CSV z nazwami kolumn do severity_score (po normalizacji nazw). Np: pain_total,stiff_total,function_total")
    ap.add_argument("--score-patterns", default=None,
                    help="CSV z regexami kolumn, które mają wejść do score. Np: (?i)womac.*pain,(?i)womac.*stiff,(?i)womac.*function")
    ap.add_argument("--reverse-cols", default=None,
                    help="CSV z kolumnami, które trzeba odwrócić (1-x) po normalizacji (np. gdy większe=lepiej).")
    ap.add_argument("--drop-missing-target", action="store_true",
                    help="Jeśli ustawione, wiersze z severity_grade=='missing' zostaną usunięte z wyjścia.")
    ap.add_argument("--target-kl", action="store_true",
                    help="Wymuś budowanie targetu w skali Kellgrena–Lawrence’a (KL 0–4).")
    ap.add_argument("--kl-cols", default=None,
                    help="CSV z nazwami kolumn zawierających ocenę KL (po normalizacji nazw). Np.: kl_left,kl_right")
    ap.add_argument("--kl-strategy", default="max", choices=["max", "left", "right", "first_nonnull"],
                    help="Jak łączyć wiele kolumn KL: max (domyślnie), lewa, prawa, albo pierwsza nie-NaN.")
    ap.add_argument("--infer-kl0", action="store_true",
                    help="Jeśli w danych brakuje KL0, spróbuj wnioskować KL0 z WOMAC lub z dolnego kwantyla score’u dla rekordów z missing.")
    ap.add_argument("--kl0-womac-thresh", type=float, default=0.02,
                    help="Próg (0–1) dla znormalizowanych składowych WOMAC, poniżej którego uznajemy \"brak dolegliwości\" (domyślnie 0.02).")
    ap.add_argument("--kl0-quantile", type=float, default=0.03,
                    help="Udział dolnego kwantyla severity_score (0–1) w missing do oznaczenia jako KL0, jeśli WOMAC brak (domyślnie 0.03).")
    ap.add_argument("--kl0-min", type=int, default=0,
                    help="Minimalna liczba przykładów KL0. Jeśli brakuje, skrypt dobierze z missing/KL1 o najniższym soft-score, a gdy nadal brak – dosyntetyzuje.")
    ap.add_argument("--kl0-synth-when-needed", action="store_true",
                    help="Pozwól na syntezę danych KL0, gdy po dorelabelowaniu nadal brakuje do --kl0-min.")
    ap.add_argument("--kl0-synth-jitter", type=float, default=0.01,
                    help="Siła jittera dla cech numerycznych przy syntezie (jako ułamek zakresu cechy, domyślnie 0.01=1%).")
    ap.add_argument("--kl0-pick-from-kl1-quantile", type=float, default=0.10,
                    help="Maksymalny kwantyl soft-score w KL1, z którego wolno dorelabelować do KL0, gdy missing nie wystarcza (domyślnie 0.10).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_csv = Path(args.output)
    out_json = out_csv.with_suffix(".preprocessing_info.json")
    Path("artifacts").mkdir(exist_ok=True)

    # 1) Wczytanie + etykiety
    ext = in_path.suffix.lower()
    used_reader, used_encoding = None, None
    if ext == ".dta":
        df_raw, meta, used_reader = try_read_dta(str(in_path))
    elif ext == ".sav":
        if args.encoding:
            df_raw, meta, used_encoding = try_read_sav(str(in_path), encodings=[args.encoding])
        else:
            df_raw, meta, used_encoding = try_read_sav(str(in_path))
        used_reader = f"pyreadstat.read_sav (encoding={used_encoding})"
    else:
        raise SystemExit("Obsługiwane są pliki .sav oraz .dta")

    print(f"[i] Wczytano {in_path.name} via {used_reader}")

    # 2) Przyjazne nazwy kolumn z etykiet (jeśli dostępne)
    name_map = build_readable_names(df_raw.columns, meta)
    df_raw = df_raw.rename(columns=name_map)
    # zapis mapowania
    pd.DataFrame({"original": list(name_map.keys()), "renamed": list(name_map.values())}) \
      .to_csv("artifacts/column_mapping.csv", index=False)

    # 3) Normalize nagłówków jeszcze raz (na wszelki wypadek)
    df_raw.columns = [normalize_col(c) for c in df_raw.columns]

    # 4) Target – wykrycie lub budowa
    candidate_cols = [c for c in df_raw.columns if any(k in c for k in [
        "severity_grade","kl_grade","kellgren","kelgren","oa_grade","severityclass","severity_class"
    ])]
    target_col = None
    for c in candidate_cols:
        try:
            if is_good_target(df_raw[c]):
                target_col = c
                break
        except Exception:
            continue

    preproc_info = {
        "target_strategy": None,
        "target_col": None,
        "bin_edges": None,
        "bin_labels": None,
        "source_file": in_path.name,
        "reader": used_reader,
        "used_encoding": used_encoding,
        "columns_renamed_from_labels": True if meta is not None else False
    }

    kl_cols = []
    if args.kl_cols:
        kl_cols = [normalize_col(c) for c in args.kl_cols.split(",")]
        kl_cols = [c for c in kl_cols if c in df_raw.columns]
    else:
        # spróbuj automatycznie wykryć KL
        # częste nazwy: kl, kellgren, kellgren_lawr, kl_grade, klleft/klright
        candidates = [c for c in df_raw.columns if re.search(r"(kellgren|kl(_?grade)?|kellgren_?law)", c)]
        # odfiltruj ewidentnie nie-KL (opcjonalnie)
        kl_cols = candidates

    if args.target_kl or (kl_cols and target_col is None):
        if not kl_cols:
            print("[i] target KL wymuszony, ale nie znaleziono kolumn KL – przechodzę do heurystyk WOMAC/generic…")
        else:
            kl_series = _pick_kl_across_cols(df_raw, kl_cols, strategy=(args.kl_strategy or "max"))
            # Przytnij do 0..4 i ustaw NaN poza zakresem
            kl_series = kl_series.apply(lambda v: v if (pd.notna(v) and 0 <= v <= 4) else np.nan)

            # Ustaw score i grade
            df_raw["severity_score"] = kl_series.astype(float)
            # Etykiety tekstowe KL0..KL4
            df_raw["severity_grade"] = df_raw["severity_score"].apply(lambda v: f"KL{int(v)}" if pd.notna(v) else "missing")

            preproc_info.update({
                "target_strategy": "kellgren_lawrence",
                "target_col": "severity_grade",
                "kl_columns_used": kl_cols,
                "kl_strategy": args.kl_strategy or "max"
            })
            target_col = "severity_grade"

    # 5) Z czego liczyć severity_score?
    selected_cols = []
    if args.score_cols:
        selected_cols = [normalize_col(c) for c in args.score_cols.split(",")]
        selected_cols = [c for c in selected_cols if c in df_raw.columns]

    if args.score_patterns:
        pats = [p.strip() for p in args.score_patterns.split(",")]
        for p in pats:
            rx = re.compile(p)
            hits = [c for c in df_raw.columns if rx.search(c)]
            selected_cols.extend(hits)
        selected_cols = sorted(set(selected_cols))

    reverse_cols = []
    if args.reverse_cols:
        reverse_cols = [normalize_col(c) for c in args.reverse_cols.split(",")]
        reverse_cols = [c for c in reverse_cols if c in df_raw.columns]

    # Jeżeli nie mamy targetu gotowego i nie podano ręcznie, spróbuj WOMAC/generic
    if not selected_cols and target_col is None:
        womac_total = [c for c in df_raw.columns if "womac" in c and "total" in c]
        if womac_total:
            selected_cols = womac_total[:1]
        else:
            generic = [c for c in df_raw.columns if re.search(r"(?i)pain|stiff|function|physical|activity", c)]
            # weź max kilka najbardziej „pasujących”:
            selected_cols = generic

    # 6) Budowa severity_score – sumujemy znormalizowane składowe
    severity = None
    if selected_cols:
        parts = []
        for c in selected_cols:
            s = minmax01(df_raw[c])
            if c in reverse_cols:
                s = 1 - s
            parts.append(s)
        if parts:
            M = pd.concat(parts, axis=1)
            severity = M.sum(axis=1, min_count=1)  # min_count=1 => same NaN -> NaN
            df_raw = df_raw.assign(severity_score=severity)
            preproc_info.update({"target_strategy": "sum_normalized_components",
                                 "score_components": selected_cols,
                                 "reverse_components": reverse_cols})
    elif target_col is None:
        # fallback: pierwsza numeryczna
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise SystemExit("Brak sensownych kolumn do zbudowania severity_score.")
        df_raw = df_raw.assign(severity_score=df_raw[num_cols[0]].astype(float))
        preproc_info.update({"target_strategy": "fallback_first_numeric",
                             "score_components": [num_cols[0]]})

    # 7) Budowa severity_grade (tylko na nie-NaN)
    if target_col is None:
        s = df_raw["severity_score"].astype(float)
        mask = s.notna()
        if mask.any() and s[mask].nunique() >= 2:
            labels_series, edges, bin_labels = robust_binning(s[mask])
            df_raw.loc[mask, "severity_grade"] = labels_series
            df_raw["severity_grade"] = df_raw["severity_grade"].fillna("missing")
            preproc_info.update({"bin_edges": edges, "bin_labels": bin_labels})
            target_col = "severity_grade"
        else:
            df_raw["severity_grade"] = "missing"
            target_col = "severity_grade"
    else:
        preproc_info.update({"target_strategy": "existing_target", "target_col": target_col})
        if "severity_score" not in df_raw.columns:
            df_raw["severity_score"] = np.nan

        # --- 7b) Opcjonalne wnioskowanie KL0 z danych objawowych (WOMAC/generic) ---
    if args.infer_kl0:
        # tylko jeśli faktycznie nie ma KL0
        has_kl0 = df_raw["severity_grade"].astype(str).eq("KL0").any()
        if not has_kl0:
            # Kandydaci: tylko rekordy, które są dziś 'missing' (brak radiologii)
            cand_mask = df_raw["severity_grade"].astype(str).eq("missing")

            # Spróbuj WOMAC: zidentyfikuj kolumny WOMAC (pain/stiff/function)
            womac_cols = [c for c in df_raw.columns if re.search(r"(?i)womac.*(pain|stiff|function)", c)]
            womac_used = []
            low_symptom_mask = None

            if womac_cols:
                parts = []
                for c in womac_cols:
                    s = minmax01(df_raw[c])
                    parts.append(s)
                if parts:
                    M = pd.concat(parts, axis=1)
                    # "brak dolegliwości": wszystkie składowe blisko zera
                    low_symptom_mask = (M <= args.kl0_womac_thresh).all(axis=1)
                    womac_used = womac_cols

            assigned_kl0 = 0

            if low_symptom_mask is not None:
                to_kl0 = cand_mask & low_symptom_mask
                if to_kl0.any():
                    df_raw.loc[to_kl0, "severity_score"] = 0.0
                    df_raw.loc[to_kl0, "severity_grade"] = "KL0"
                    assigned_kl0 = int(to_kl0.sum())

            # Jeśli WOMAC-ów brak lub nic nie złapało — użyj dolnego kwantyla miękkiego score’u
            if assigned_kl0 == 0:
                # Zbuduj pomocniczy score objawowy, jeśli go nie ma:
                # (Uwaga: jeżeli target KL nadpisał 'severity_score' liczbą 0..4, to
                #  nie nadaje się do kwantyla objawów; zbudujemy osobny soft_score.)
                soft_cols = []
                if not womac_cols:
                    soft_cols = [c for c in df_raw.columns if re.search(r"(?i)womac|pain|stiff|function|physical|activity", c)]
                else:
                    soft_cols = womac_cols

                soft_score = None
                if soft_cols:
                    parts = []
                    for c in soft_cols:
                        parts.append(minmax01(df_raw[c]))
                    if parts:
                        Msoft = pd.concat(parts, axis=1)
                        # soft_score = Msoft.mean(axis=1, skipna=True)  # działa w starszych Pandas
                        valid_counts = Msoft.count(axis=1)
                        soft_score = Msoft.sum(axis=1, skipna=True) / valid_counts
                        soft_score[valid_counts == 0] = np.nan

                if soft_score is not None:
                    ss = soft_score[cand_mask].astype(float)
                    ss = ss.dropna()
                    if len(ss) > 0:
                        q = float(np.clip(args.kl0_quantile, 0.0, 0.5))
                        thr = ss.quantile(q)
                        to_kl0 = cand_mask & (soft_score <= thr)
                        if to_kl0.any():
                            df_raw.loc[to_kl0, "severity_score"] = 0.0
                            df_raw.loc[to_kl0, "severity_grade"] = "KL0"
                            assigned_kl0 = int(to_kl0.sum())

            # Zapisz diagnostykę do preproc_info
            preproc_info["kl0_inferred"] = True
            preproc_info["kl0_inferred_n"] = int(df_raw["severity_grade"].astype(str).eq("KL0").sum())
            preproc_info["kl0_from_womac_cols"] = womac_used
            preproc_info["kl0_womac_thresh"] = args.kl0_womac_thresh
            preproc_info["kl0_quantile"] = args.kl0_quantile
        else:
            preproc_info["kl0_inferred"] = False

        # --- 7c) Gwarancja minimalnej liczby KL0 (dorelabelowanie + opcjonalna synteza) ---
    if args.kl0_min and args.kl0_min > 0:
        def _build_soft_score(df_local):
            # Zbuduj „soft score” objawowy jak w 7b (średnia z cech objawowych po min-max)
            # 1) kandydaci WOMAC/pain/stiff/function/physical/activity
            soft_cols = [c for c in df_local.columns if re.search(r"(?i)womac|pain|stiff|function|physical|activity", c)]
            if not soft_cols:
                return pd.Series(np.nan, index=df_local.index)
            parts = []
            for c in soft_cols:
                parts.append(minmax01(df_local[c]))
            Msoft = pd.concat(parts, axis=1)
            valid_counts = Msoft.count(axis=1)
            soft = Msoft.sum(axis=1, skipna=True) / valid_counts
            soft[valid_counts == 0] = np.nan
            return soft

        # Policz obecne KL0
        current_kl0 = df_raw["severity_grade"].astype(str).eq("KL0")
        need = int(args.kl0_min) - int(current_kl0.sum())

        if need > 0:
            soft_score_all = _build_soft_score(df_raw)

            # 7c-1) Dorelabelowanie z 'missing' (najpierw to)
            cand_missing = df_raw["severity_grade"].astype(str).eq("missing")
            miss_scores = soft_score_all[cand_missing].dropna()
            if len(miss_scores) > 0 and need > 0:
                # najniższe soft-score => najbardziej „bezobjawowe”
                take_idx = miss_scores.sort_values().index[:need]
                if len(take_idx) > 0:
                    df_raw.loc[take_idx, "severity_grade"] = "KL0"
                    df_raw.loc[take_idx, "severity_score"] = 0.0
                    df_raw.loc[take_idx, "is_synthetic"] = 0
                    need -= len(take_idx)

            # 7c-2) Dorelabelowanie z dolnego ogona KL1 (opcjonalnie, bardzo oszczędnie)
            if need > 0 and args.kl0_pick_from_kl1_quantile > 0:
                cand_kl1 = df_raw["severity_grade"].astype(str).eq("KL1")
                kl1_scores = soft_score_all[cand_kl1].dropna()
                if len(kl1_scores) > 0:
                    q = float(np.clip(args.kl0_pick_from_kl1_quantile, 0.0, 0.5))
                    thr_kl1 = kl1_scores.quantile(q)
                    kl1_low = kl1_scores[kl1_scores <= thr_kl1].sort_values()
                    if len(kl1_low) > 0:
                        take_idx = kl1_low.index[:need]
                        if len(take_idx) > 0:
                            df_raw.loc[take_idx, "severity_grade"] = "KL0"
                            df_raw.loc[take_idx, "severity_score"] = 0.0
                            df_raw.loc[take_idx, "is_synthetic"] = 0
                            need -= len(take_idx)

            # 7c-3) Synteza (jeśli włączona i nadal brakuje)
            if need > 0 and args.kl0_synth_when_needed:
                # Pool do syntezy: rekordy KL0 (po dorelabelowaniu) oraz 'missing'/'KL1' z najniższym soft-score
                pool_mask = df_raw["severity_grade"].astype(str).isin(["KL0", "missing", "KL1"])
                pool = df_raw[pool_mask].copy()
                pool_soft = soft_score_all[pool_mask].fillna(pool_soft := soft_score_all.median())
                pool = pool.loc[pool_soft.sort_values().index]  # posortuj od najniższego soft-score

                if len(pool) > 0:
                    # Ustaw pomocnicze: które kolumny są numeryczne/kategoryczne
                    num_cols = [c for c in pool.columns if c not in ["severity_grade","severity_score"] and pd.api.types.is_numeric_dtype(pool[c])]
                    cat_cols = [c for c in pool.columns if c not in ["severity_grade","severity_score"] and not pd.api.types.is_numeric_dtype(pool[c])]

                    # Zakresy do jittera
                    if num_cols:
                        col_mins = pool[num_cols].min()
                        col_maxs = pool[num_cols].max()
                        col_ranges = (col_maxs - col_mins).replace(0, 1.0)

                    synth_rows = []
                    base_rows = pool.head(max(need, 1))  # bierzemy najniższe soft-score jako „prototypy”
                    jitter_scale = float(max(args.kl0_synth_jitter, 1e-4))

                    for i in range(need):
                        base = base_rows.iloc[i % len(base_rows)].copy()
                        new = base.copy()

                        # jitter dla numerycznych: N(0, (jitter*range)^2)
                        if num_cols:
                            noise = np.random.normal(loc=0.0, scale=(jitter_scale * col_ranges).values)
                            # ważne: trzymaj w granicach [min, max]
                            new[num_cols] = np.clip(new[num_cols].values + noise, col_mins.values, col_maxs.values)

                        # dla kategorycznych: zachowaj wartości bazowe (ew. można losować z rozkładu modalnego)
                        # etykiety:
                        new["severity_grade"] = "KL0"
                        new["severity_score"] = 0.0
                        new["is_synthetic"] = 1
                        synth_rows.append(new)

                    if synth_rows:
                        df_raw = pd.concat([df_raw, pd.DataFrame(synth_rows)], ignore_index=True)
                        need = 0  # zaspokojone

            # zapisz metadane
            preproc_info["kl0_min_requested"] = int(args.kl0_min)
            preproc_info["kl0_final_n"] = int(df_raw["severity_grade"].astype(str).eq("KL0").sum())
            preproc_info["kl0_synth_used"] = bool(args.kl0_synth_when_needed and need == 0)


    # 8) Opcjonalnie usuń wiersze bez targetu
    if args.drop_missing_target:
        before = len(df_raw)
        df_raw = df_raw[df_raw["severity_grade"].astype(str) != "missing"].copy()
        print(f"[i] Usunięto {before - len(df_raw)} wierszy z missing target")

    # 9) Usuwamy ew. ID, imputacja, porządek kolumn
    id_like = [c for c in df_raw.columns if re.search(r"(?:^|_)id(?:_|$)|patient_id|subject_id|record_id", c)]
    df_raw.drop(columns=id_like, inplace=True, errors="ignore")

    # imputacja: numeryczne -> mediana lub 0, kategoryczne -> 'missing'
    for c in df_raw.columns:
        if c == "severity_grade":
            df_raw[c] = df_raw[c].astype(str)
            continue
        if pd.api.types.is_numeric_dtype(df_raw[c]):
            med = pd.to_numeric(df_raw[c], errors="coerce").median()
            if pd.isna(med):
                df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0)
            else:
                df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(med)
        else:
            df_raw[c] = df_raw[c].astype(str).fillna("missing")

    # końcowe ułożenie kolumn
    if "severity_score" not in df_raw.columns:
        df_raw["severity_score"] = np.nan
    if "severity_grade" not in df_raw.columns:
        df_raw["severity_grade"] = "missing"

    cols = [c for c in df_raw.columns if c not in ["severity_score","severity_grade"]] + ["severity_score","severity_grade"]
    df_final = df_raw.loc[:, cols]

    # 10) Zapis
    df_final.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(preproc_info, f, ensure_ascii=False, indent=2)

    # 11) Diagnostyka
    s = pd.to_numeric(df_final["severity_score"], errors="coerce")
    y = df_final["severity_grade"].astype(str)
    print(f"[OK] Zapisano {out_csv}")
    print(f"[i] Info o preprocesingu: {out_json}")
    print(f"[i] severity_score: non-null={int(s.notna().sum())}/{len(s)}, unique(non-null)={int(s.dropna().nunique())}")
    print("[i] severity_grade rozkład:")
    print(y.value_counts(dropna=False))
    print("[i] Zapisano mapę nazw kolumn do artifacts/column_mapping.csv")

if __name__ == "__main__":
    main()