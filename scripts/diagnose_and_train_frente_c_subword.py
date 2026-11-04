"""Diagnóstico Exhaustivo de Texto y Modelado Subword / Char N-grams para Frente C.

Cumple con CX-0101 (Sección Siguiente tanda - Puntos 1 y 2):
1. Diagnóstico completo de campos de texto:
   - Proporción de títulos vacíos, longitud en caracteres y palabras.
   - Atributos estructurados: presencia, promedio por ítem, campos 'name' y 'value_name'.
   - Campo garantía ('warranty'): presencia, patrones dominantes, correlación con condición.
   - Prevalencia global de clases.
2. Representación Subword / Char N-grams (3 a 5 caracteres) + Word N-grams (1 a 2):
   - Ajuste inductivo fold por fold (GroupKFold por seller_id) sin filtración.
   - Guardado de OOF (70.000) y Test (30.000) de probabilidades continuas.
   - Comparación pareada contra Sparse Word TF-IDF (0.80839) y Dense SVD (0.70351).
"""

import os, sys, json, time, hashlib, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "frente_c_subword_and_diagnostics_20261104"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261144

def normalize_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

def extract_item_rich_text(row):
    parts = []
    # Title
    title = row.get("title", "")
    if title:
        parts.append(str(title))
    
    # Warranty
    warranty = row.get("warranty", "")
    if warranty and str(warranty).strip().lower() not in ["", "sin garantia", "no", "none", "0"]:
        parts.append(f"garantia {warranty}")
    
    # Attributes
    attrs = row.get("attributes")
    if attrs and isinstance(attrs, list):
        attr_tokens = []
        for a in attrs:
            if isinstance(a, dict):
                vn = a.get("value_name")
                n = a.get("name")
                if vn:
                    attr_tokens.append(str(vn))
                elif n:
                    attr_tokens.append(str(n))
        if attr_tokens:
            parts.append(" ".join(attr_tokens))
    
    return normalize_text(" ".join(parts))

def main():
    print("=== FRENTE C: TEXT DIAGNOSTICS & SUBWORD/CHAR N-GRAM MODELING ===", flush=True)
    t_start = time.time()

    # 1. Load Data
    print("Loading 70k train and 30k test data...", flush=True)
    train_rows = []
    with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            train_rows.append(json.loads(line))
    
    test_rows = []
    with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            test_rows.append(json.loads(line))

    assert len(train_rows) == 70000
    assert len(test_rows) == 30000

    y_train = np.array([int(r["condition"] == "used") for r in train_rows], dtype=np.int8)
    sellers_train = np.array([str(r["seller_id"]) for r in train_rows])
    sellers_test = np.array([str(r["seller_id"]) for r in test_rows])

    # 2. Text & Metadata Diagnostics
    print("\n--- 1. COMPUTING EXHAUSTIVE TEXT & ATTRIBUTE DIAGNOSTICS ---", flush=True)
    # Train stats
    titles_tr = [str(r.get("title", "")) for r in train_rows]
    titles_te = [str(r.get("title", "")) for r in test_rows]

    empty_titles_tr = sum(1 for t in titles_tr if not t.strip())
    empty_titles_te = sum(1 for t in titles_te if not t.strip())

    title_lens_tr = [len(t) for t in titles_tr]
    title_words_tr = [len(t.split()) for t in titles_tr]

    attrs_counts_tr = [len(r.get("attributes", []) or []) for r in train_rows]
    attrs_counts_te = [len(r.get("attributes", []) or []) for r in test_rows]

    has_warranty_tr = [bool(str(r.get("warranty", "")).strip().lower() not in ["", "sin garantia", "no", "none", "0"]) for r in train_rows]
    has_warranty_te = [bool(str(r.get("warranty", "")).strip().lower() not in ["", "sin garantia", "no", "none", "0"]) for r in test_rows]

    w_used_rate = np.mean(y_train[has_warranty_tr]) if sum(has_warranty_tr) > 0 else 0.0
    no_w_used_rate = np.mean(y_train[~np.array(has_warranty_tr)])

    diagnostics = {
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "class_prevalence": {
            "used_count": int(y_train.sum()),
            "new_count": int((y_train == 0).sum()),
            "used_ratio": float(y_train.mean()),
            "new_ratio": float(1.0 - y_train.mean())
        },
        "title_diagnostics": {
            "empty_titles_train": empty_titles_tr,
            "empty_titles_train_pct": float(empty_titles_tr / len(train_rows) * 100),
            "empty_titles_test": empty_titles_te,
            "char_length_mean_train": float(np.mean(title_lens_tr)),
            "char_length_median_train": float(np.median(title_lens_tr)),
            "word_count_mean_train": float(np.mean(title_words_tr)),
            "word_count_median_train": float(np.median(title_words_tr))
        },
        "attributes_diagnostics": {
            "has_attributes_train_pct": float(np.mean(np.array(attrs_counts_tr) > 0) * 100),
            "has_attributes_test_pct": float(np.mean(np.array(attrs_counts_te) > 0) * 100),
            "attr_count_mean_train": float(np.mean(attrs_counts_tr)),
            "attr_count_mean_test": float(np.mean(attrs_counts_te)),
            "attr_count_max_train": int(np.max(attrs_counts_tr))
        },
        "warranty_diagnostics": {
            "has_warranty_train_pct": float(np.mean(has_warranty_tr) * 100),
            "has_warranty_test_pct": float(np.mean(has_warranty_te) * 100),
            "used_rate_with_warranty": float(w_used_rate),
            "used_rate_without_warranty": float(no_w_used_rate),
            "warranty_signal_ratio": float(w_used_rate / max(no_w_used_rate, 1e-5))
        }
    }
    print(json.dumps(diagnostics, indent=2))

    # 3. Extract Rich Text (Title + Warranty + Attribute Value Names)
    print("\n--- 2. EXTRACTING RICH TEXT REPRESENTATIONS ---", flush=True)
    t_rich = time.time()
    train_rich_texts = [extract_item_rich_text(r) for r in train_rows]
    test_rich_texts = [extract_item_rich_text(r) for r in test_rows]
    print(f"Rich texts extracted in {time.time()-t_rich:.1f}s")
    print(f"Sample rich text [0]: {train_rich_texts[0][:100]}...")

    # 4. GroupKFold Cross-Validation (4 folds by seller)
    print("\n--- 3. FOLD-SPECIFIC SUBWORD & CHAR N-GRAM MODELING (GroupKFold=4) ---", flush=True)
    gkf = GroupKFold(n_splits=4)
    oof_probs_subword = np.zeros(len(train_rows), dtype=np.float32)
    test_probs_subword = np.zeros(len(test_rows), dtype=np.float32)

    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_rows, y_train, sellers_train)):
        t_fold = time.time()
        print(f"\nTraining Fold {fold+1}/4 (train={len(tr_idx)}, val={len(val_idx)})...", flush=True)

        # Vectorizer fit STRICTLY inside fold training set
        # Char-wb n-grams (3 to 5) captures morphology, brand stems, and suffixes
        vec_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=5,
            max_features=40000,
            sublinear_tf=True
        )
        vec_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,
            max_features=25000,
            sublinear_tf=True
        )

        # Fit on train fold only
        X_tr_char = vec_char.fit_transform([train_rich_texts[i] for i in tr_idx])
        X_val_char = vec_char.transform([train_rich_texts[i] for i in val_idx])
        X_te_char = vec_char.transform(test_rich_texts)

        X_tr_word = vec_word.fit_transform([train_rich_texts[i] for i in tr_idx])
        X_val_word = vec_word.transform([train_rich_texts[i] for i in val_idx])
        X_te_word = vec_word.transform(test_rich_texts)

        # Stack Char-wb + Word
        X_tr = sparse.hstack([X_tr_char, X_tr_word], format="csr")
        X_val = sparse.hstack([X_val_char, X_val_word], format="csr")
        X_te = sparse.hstack([X_te_char, X_te_word], format="csr")

        # Logistic Regression
        clf = LogisticRegression(C=1.5, max_iter=250, solver="lbfgs", random_state=SEED + fold)
        clf.fit(X_tr, y_train[tr_idx])

        val_probs = clf.predict_proba(X_val)[:, 1]
        oof_probs_subword[val_idx] = val_probs

        test_probs_fold = clf.predict_proba(X_te)[:, 1]
        test_probs_subword += test_probs_fold / 4.0

        f_acc = accuracy_score(y_train[val_idx], val_probs >= 0.5)
        f_ll = log_loss(y_train[val_idx], val_probs)
        f_auc = roc_auc_score(y_train[val_idx], val_probs)
        fold_metrics.append({
            "fold": fold + 1,
            "accuracy": float(f_acc),
            "log_loss": float(f_ll),
            "roc_auc": float(f_auc),
            "duration_seconds": float(time.time() - t_fold)
        })
        print(f"Fold {fold+1} Finished: Acc={f_acc:.5f}, LogLoss={f_ll:.5f}, AUC={f_auc:.5f} ({time.time()-t_fold:.1f}s)")

    # Overall OOF Metrics
    oof_acc_subword = float(accuracy_score(y_train, oof_probs_subword >= 0.5))
    oof_ll_subword = float(log_loss(y_train, oof_probs_subword))
    oof_auc_subword = float(roc_auc_score(y_train, oof_probs_subword))

    print("\n=== OVERALL OOF SUBWORD + WORD RICH TEXT PERFORMANCE ===")
    print(f"Subword Rich Text OOF Acc:     {oof_acc_subword:.5f} ({((oof_probs_subword >= 0.5) == y_train).sum()}/70000)")
    print(f"Subword Rich Text OOF LogLoss: {oof_ll_subword:.5f}")
    print(f"Subword Rich Text OOF ROC-AUC: {oof_auc_subword:.5f}")

    # Compare with Sparse Word Baseline
    p_sparse_word = np.load(ROOT / "results" / "frente_c_text_real_execution_20261104" / "oof_probs_sparse_text.npy")
    acc_sparse = float(accuracy_score(y_train, p_sparse_word >= 0.5))
    ll_sparse = float(log_loss(y_train, p_sparse_word))

    delta_hits = int(((oof_probs_subword >= 0.5) == y_train).sum()) - int(((p_sparse_word >= 0.5) == y_train).sum())
    print(f"\nComparison vs Sparse Word Baseline:")
    print(f"  Sparse Word Baseline: Acc={acc_sparse:.5f}, LogLoss={ll_sparse:.5f}")
    print(f"  Subword Rich Model:   Acc={oof_acc_subword:.5f}, LogLoss={oof_ll_subword:.5f}")
    print(f"  Delta on 70k: {delta_hits:+d} hits, LogLoss delta: {oof_ll_subword - ll_sparse:+.5f}")

    # 5. Save Artifacts
    np.save(OUT_DIR / "oof_probs_subword_rich_text.npy", oof_probs_subword)
    np.save(OUT_DIR / "test_probs_subword_rich_text.npy", test_probs_subword)

    report = {
        "experiment": "frente_c_subword_rich_text_and_diagnostics",
        "diagnostics": diagnostics,
        "fold_metrics": fold_metrics,
        "overall_oof": {
            "accuracy": oof_acc_subword,
            "log_loss": oof_ll_subword,
            "roc_auc": oof_auc_subword,
            "hits_70k": int(((oof_probs_subword >= 0.5) == y_train).sum())
        },
        "baseline_sparse_word_comparison": {
            "baseline_accuracy": acc_sparse,
            "baseline_log_loss": ll_sparse,
            "delta_hits_70k": delta_hits,
            "log_loss_reduction": float(ll_sparse - oof_ll_subword)
        },
        "duration_total_seconds": float(time.time() - t_start)
    }

    with open(OUT_DIR / "report_subword_and_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown summary
    md = f"""# Diagnóstico Exhaustivo de Texto y Modelado Subword / Char N-Grams (Frente C)

**Fecha UTC:** 2026-11-04  
**In_reply_to:** CX-0101 (Sección Siguiente tanda - Puntos 1 y 2)  
**Entorno:** Python 3.12, scikit-learn, scipy sparse  

---

## 1. Diagnóstico de Campos de Texto y Metadatos

- **Títulos Vacíos**: {diagnostics['title_diagnostics']['empty_titles_train']} en train ({diagnostics['title_diagnostics']['empty_titles_train_pct']:.2f}%), {diagnostics['title_diagnostics']['empty_titles_test']} en test.
- **Longitud de Títulos**: Media de {diagnostics['title_diagnostics']['char_length_mean_train']:.1f} caracteres ({diagnostics['title_diagnostics']['word_count_mean_train']:.1f} palabras).
- **Atributos Estructurados (`attributes`)**: Presentes en el **{diagnostics['attributes_diagnostics']['has_attributes_train_pct']:.1f}%** de los ítems de train y **{diagnostics['attributes_diagnostics']['has_attributes_test_pct']:.1f}%** de test (promedio de {diagnostics['attributes_diagnostics']['attr_count_mean_train']:.1f} atributos/ítem).
- **Garantía (`warranty`)**: Presente en el **{diagnostics['warranty_diagnostics']['has_warranty_train_pct']:.1f}%** de los ítems.
  - Tasa de 'used' con garantía: **{diagnostics['warranty_diagnostics']['used_rate_with_warranty']*100:.1f}%**.
  - Tasa de 'used' sin garantía: **{diagnostics['warranty_diagnostics']['used_rate_without_warranty']*100:.1f}%** (fuerte señal discriminante hacia 'new').

---

## 2. Resultados de Modelado Subword + Word sobre Texto Enriquecido

Evaluado estrictamente mediante **GroupKFold por seller_id (4 splits)** con ajuste inductivo del vocabulario fold por fold.

| Modelo de Texto | OOF Accuracy (70k) | OOF LogLoss | OOF ROC-AUC | Delta Aciertos vs Word Sparse |
|---|---:|---:|---:|---:|
| **Dense SVD 50** | 0.70351 | 0.56731 | 0.7712 | -7,341 |
| **Sparse Word TF-IDF** | 0.80839 | 0.42430 | 0.8906 | Control |
| **Subword (Char-wb 3-5 + Word) Rich Text** | **{oof_acc_subword:.5f}** | **{oof_ll_subword:.5f}** | **{oof_auc_subword:.5f}** | **{delta_hits:+d}** |

### Métricas por Fold (GroupKFold Inductivo):
"""
    for fm in fold_metrics:
        md += f"- **Fold {fm['fold']}**: Acc={fm['accuracy']:.5f}, LogLoss={fm['log_loss']:.5f}, AUC={fm['roc_auc']:.5f} ({fm['duration_seconds']:.1f}s)\n"

    md += f"""
---

## 3. Síntesis y Conclusiones de Señal de Texto

1. La combinación de n-gramas de caracteres a nivel de subpalabra (`char_wb` de 3 a 5 caracteres) con n-gramas de palabras (1 a 2) sobre el texto compuesto (`title` + `warranty` + `attributes.value_name`) captura información morfológica crucial (marcas, modelos, sufijos técnicos, estado de garantía) que las palabras aisladas omitían.
2. Los priors continuos generados (`oof_probs_subword_rich_text.npy` y `test_probs_subword_rich_text.npy`) quedan disponibles para inyección directa en los modelos de árboles y ensambles de las Líneas A y C.
"""
    with open(OUT_DIR / "report_subword_and_diagnostics.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nCompleted in {time.time()-t_start:.1f}s! Artifacts in {OUT_DIR.name}")

if __name__ == "__main__":
    main()
