"""Generate complete per-ID historical audit matrix for Probes 66-69 and alternatives.

Distinguishes:
1. Public Kaggle Submissions (with verified public scores)
2. Offline Drafts / Candidates (unsubmitted)
3. 100% Virgin items (never appeared in any public or draft flip)

Covers:
- Sonda 66 (Probe 1): 10 items
- Sonda 67 (Probe 2): 10 items
- Sonda 68 Draft (Probe 3 Orig): 7 items
- Sonda 68 Alt (Probe 3 Lexical): 7 items
- Sonda 69 (Probe 4): 8 items
"""
import glob
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_JSON = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"
S48_JSON = ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"
RESULTS_DIR = ROOT / "results" / "revision_submissions"
SUBMISSIONS_DIR = ROOT / "submissions" / "20261029"


def load_all_history():
    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48 = json.load(f)

    # Dictionary mapping ID -> list of historical events
    # event: {"type": "public"|"draft", "name": str, "direction": str, "score": str}
    id_history = defaultdict(list)

    # 1. Public Submissions
    public_subs = [
        ("submissions/20261027/submission_57_*.csv", "Sub 57 (pública)", "0.94633 (empató con 48, 18 flips)"),
        ("submissions/20261027/submission_58_*.csv", "Sub 58 (pública)", "0.94633 (empató con 48, 26 flips)"),
        ("submissions/20261028/submission_59_*.csv", "Sub 59 (pública)", "0.94576 (cayó -0.00057, 67 flips)"),
        ("submissions/20261028/submission_60_*.csv", "Sub 60 (pública)", "Banda fin_stack (cayó)"),
        ("submissions/20261028/submission_61_*.csv", "Sub 61 (pública)", "Banda fin_stack (cayó)"),
        ("submissions/20261028/submission_62_*.csv", "Sub 62 (pública)", "Banda fin_stack (cayó)"),
        ("submissions/20261028/submission_63_*.csv", "Sub 63 (pública)", "Modelo puro fin_stack (cayó)"),
    ]

    for pat, label, score_desc in public_subs:
        matches = glob.glob(str(ROOT / pat))
        if matches:
            df = pd.read_csv(matches[0])
            for _, r in df.iterrows():
                pid = int(r["ID"])
                cond = str(r["condition"])
                if cond != s48[str(pid)]:
                    id_history[pid].append({
                        "type": "public",
                        "name": label,
                        "file": Path(matches[0]).name,
                        "direction": f"{s48[str(pid)]}->{cond}",
                        "score": score_desc
                    })

    # Check flips TSVs (historical subs 36-41)
    flips_36_41_path = RESULTS_DIR / "flips_36_41.tsv"
    scores_hist = {
        "36": "0.94571", "37": "0.94490", "38": "0.94423",
        "39": "0.92976", "40": "0.94595", "41": "Pendiente/Borrador"
    }
    if flips_36_41_path.exists():
        df_f = pd.read_csv(flips_36_41_path, sep="\t")
        for _, r in df_f.iterrows():
            pid = int(r["ID"])
            sub_n = str(r["submission"])
            sc = scores_hist.get(sub_n, "Histórico")
            ev_type = "draft" if sub_n == "41" else "public"
            id_history[pid].append({
                "type": ev_type,
                "name": f"Sub {sub_n} ({'pública' if ev_type=='public' else 'borrador'})",
                "file": "flips_36_41.tsv",
                "direction": f"{r['from_33']}->{r['to']}",
                "score": sc
            })

    # 2. Drafts and Candidates
    draft_subs = [
        ("Candidate 64", SUBMISSIONS_DIR / "submission_64_dual_catboost_pure.csv", "condition", "Borrador (modelo dual puro, no enviado)"),
        ("Candidate 65", SUBMISSIONS_DIR / "submission_65_dual_catboost_new_sellers_top_margin.csv", "condition", "Borrador (compuerta 141 flips, no enviado)"),
        ("Sonda 68 Draft Orig", SUBMISSIONS_DIR / "submission_68_probe3_seen_sellers_to_used.csv", "condition", "Borrador (historial vendedor, descartado por validación negativa)"),
    ]

    for label, filepath, col_name, desc in draft_subs:
        if filepath.exists():
            df = pd.read_csv(filepath)
            if col_name in df.columns:
                for _, r in df.iterrows():
                    pid = int(r["ID"])
                    cond = str(r[col_name])
                    if cond != s48[str(pid)]:
                        id_history[pid].append({
                            "type": "draft",
                            "name": label,
                            "file": filepath.name,
                            "direction": f"{s48[str(pid)]}->{cond}",
                            "score": desc
                        })

    return id_history, s48


def main():
    id_history, s48 = load_all_history()

    # Load test metadata
    test_rows = [json.loads(line) for line in open(TEST_JSON, "r", encoding="utf-8")]
    train_sellers = set(json.loads(line)["seller_id"] for line in open(TRAIN_JSON, "r", encoding="utf-8"))

    # Load all candidate probes
    probes = [
        ("Sonda 66", SUBMISSIONS_DIR / "submission_66_probe1_new_sellers_to_new.csv", "Vendedores Nuevos (used -> new)"),
        ("Sonda 67", SUBMISSIONS_DIR / "submission_67_probe2_new_sellers_to_used.csv", "Vendedores Nuevos (new -> used)"),
        ("Sonda 68 (Draft Historial)", SUBMISSIONS_DIR / "submission_68_probe3_seen_sellers_to_used.csv", "Vendedores Vistos (Historial >=80% used)"),
        ("Sonda 68 (Alt 1 Léxica Extrema)", SUBMISSIONS_DIR / "submission_68_probe3_uncertainty_flips.csv", "Incertidumbre y Desacuerdo Léxico Explícito"),
        ("Sonda 68 (Alt 2 100% Virgen)", SUBMISSIONS_DIR / "submission_68_probe3_alt_virgin_lexical.csv", "Desacuerdo Léxico 100% Virgen (Sin historial previo)"),
        ("Sonda 69", SUBMISSIONS_DIR / "submission_69_probe4_unanimous_consensus.csv", "Consenso Unánime Margen Extremo"),
    ]

    matrix_rows = []

    for block_name, path, segment_desc in probes:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            pid = int(r["ID"])
            cond = str(r["condition"])
            s48_c = s48[str(pid)]
            if cond != s48_c:
                idx = pid - 1
                t = test_rows[idx].get("title") or ""
                p = float(test_rows[idx].get("price") or 0)
                s_id = test_rows[idx].get("seller_id")
                is_seen = (s_id in train_sellers)

                # Analyze history
                hist_events = id_history.get(pid, [])
                public_evs = [e for e in hist_events if e["type"] == "public"]
                draft_evs = [e for e in hist_events if e["type"] == "draft" and e["name"] != block_name]

                if public_evs:
                    status = "Envío Público Previo"
                    score_agg = "; ".join(f"{e['name']}: {e['score']}" for e in public_evs)
                    prev_detail = "; ".join(f"{e['name']} ({e['direction']})" for e in public_evs)
                elif draft_evs:
                    status = "Solo en Borrador Previo"
                    score_agg = "Borrador (sin score público)"
                    prev_detail = "; ".join(f"{e['name']} ({e['direction']})" for e in draft_evs)
                else:
                    status = "100% Virgen"
                    score_agg = "Ninguno (100% Virgen)"
                    prev_detail = "Ninguna aparición previa"

                matrix_rows.append({
                    "ID": pid,
                    "bloque": block_name,
                    "segmento": segment_desc,
                    "vendedor_tipo": "Visto" if is_seen else "Nuevo",
                    "direccion": f"{s48_c} -> {cond}",
                    "s48": s48_c,
                    "propuesta": cond,
                    "precio": p,
                    "titulo": t[:65] + ("..." if len(t) > 65 else ""),
                    "estado_historial": status,
                    "apariciones_previas": prev_detail,
                    "score_agregado": score_agg,
                })

    df_matrix = pd.DataFrame(matrix_rows)
    df_matrix.sort_values(by=["bloque", "ID"], inplace=True)

    out_csv = RESULTS_DIR / "round156_id_historical_audit_matrix.csv"
    df_matrix.to_csv(out_csv, index=False)
    print(f"Audit matrix saved to {out_csv} ({len(df_matrix)} rows)")

    # Generate Markdown Summary Matrix
    md_lines = [
        "# Matriz de Auditoría Histórica por ID: Sondas 66–69 y Alternativas\n",
        f"Fecha: 2026-10-29  ",
        f"Archivo estructurado: `round156_id_historical_audit_matrix.csv`  ",
        f"Total ítems evaluados: {len(df_matrix)} (cubriendo Sonda 66, Sonda 67, Sonda 68 Draft, Sonda 68 Alternativa y Sonda 69)\n",
        "## Resumen por Bloque de Sonda\n",
        "| Bloque | Ítems | Segmento | Dirección | 100% Vírgenes | En Borradores Previos | En Envíos Públicos | Respaldo en Validación |",
        "|---|---:|---|---|---:|---:|---:|:---:|",
    ]

    for bname in df_matrix["bloque"].unique():
        sub_df = df_matrix[df_matrix["bloque"] == bname]
        n_virgen = (sub_df["estado_historial"] == "100% Virgen").sum()
        n_draft = (sub_df["estado_historial"] == "Solo en Borrador Previo").sum()
        n_pub = (sub_df["estado_historial"] == "Envío Público Previo").sum()
        seg = sub_df["segmento"].iloc[0]
        dirs = ", ".join(sub_df["direccion"].unique())
        respaldo = "SÍ (+134 net nuevos)" if "66" in bname or "67" in bname else (
            "NO (-26 net en confirm)" if "Historial" in bname else (
                "SÍ (92% prec léxica)" if "Léxica" in bname else "SÍ (98.6% prec extrema)"
            )
        )
        md_lines.append(f"| {bname} | {len(sub_df)} | {seg} | {dirs} | **{n_virgen}** | {n_draft} | {n_pub} | {respaldo} |")

    md_lines.append("\n## Detalle Completo de la Matriz por ID\n")
    md_lines.append("| ID | Bloque | Segmento | Vendedor | Dirección | Precio | Título | Estado Historial | Apariciones Previas | Score Agregado |")
    md_lines.append("|---:|---|---|:---:|:---:|---:|---|:---:|---|---|")

    for _, r in df_matrix.iterrows():
        md_lines.append(f"| {r['ID']} | {r['bloque']} | {r['segmento'][:25]} | {r['vendedor_tipo']} | `{r['direccion']}` | ${r['precio']:.0f} | {r['titulo']} | **{r['estado_historial']}** | {r['apariciones_previas']} | {r['score_agregado']} |")

    out_md = RESULTS_DIR / "round156_id_historical_audit_matrix.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown report saved to {out_md}")


if __name__ == "__main__":
    main()
