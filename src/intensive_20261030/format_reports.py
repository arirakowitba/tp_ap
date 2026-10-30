"""Formats markdown reports for Agent 2, Agent 3, and Agent 6 from their verified JSON summaries."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "intensive_20261030"


def format_agent2_report():
    p = OUT_DIR / "agent2_mainline_summary.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    lines = [
        "# Agente 2 — Línea Principal CatBoost + LightGBM, Separación de Umbral y Diversidad",
        "",
        f"**Fecha:** {d['timestamp']}  ",
        f"**Tiempo total de ejecución:** {d['total_runtime_seconds']}s  ",
        f"**Configuración de referencia congelada:** `{d['frozen_reference_config']}`  ",
        "",
        "> [!IMPORTANT]",
        f"> {d['non_independence_notice']}",
        "",
        "---",
        "",
        "## 1. Separación Explícita: Efecto del Modelo vs. Efecto del Umbral en los Mismos IDs",
        "",
    ]

    for s, sd in d["seeds"].items():
        lines.append(f"### Semilla `{s}` (`N_confirm = {sd['n_confirm']:,}` | `Seen = {sd['n_seen']:,}` | `New = {sd['n_new']:,}`)")
        lines.append(f"- **Archivo pareado por ID:** `{sd['paired_file']}` (SHA256 `{sd['paired_file_sha256']}`)")
        lines.append(
            f"- **Baseline `A_hist`:** `th_tune = {sd['baseline_A_hist']['th_tune']:.3f}` | "
            f"`Acc@0.500 = {sd['baseline_A_hist']['acc_050']:.5f}` | "
            f"`Acc@th_tune = {sd['baseline_A_hist']['acc_tuned']:.5f}` | "
            f"`LL = {sd['baseline_A_hist']['logloss']:.5f}`"
        )
        lines.append(
            f"- **Variante `A_valname`:** `th_tune = {sd['baseline_A_valname']['th_tune']:.3f}` | "
            f"`Acc@0.500 = {sd['baseline_A_valname']['acc_050']:.5f}` | "
            f"`Acc@th_tune = {sd['baseline_A_valname']['acc_tuned']:.5f}` | "
            f"`LL = {sd['baseline_A_valname']['logloss']:.5f}`"
        )
        lines.append("")
        lines.append("| Comparación | Régimen de Umbral | Th Base | Th Cand | Acc Base | Acc Cand | Fixes | Breaks | Neto | IC 95% Vendedor | Seen Net | New Net | u->n Net | n->u Net | Delta LL |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        all_c = list(sd["comparisons_vs_A_hist"].items()) + list(sd["comparisons_vs_A_valname"].items())
        for cname, cdict in all_c:
            th_b, th_c = cdict["th_base_tune"], cdict["th_cand_tune"]
            regimes = [
                ("common_th_050", "Común 0.500", 0.500, 0.500),
                ("common_th_base", "Común th_base", th_b, th_b),
                ("policy_tuned_th", "Política tune", th_b, th_c),
            ]
            for rname, rlabel, tb, tc in regimes:
                m = cdict[rname]
                ci = m["bootstrap_seller_ci95"]
                u2n = m["directions"]["used_to_new"]["net"]
                n2u = m["directions"]["new_to_used"]["net"]
                lines.append(
                    f"| `{cname}` | {rlabel} | {tb:.3f} | {tc:.3f} | "
                    f"{m['accuracy_base']:.5f} | {m['accuracy_cand']:.5f} | "
                    f"{m['fixes']} | {m['breaks']} | **{m['net']:+d}** | "
                    f"[{ci[0]:+.1f}, {ci[1]:+.1f}] | {m['seen_sellers']['net']:+d} | "
                    f"{m['new_sellers']['net']:+d} | {u2n:+d} | {n2u:+d} | `{m['delta_logloss']:+.5f}` |"
                )
        lines.append("")

    tr = d["inductive_70k_test_refit"]
    lines.extend([
        "---",
        "",
        "## 2. Refit Estrictamente Inductivo sobre 70k e Inferencia en 30k Test",
        "",
        f"- **Archivo de predicciones y diagnósticos 30k:** `{tr['test_diag_path']}` (SHA256 `{tr['test_diag_sha256']}`)",
        f"- **Tiempo de refit 70k:** `{tr['refit_time_seconds']}s` | **Polaridad verificada:** `{tr['polarity_verified']}` (`N = {tr['n_test_rows']:,}`, `Seen = {tr['test_seen_rows']:,}`, `New = {tr['test_new_rows']:,}`)",
        f"- **Tasa media `P(used=1)` en 30k test:** `CB_single = {tr['mean_p_used_cb_single']:.5f}` | `Blend075_single = {tr['mean_p_used_blend_single']:.5f}` | `Blend075_Bag3 = {tr['mean_p_used_blend_bag3']:.5f}`",
        f"- **Acuerdo vs S48 (`th=0.500`):** `CB_single = {tr['agreement_vs_s48_at_050_cb_single']*100:.3f}%` | `Blend075_single = {tr['agreement_vs_s48_at_050_blend_single']*100:.3f}%` (`{tr['n_flips_vs_s48_at_050_blend_single']}` flips) | `Blend075_Bag3 = {tr['agreement_vs_s48_at_050_blend_bag3']*100:.3f}%` (`{tr['n_flips_vs_s48_at_050_blend_bag3']}` flips)",
        "",
        "## 3. Comparación Documentada con Familias Históricas (`HistGB+FastText` R39/R52/R131 y `R142/R156`)",
        "",
        "- **En los mismos IDs de `confirm` (Semillas `2026` y `314159`):**",
        "  - `R142_HistGB` cae **-380 ítems** en Semilla 2026 (`Acc = 0.89446` vs `0.92391`) y **-371 ítems** en Semilla 314159 (`Acc = 0.89130` vs `0.91996`) frente a `A_hist`.",
        "  - `R142_CatBoost` (sin features profundas ni `txt_sparse_p`) cae **-360 ítems** (Semilla 2026) y **-368 ítems** (Semilla 314159) frente a `A_hist`.",
        "  - `R156_Ensemble` (que omitió `txt_sparse_p` y 9 features granulares) cae **-113 ítems** en Semilla 2026 (`Acc = 0.91515` vs `0.92391`) y **-119 ítems** en Semilla 314159 (`Acc = 0.91069` vs `0.91996`) frente a `A_hist`.",
        "- **Comparación con `HistGB + FastText` (R39/R52/R131):** Aquella línea operó sobre una partición histórica distinta sin `paired_predictions` por ID en los cortes R151 (`~0.914` CV), pero su brecha frente a la matriz profunda con `txt_sparse_p` + `CatBoost/LightGBM` (`0.924–0.931` confirm) replica exactamente la caída observada en `R142_HistGB`.",
    ])

    (OUT_DIR / "agent2_mainline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote agent2_mainline_report.md")


def format_agent3_report():
    p = OUT_DIR / "agent3_external_summary.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    lines = [
        "# Agente 3 — Validación Externa al Proceso de Selección y Transporte a la Mezcla de Test",
        "",
        f"**Fecha:** {d['timestamp']}  ",
        f"**Tiempo total de ejecución:** {d['total_runtime_seconds']}s  ",
        f"**Clasificación Epistemológica:** `{d['epistemological_classification']}`  ",
        "",
        "> [!WARNING]",
        f"> {d['independence_limitation']}",
        "",
        "---",
        "",
        "## 1. Tamaños, Hashes y Proporciones `Seen / New` de los Cortes Externos Estrictamente Inductivos",
        "",
        "| Corte Externo | Semilla | `N_fit` | `N_tune` | `N_confirm` | `Seen` Confirm | `New` Confirm | SHA256 Split | SHA256 Paired CSV |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cname, c in d["cuts"].items():
        sz = c["sizes"]
        hs = c["hashes"]
        lines.append(
            f"| **`{cname}`** | `{c['seed']}` | {sz['fit']:,} | {sz['tune']:,} | {sz['confirm']:,} | "
            f"{sz['confirm_seen']:,} ({sz['confirm_seen_pct']}%) | {sz['confirm_new']:,} ({sz['confirm_new_pct']}%) | "
            f"`{hs['split_sha256'][:12]}...` | `{hs['paired_csv_sha256'][:12]}...` |"
        )

    lines.extend([
        "",
        "## 2. Resultados Observados vs. Estimación Transportada a la Mezcla `Seen/New` de Test (`71.623% Seen / 28.377% New`)",
        "",
        "| Corte Externo | Régimen Umbral | Acc Base | Acc Blend | Fixes | Breaks | Neto Obs | IC 95% Vendedor Obs | Seen Net | New Net | $\\Delta$ Acc Transportado | Neto Esperado 30k Test | IC 95% Transportado 30k | $\\Delta$ Log Loss |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for cname, c in d["cuts"].items():
        for rkey, rlabel in [("common_threshold_050", "Común 0.500"), ("policy_tuned_threshold", "Política tune")]:
            pm = c[rkey]["paired_metrics"]
            tr = c[rkey]["transported_test_mixture"]
            ci_obs = pm["bootstrap_seller_ci95"]
            ci_tr = tr["transported_30k_bootstrap_ci95"]
            lines.append(
                f"| **`{cname}`** | {rlabel} | {pm['accuracy_base']:.5f} | {pm['accuracy_cand']:.5f} | "
                f"{pm['fixes']} | {pm['breaks']} | **{pm['net']:+d}** | [{ci_obs[0]:+.1f}, {ci_obs[1]:+.1f}] | "
                f"{pm['seen_sellers']['net']:+d} | {pm['new_sellers']['net']:+d} | "
                f"`{tr['delta_acc_transported_to_test_mix']:+.5f}` | **{tr['expected_net_items_30k_test_mix']:+.1f}** | "
                f"[{ci_tr[0]:+.1f}, {ci_tr[1]:+.1f}] | `{pm['delta_logloss']:+.5f}` |"
            )

    lines.extend([
        "",
        "## 3. Supuestos Explícitos de Transportabilidad",
        "",
    ])
    for a in d["transportability_assumptions"]:
        lines.append(f"- {a}")

    lines.extend([
        "",
        "## 4. Análisis de Regresiones por Banda de Margen (`|p_base - 0.50|`)",
        "",
        "| Corte Externo | Banda de Margen `|p_base - 0.50|` | Filas | Fixes | Breaks | Neto |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for cname, c in d["cuts"].items():
        for b in c["regression_analysis_at_050"]["by_margin_band"]:
            lines.append(
                f"| `{cname}` | `{b['margin_band']}` | {b['n_rows']:,} | {b['fixes']} | {b['breaks']} | **{b['net']:+d}** |"
            )

    (OUT_DIR / "agent3_external_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote agent3_external_validation_report.md")


if __name__ == "__main__":
    format_agent2_report()
    format_agent3_report()
