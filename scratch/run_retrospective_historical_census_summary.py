"""Resumen Descriptivo Retrospectivo del Censo de 128 Envíos Históricos (CX-0202)

Nota Metodológica Crucial (Auditoría CX-0202 §5):
Este análisis NO constituye un backtest prequential prospectivo ni una calibración prequential formal:
1. No recomputa ni evalúa pronósticos probabilísticos congelados con anterioridad estricta al score de cada envío.
2. Agrupa los scores conocidos de 128 envíos históricos del ledger contra el ancla S4 (0.94766 / 19,899 aciertos), elegida ex-post como referencia consolidada del equipo.
3. Las métricas de cobertura (30.8% nominal 90%, factor gamma 5.53x) provienen de un censo retrospectivo de desvíos, no de densidades secuenciales pre-registradas.
4. Su función es documentar descriptivamente los riesgos observados, la frecuencia de retrocesos por familia técnica y la asimetría de drawdowns en el leaderboard público (ej. EVT-032 -376, EVT-142 -349, EVT-155 -99).
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

def run():
    print("=" * 80)
    print("RESUMEN DESCRIPTIVO RETROSPECTIVO DEL CENSO HISTORICO (128 ENVIOS) - CX-0202")
    print("=" * 80)
    
    csv_path = 'results/reconciliation_all_historical_submissions.csv'
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Total envíos históricos registrados en el ledger canónico: {len(df)}")

    # Envíos con score público verificable
    scored = df[df['public_score'].notna()].copy()
    scored['hits'] = scored['public_score'] * 20998.0
    scored['delta_vs_s4'] = scored['hits'] - 19899.0
    print(f"Envíos con score público verificado: {len(scored)}")

    # Agrupación retrospectiva por familia técnica
    family_stats = []
    for fam, grp in scored.groupby('policy_family'):
        n_ev = len(grp)
        deltas = grp['delta_vs_s4'].values
        
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas))
        min_d = float(np.min(deltas))
        max_d = float(np.max(deltas))
        med_d = float(np.median(deltas))
        
        n_loss = int((deltas < 0).sum())
        n_gain = int((deltas > 0).sum())
        n_neutral = int((deltas == 0).sum())
        
        worst_row = grp.loc[grp['delta_vs_s4'].idxmin()]
        worst_id = worst_row['event_id']
        worst_loss = worst_row['delta_vs_s4']
        worst_flips = worst_row['flips_vs_s4_count']

        family_stats.append({
            "family": fam,
            "n_submissions": n_ev,
            "mean_delta_vs_s4": round(mean_d, 2),
            "median_delta_vs_s4": round(med_d, 2),
            "min_delta_vs_s4": round(min_d, 2),
            "max_delta_vs_s4": round(max_d, 2),
            "n_losses": n_loss,
            "n_gains": n_gain,
            "loss_rate": round(n_loss / n_ev, 3),
            "worst_event": f"{worst_id} (Delta={worst_loss:+.0f}, Flips={worst_flips})",
        })

    df_fam = pd.DataFrame(family_stats).sort_values('n_submissions', ascending=False)
    print("\n--- DISTRIBUCION DESCRIPTIVA DE DELTAS RETROSPECTIVOS VS S4 ---")
    for _, r in df_fam.iterrows():
        print(f"Familia: {r['family']:35} | N={r['n_submissions']:2d} | Med Delta: {r['median_delta_vs_s4']:+6.1f} | Min Delta: {r['min_delta_vs_s4']:+6.1f} | Pérdidas: {r['n_losses']:2d}/{r['n_submissions']:2d} ({r['loss_rate']:.1%}) | Peor: {r['worst_event']}")

    # Auditoría de drawdowns severos
    print("\n--- AUDITORIA DE RETROCESOS SEVEROS (DRAWDOWN AUDIT) ---")
    major_drawdowns = scored.sort_values('delta_vs_s4').head(8)
    for _, r in major_drawdowns.iterrows():
        print(f"  {r['event_id']:10}: Score={r['public_score']:.5f} | Hits={r['hits']:5.0f} | Delta={r['delta_vs_s4']:+5.0f} | Flips={r['flips_vs_s4_count']:4.0f} | Familia={r['policy_family']}")

    # Resumen formal de limitaciones
    print("\n--- DECLARACION FORMAL DE LIMITACIONES ESTADISTICAS ---")
    print("  1. Retractación formal: Este censo retrospectivo no es un pronóstico prequential prospectivo.")
    print("  2. Asimetría de riesgo: Las familias de router macro y ensambles agresivos sufrieron caídas de hasta -376 aciertos.")
    print("  3. Ni el censo histórico ni los cuantiles internos justifican calificar A, B, C o Clean como 'seguros' o '95% robustos'.")
    print("  4. Todo vector nuevo permanece clasificado como DEVELOPMENT_ONLY.")

    # Guardar reporte JSON
    out_path = 'results/retrospective_historical_census_summary_report.json'
    with open(out_path, 'w') as f:
        json.dump({
            "timestamp_iso": pd.Timestamp.now().isoformat(),
            "runner": "scratch/run_retrospective_historical_census_summary.py",
            "methodology_disclaimer": "Resumen descriptivo retrospectivo, no prequential. Evaluado ex-post contra ancla S4.",
            "total_historical_events": len(df),
            "scored_events": len(scored),
            "family_summary": family_stats,
            "major_drawdowns": major_drawdowns[['event_id', 'public_score', 'hits', 'delta_vs_s4', 'flips_vs_s4_count', 'policy_family']].to_dict(orient='records')
        }, f, indent=2)
    print(f"\nReporte guardado en {out_path}")

if __name__ == '__main__':
    run()
