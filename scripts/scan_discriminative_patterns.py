"""Comprehensive scan of discriminative patterns in Train and Test data.
Analyzes precision, log-odds, and frequency to engineer optimal feedback features.
"""

import re
import pandas as pd
import numpy as np

def main():
    print("Loading train and test data...", flush=True)
    train = pd.read_json("data/ap_2026_q_2/raw/train_data.jsonlines", lines=True)
    y_train = (train["condition"] == "used").astype(int).values
    titles = train["title"].str.lower().fillna("").tolist()

    test = pd.read_json("data/ap_2026_q_2/raw/test_data.jsonlines", lines=True)
    test_titles = test["title"].str.lower().fillna("").tolist()

    candidate_patterns = [
        # Strong USED candidates
        (r"\bcontrol remoto\b", "control_remoto"),
        (r"\bpara repuesto\b|\bpara despiece\b|\bno funciona\b|\bno enciende\b", "repuesto_despiece"),
        (r"\blote de\b|\blote x\b|\bcombo de\b", "lote_combo"),
        (r"\bcomo nuevo\b|\bcomo nueva\b", "como_nuevo"),
        (r"\bmuy buen estado\b|\bbuen estado\b", "buen_estado"),
        (r"\bexcelente estado\b", "excelente_estado"),
        (r"\bdetalles de uso\b|\bmarcas de uso\b", "detalles_uso"),
        (r"\bsegunda mano\b|\busado\b|\busada\b", "segunda_mano"),
        (r"\bantiguo\b|\bantigua\b|\bvitage\b|\bvintage\b|\bretro\b", "antiguo_vintage"),
        (r"\bde coleccion\b|\bcoleccionable\b", "coleccionable"),
        (r"\bcirculado\b|\bmatasellos\b|\bfilatelia\b", "filatelia_circulado"),
        (r"\bfaltan piezas\b|\bcompleto en caja\b", "completo_caja"),
        (r"\breparar\b|\ba reparar\b", "a_reparar"),
        (r"\bfunciona perfecto\b|\banda de diez\b|\bfunciona correctamente\b", "funciona_bien"),
        (r"\bimpecable\b|\bimpecables\b", "impecable_general"),
        (r"\bcanje\b|\bpermutas\b|\bpermuto\b", "permuto_canje"),
        (r"\bdesgaste\b|\brayas\b|\bdetalles\b", "desgaste_detalles"),

        # Strong NEW candidates
        (r"\ba estrenar\b|\bsin uso\b|\bnuevo sin uso\b", "a_estrenar"),
        (r"\b0km\b|\b0 km\b|\bcero kilometro\b", "cero_km"),
        (r"\ben caja cerrada\b|\bcerrado de fabrica\b|\bcaja sellada\b|\ben blister\b", "caja_cerrada_blister"),
        (r"\bsellado\b|\bsellada\b", "sellado"),
        (r"\bgarantia oficial\b|\bcon garantia escrita\b|\b12 meses de garantia\b", "garantia_oficial"),
        (r"\bfactura a o b\b|\bhacemos factura\b|\biva incluido\b", "factura_ab"),
        (r"\benvio gratis\b|\benvios a todo el pais\b", "envio_gratis"),
        (r"\boriginal nuevo\b|\bnuevo original\b", "nuevo_original"),
        (r"\bdummy\b|\bmaqueta\b", "dummy_maqueta"),
        (r"\bartesanal\b|\bhecho a mano\b", "artesanal_mano"),
        (r"\bimportado de usa\b|\bimportada\b", "importado"),
        (r"\bdistribuidor oficial\b|\blocales a la calle\b", "distribuidor_oficial"),
        (r"\bx mayor\b|\bpor mayor\b|\bmayorista\b", "mayorista"),
        (r"\bpersonalizado\b|\bpersonalizada\b|\ba pedido\b", "personalizado_pedido"),
        (r"\blisto para instalar\b|\blista para colgar\b", "listo_para_usar"),
    ]

    print(f"Pattern scan over {len(train)} train rows and {len(test)} test rows:\n")
    header = f"{'Pattern':<25} | {'Train Count':<11} | {'Train %Used':<11} | {'Log-Odds':<9} | {'Test Count':<10}"
    print(header)
    print("-" * len(header))

    results = []
    for pat, name in candidate_patterns:
        regex = re.compile(pat)
        tr_matches = np.array([bool(regex.search(t)) for t in titles])
        te_matches = np.array([bool(regex.search(t)) for t in test_titles])

        cnt_tr = int(tr_matches.sum())
        cnt_te = int(te_matches.sum())
        if cnt_tr > 0:
            p_used = float(y_train[tr_matches].mean())
            p_smooth = float((y_train[tr_matches].sum() + 1) / (cnt_tr + 2))
            log_odds = float(np.log(p_smooth / (1 - p_smooth)))
            results.append({
                "pattern": name,
                "train_count": cnt_tr,
                "train_pct_used": p_used * 100,
                "log_odds": log_odds,
                "test_count": cnt_te
            })
            print(f"{name:<25} | {cnt_tr:<11d} | {p_used*100:6.2f}%    | {log_odds:+7.2f} | {cnt_te:<10d}")

if __name__ == "__main__":
    main()
