"""Compile Comprehensive Historical Submissions Registry.

Directly addresses User Request Task 2:
- Compiles public submissions table with exact filename, Ariel score, SHA256, base, flip block.
- Separates drafts/offline candidates from public submissions.
- Incorporates 37-40, 53-58, and all 5 fin_stack public submissions.
- Explicitly flags missing CSVs or scores without inventing assumptions.
"""
import os
import json
import hashlib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SUB_DIR = ROOT / "submissions"
REV_DIR = ROOT / "results" / "revision_submissions"

def get_file_sha256(path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    # Load s48
    s48_path = REV_DIR / "posterior_flips_20261027" / "s48.json"
    s48 = json.load(open(s48_path))

    # All known public submissions with confirmed scores from Ariel
    # Sources: assemble.py, submission_feedback_log.csv, round138_fin_stack_segment_samples_20261028.md, inventario_01_41.md, batch_41_44_direction_probe_20261025.md
    public_subs = [
        # Pre-numbered / Phase 1
        {"id_num": "01", "name": "submission_01_candidate_a.csv", "rel_path": "submission_round_20261017_2019/submission_01_candidate_a.csv", "score": 0.94433, "base": "previa", "flips": "desconocido", "status": "pública", "source": "submission_feedback_log.csv", "notes": "Falta CSV local"},
        {"id_num": "02", "name": "submission_02_probe_char_tpe_5flip.csv", "rel_path": "submission_round_20261017_2019/submission_02_probe_char_tpe_5flip.csv", "score": 0.94419, "base": "previa", "flips": "5 flips", "status": "pública", "source": "submission_feedback_log.csv", "notes": "Falta CSV local"},
        {"id_num": "03", "name": "submission_03_probe_char_lgb_5flip.csv", "rel_path": "submission_round_20261017_2019/submission_03_probe_char_lgb_5flip.csv", "score": 0.94447, "base": "previa", "flips": "5 flips", "status": "pública", "source": "submission_feedback_log.csv", "notes": "Falta CSV local"},
        {"id_num": "04", "name": "submission_04_probe_low_post_10flip.csv", "rel_path": None, "score": 0.94457, "base": "07", "flips": "10 flips", "status": "pública", "source": "assemble.py / flips_04_09.tsv", "notes": "Reconstruible via flips_04_09.tsv sobre base07"},
        {"id_num": "05", "name": "submission_05_probe_mid_post_10flip.csv", "rel_path": None, "score": 0.94447, "base": "07", "flips": "10 flips", "status": "pública", "source": "assemble.py / flips_04_09.tsv", "notes": "Reconstruible via flips_04_09.tsv sobre base07"},
        {"id_num": "06", "name": "submission_06_probe_high_post_10flip.csv", "rel_path": None, "score": 0.94457, "base": "07", "flips": "10 flips", "status": "pública", "source": "assemble.py / flips_04_09.tsv", "notes": "Reconstruible via flips_04_09.tsv sobre base07"},
        {"id_num": "07", "name": "base07_labels_1_to_30000.txt", "rel_path": "results/revision_submissions/base07_labels_1_to_30000.txt", "score": 0.94476, "base": "base", "flips": "0 (base)", "status": "pública", "source": "assemble.py / base07", "notes": "Base de cohorte 04-09"},
        {"id_num": "10", "name": "submission_10_consolidada_disjunta_7_a_9.csv", "rel_path": None, "score": 0.94500, "base": "07", "flips": "desconocido", "status": "pública", "source": "submission_feedback_log.csv / feedback_10", "notes": "Falta CSV local"},
        {"id_num": "11", "name": "submission_11", "rel_path": None, "score": 0.94519, "base": "15", "flips": "flips_11_14_27_35.tsv", "status": "pública", "source": "assemble.py", "notes": "Reconstruible via tsv"},
        {"id_num": "12", "name": "submission_12", "rel_path": None, "score": 0.94419, "base": "15", "flips": "flips_11_14_27_35.tsv", "status": "pública", "source": "assemble.py", "notes": "Reconstruible via tsv"},
        {"id_num": "13", "name": "submission_13", "rel_path": None, "score": 0.94428, "base": "15", "flips": "flips_11_14_27_35.tsv", "status": "pública", "source": "assemble.py", "notes": "Reconstruible via tsv"},
        {"id_num": "15", "name": "submission_15_top16_Q4_sobre_mejor11.csv", "rel_path": "submissions/20261019/submission_15_top16_Q4_sobre_mejor11.csv", "score": 0.94523, "base": "base", "flips": "0 (base)", "status": "pública", "source": "assemble.py / base15", "notes": "Base de cohorte 16-26"},
        {"id_num": "16", "name": "submission_16_frescos_consenso64_sobre15.csv", "rel_path": "submissions/20261020/submission_16_frescos_consenso64_sobre15.csv", "score": 0.94500, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "17", "name": "submission_17_informacion_cruzada_A_C_sobre15.csv", "rel_path": "submissions/20261020/submission_17_informacion_cruzada_A_C_sobre15.csv", "score": 0.94447, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "18", "name": "submission_18_informacion_cruzada_B_C_sobre15.csv", "rel_path": "submissions/20261020/submission_18_informacion_cruzada_B_C_sobre15.csv", "score": 0.94404, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "19", "name": "submission_19_informacion_cruzada_B_D_sobre15.csv", "rel_path": "submissions/20261020/submission_19_informacion_cruzada_B_D_sobre15.csv", "score": 0.94409, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "20", "name": "submission_20_informacion_cruzada_C_D_E_sobre15.csv", "rel_path": "submissions/20261020/submission_20_informacion_cruzada_C_D_E_sobre15.csv", "score": 0.94285, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "21", "name": "submission_21_A_E1_E2_sobre15.csv", "rel_path": "submissions/20261021/submission_21_A_E1_E2_sobre15.csv", "score": 0.94504, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "22", "name": "submission_22_A_E1_E3_sobre15.csv", "rel_path": "submissions/20261021/submission_22_A_E1_E3_sobre15.csv", "score": 0.94471, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "23", "name": "submission_23_A_E1_E4_sobre15.csv", "rel_path": "submissions/20261021/submission_23_A_E1_E4_sobre15.csv", "score": 0.94495, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "24", "name": "submission_24_control_A32_consolidada_sobre15.csv", "rel_path": "submissions/20261021/submission_24_control_A32_consolidada_sobre15.csv", "score": 0.94533, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "25", "name": "submission_25_A_E2_top16_catboost.csv", "rel_path": "submissions/20261021/submission_25_A_E2_top16_catboost.csv", "score": 0.94519, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "26", "name": "submission_26_A_E2_complemento16.csv", "rel_path": "submissions/20261021/submission_26_A_E2_complemento16.csv", "score": 0.94547, "base": "15", "flips": "flips_16_26.tsv", "status": "pública", "source": "assemble.py", "notes": "CSV verificado"},
        {"id_num": "27", "name": "submission_27", "rel_path": None, "score": 0.94228, "base": "33", "flips": "flips_11_14_27_35.tsv", "status": "pública", "source": "assemble.py", "notes": "Reconstruible via tsv"},
        {"id_num": "28", "name": "submission_28", "rel_path": None, "score": 0.94400, "base": "33", "flips": "flips_11_14_27_35.tsv", "status": "pública", "source": "assemble.py", "notes": "Reconstruible via tsv"},
        {"id_num": "29", "name": "submission_29", "rel_path": None, "score": 0.94571, "base": "33", "flips": "flips_11_14_27_35.tsv", "status": "pública", "source": "assemble.py", "notes": "Reconstruible via tsv"},
        {"id_num": "30", "name": "submission_30", "rel_path": None, "score": 0.94561, "base": "33", "flips": "flips_11_14_27_35.tsv", "status": "pública", "source": "assemble.py", "notes": "Reconstruible via tsv"},
        {"id_num": "31", "name": "submission_31", "rel_path": None, "score": 0.94585, "base": "33", "flips": "flips_11_14_27_35.tsv", "status": "pública", "source": "assemble.py", "notes": "Reconstruible via tsv"},
        {"id_num": "32", "name": "submission_32", "rel_path": None, "score": 0.94561, "base": "33", "flips": "flips_11_14_27_35.tsv", "status": "pública", "source": "assemble.py", "notes": "Reconstruible via tsv"},
        {"id_num": "33", "name": "base33_labels_1_to_30000.txt", "rel_path": "results/revision_submissions/base33_labels_1_to_30000.txt", "score": 0.94614, "base": "base", "flips": "0 (base)", "status": "pública", "source": "assemble.py / base33", "notes": "Base de cohorte 34-44"},
        {"id_num": "34", "name": "submission_34", "rel_path": None, "score": 0.94580, "base": "33", "flips": "flips_11_14_27_35.tsv", "status": "pública", "source": "assemble.py", "notes": "Reconstruible via tsv"},
        {"id_num": "35", "name": "submission_35", "rel_path": None, "score": 0.94609, "base": "33", "flips": "13 flips (8 new + 5 seen)", "status": "pública", "source": "assemble.py / flips_11_14_27_35.tsv", "notes": "8 new retest en 45"},
        {"id_num": "36", "name": "submission_36", "rel_path": None, "score": 0.94571, "base": "33", "flips": "18 flips", "status": "pública", "source": "assemble.py / flips_36_41.tsv", "notes": "Reconstruible via tsv"},
        {"id_num": "37", "name": "submission_37", "rel_path": None, "score": 0.94490, "base": "33", "flips": "146 flips (140 seen + 6 new)", "status": "pública", "source": "assemble.py / flips_36_41.tsv", "notes": "Reconstruible via tsv; 140 seen degradaron"},
        {"id_num": "38", "name": "submission_38", "rel_path": None, "score": 0.94423, "base": "33", "flips": "201 flips", "status": "pública", "source": "assemble.py / flips_36_41.tsv", "notes": "Reconstruible via tsv"},
        {"id_num": "39", "name": "submission_39", "rel_path": None, "score": 0.92976, "base": "33", "flips": "1228 flips masivos", "status": "pública", "source": "assemble.py / flips_36_41.tsv", "notes": "Caída masiva por flips agresivos"},
        {"id_num": "40", "name": "submission_40", "rel_path": None, "score": 0.94595, "base": "33", "flips": "4 flips", "status": "pública", "source": "assemble.py / flips_36_41.tsv", "notes": "Reconstruible via tsv"},
        {"id_num": "41", "name": "submission_41_consensus_6_vendedores_nuevos_sobre33.csv", "rel_path": None, "score": 0.94628, "base": "33", "flips": "6 flips (3 u->n, 3 n->u)", "status": "pública", "source": "assemble.py / batch_41_44", "notes": "SHA256 50a3f264d97adfc353382ea3e708c1f8e7012b2525b5ae5f73c26a2f309450b1"},
        {"id_num": "42", "name": "submission_42_fasttext_new_margin_ge005_13_sobre33.csv", "rel_path": None, "score": 0.94614, "base": "33", "flips": "13 flips u->n new", "status": "pública", "source": "assemble.py / fasttext manifest", "notes": "Empate con 33; SHA256 ab1cdfd90f6a..."},
        {"id_num": "43", "name": "submission_43_fasttext_new_full34_sobre33.csv", "rel_path": None, "score": 0.94614, "base": "33", "flips": "34 flips u->n new", "status": "pública", "source": "assemble.py / fasttext manifest", "notes": "Empate con 33; SHA256 8df48f148..."},
        {"id_num": "44", "name": "submission_44_consensus_new_sellers_used_to_new_3_sobre33.csv", "rel_path": None, "score": 0.94619, "base": "33", "flips": "3 flips u->n (de 41)", "status": "pública", "source": "assemble.py / batch_41_44", "notes": "SHA256 a1bfb711c49fcb77..."},
        {"id_num": "45", "name": "submission_45_gate8_vendedores_nuevos_sobre41.csv", "rel_path": None, "score": 0.94623, "base": "41", "flips": "8 flips (4 u->n, 4 n->u)", "status": "pública", "source": "assemble.py / sub 45 doc", "notes": "SHA256 4039ee585ad32a3a..."},
        {"id_num": "46", "name": "submission_46_consensus_7_vendedores_nuevos_sobre41.csv", "rel_path": None, "score": 0.94623, "base": "41", "flips": "7 flips (4 u->n, 3 n->u)", "status": "pública", "source": "assemble.py / sub 46-47 doc", "notes": "SHA256 a619ba9ec77e1ebf..."},
        {"id_num": "47", "name": "submission_47_consensus_4_used_to_new_sobre41.csv", "rel_path": None, "score": 0.94628, "base": "41", "flips": "4 flips u->n (de 46)", "status": "pública", "source": "assemble.py / sub 46-47 doc", "notes": "SHA256 e6ae2508ce1f56dc..."},
        {"id_num": "48", "name": "s48.json / submission_48_best_known", "rel_path": "results/revision_submissions/posterior_flips_20261027/s48.json", "score": 0.94633, "base": "41", "flips": "2 flips u->n (13583, 25426)", "status": "pública", "source": "assemble.py / s48.json", "notes": "BENCHMARK ACTUAL (.94633); SHA256 3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722"},
        {"id_num": "49", "name": "submission_49_exploration_11_flips_sobre48.csv", "rel_path": None, "score": 0.94628, "base": "48", "flips": "11 flips (5 u->n, 6 n->u)", "status": "pública", "source": "assemble.py / sub 49 doc", "notes": "SHA256 59a2bca7d01772ee..."},
        {"id_num": "50", "name": "submission_50_final_probe_5_flips_sobre48.csv", "rel_path": None, "score": 0.94628, "base": "48", "flips": "5 flips u->n (de 49)", "status": "pública", "source": "assemble.py / sub 50 doc", "notes": "SHA256 8b56d1d0f066bfe4..."},
        {"id_num": "57", "name": "submission_57_posterior_sobre48.csv", "rel_path": "submissions/20261027/submission_57_posterior_sobre48.csv", "score": 0.94633, "base": "48", "flips": "18 flips", "status": "pública", "source": "assemble.py / sub 57", "notes": "Empate con 48; SHA256 eba5b05a676..."},
        {"id_num": "58", "name": "submission_58_posterior_sobre48.csv", "rel_path": "submissions/20261027/submission_58_posterior_sobre48.csv", "score": 0.94633, "base": "48", "flips": "26 flips", "status": "pública", "source": "assemble.py / sub 58", "notes": "Empate con 48; SHA256 467dbb1f4fa..."},
        # 2026-10-28: fin_stack tests
        {"id_num": "59_full", "name": "submission_59_fin_stack_banda_0.85_1.00_sobre48.csv", "rel_path": "submissions/20261028/submission_59_fin_stack_banda_0.85_1.00_sobre48.csv", "score": 0.94576, "base": "48", "flips": "67 flips (41 vistos, 26 nuevos)", "status": "pública", "source": "round138 / Ariel", "notes": "SHA256 b34723587e1ef09c..."},
        {"id_num": "59_new", "name": "submission_59_subset_nuevos_26_sobre48.csv", "rel_path": None, "score": 0.94619, "base": "48", "flips": "26 flips nuevos", "status": "pública", "source": "round138 / Ariel", "notes": "Subset 26 nuevos de 59"},
        {"id_num": "59_seen_new", "name": "submission_59_subset_vistos_to_new_27_sobre48.csv", "rel_path": None, "score": 0.94590, "base": "48", "flips": "27 flips vistos->new", "status": "pública", "source": "round138 / Ariel", "notes": "Subset 27 vistos->new de 59"},
        {"id_num": "probe_fin_stack_new150", "name": "probe_fin_stack_new_unprobed150_sobre48.csv", "rel_path": None, "score": 0.94495, "base": "48", "flips": "150 flips nuevos (bandas 60-62)", "status": "pública", "source": "round138 / Ariel", "notes": "SHA256 ebb365337aa674d5..."},
        {"id_num": "probe_fin_stack_seen150", "name": "probe_fin_stack_seen_unprobed150_sobre48.csv", "rel_path": None, "score": 0.94257, "base": "48", "flips": "150 flips vistos (bandas 60-62)", "status": "pública", "source": "round138 / Ariel", "notes": "SHA256 37a8c519abff07ae..."}
    ]

    # Offline Drafts / Candidates (NEVER SUBMITTED / NO PUBLIC SCORE)
    draft_subs = [
        {"id_num": "08", "name": "submission_08_full_mid_posterior_31flip.csv", "score": "SIN SCORE", "base": "07", "flips": "31 flips", "status": "borrador offline", "source": "inventario_01_41.md", "notes": "Preparada, no enviada"},
        {"id_num": "09", "name": "submission_09_full_low_posterior_31flip.csv", "score": "SIN SCORE", "base": "07", "flips": "31 flips", "status": "borrador offline", "source": "inventario_01_41.md", "notes": "Preparada, no enviada"},
        {"id_num": "14", "name": "submission_14", "score": "SIN SCORE", "base": "15", "flips": "desconocido", "status": "borrador offline", "source": "inventario_01_41.md", "notes": "Preparada, no enviada"},
        {"id_num": "51", "name": "submission_51_seller_loo_seen_on_48.csv", "score": "SIN SCORE", "base": "48", "flips": "373 flips vistos", "status": "borrador offline", "source": "round94_97", "notes": "SHA256 03a19e21a29d... No enviada"},
        {"id_num": "52", "name": "submission_52_seller_smoothing_ensemble.csv", "score": "SIN SCORE", "base": "48", "flips": "flips vistos", "status": "borrador offline", "source": "round98_99", "notes": "No enviada"},
        {"id_num": "53", "name": "submission_53_seller_alpha1_seen_on_48.csv", "score": "SIN SCORE", "base": "48", "flips": "411 flips vistos", "status": "borrador offline", "source": "round105_111", "notes": "SHA256 a894af2ebabbe... No enviada"},
        {"id_num": "54", "name": "submission_54_draft", "score": "SIN SCORE", "base": "48", "flips": "desconocido", "status": "borrador offline", "source": "git log", "notes": "No enviada"},
        {"id_num": "55", "name": "submission_55_draft", "score": "SIN SCORE", "base": "48", "flips": "desconocido", "status": "borrador offline", "source": "git log", "notes": "No enviada"},
        {"id_num": "56", "name": "submission_56_draft", "score": "SIN SCORE", "base": "48", "flips": "desconocido", "status": "borrador offline", "source": "git log", "notes": "No enviada"},
        {"id_num": "60", "name": "submission_60_fin_stack_banda_0.70_0.85_sobre48.csv", "score": "SIN SCORE", "base": "48", "flips": "banda 0.70-0.85", "status": "borrador offline", "source": "submissions/20261028", "notes": "No enviada completa"},
        {"id_num": "61", "name": "submission_61_fin_stack_banda_0.60_0.70_sobre48.csv", "score": "SIN SCORE", "base": "48", "flips": "banda 0.60-0.70", "status": "borrador offline", "source": "submissions/20261028", "notes": "No enviada completa"},
        {"id_num": "62", "name": "submission_62_fin_stack_banda_0.50_0.60_sobre48.csv", "score": "SIN SCORE", "base": "48", "flips": "banda 0.50-0.60", "status": "borrador offline", "source": "submissions/20261028", "notes": "No enviada completa"},
        {"id_num": "63", "name": "submission_63_fin_stack_modelo_puro.csv", "score": "SIN SCORE", "base": "48", "flips": "1518 flips", "status": "borrador offline", "source": "submissions/20261028", "notes": "No enviada (bloqueada tras fallo de 59)"},
        {"id_num": "64", "name": "submission_64_dual_catboost_pure.csv", "score": "SIN SCORE", "base": "48", "flips": "1507 flips", "status": "borrador offline", "source": "submissions/20261029", "notes": "CatBoost puro no enviado"},
        {"id_num": "65", "name": "submission_65_dual_catboost_new_sellers_top_margin.csv", "score": "SIN SCORE", "base": "48", "flips": "17 flips", "status": "borrador offline", "source": "submissions/20261029", "notes": "Auditada en R150 (-5/+4 en tune/conf); no enviada"},
        {"id_num": "68_draft", "name": "submission_68_probe3_seen_sellers_to_used.csv", "score": "SIN SCORE", "base": "48", "flips": "7 flips vistos u->n", "status": "borrador descartado", "source": "submissions/20261029", "notes": "Gate de seller refutado en R141/R156 (-16/-10 netos)"},
        {"id_num": "68_alt_uncert", "name": "submission_68_probe3_uncertainty_flips.csv", "score": "SIN SCORE", "base": "48", "flips": "7 flips léxicos", "status": "borrador alternativo", "source": "submissions/20261029", "notes": "Versión alternativa de incertidumbre"},
        {"id_num": "70", "name": "submission_70_final_optimized.csv", "score": "SIN SCORE", "base": "48", "flips": "10 flips", "status": "borrador offline", "source": "submissions/20261029", "notes": "Derivada de R156 (+144 espurio); no enviada"},
        {"id_num": "71", "name": "submission_71_strong_ensemble_candidate.csv", "score": "SIN SCORE", "base": "48", "flips": "1338 flips", "status": "candidata no promovida", "source": "submissions/20261029", "notes": "R157 refit 70k; CI bootstrap cruza 0, inferior a deep base histórica"}
    ]

    # Compute actual SHA256 for all existing files in repo
    for row in public_subs:
        if row.get("rel_path"):
            p = ROOT / row["rel_path"]
            row["sha256"] = get_file_sha256(p)
        else:
            row["sha256"] = "CSV no guardado como archivo plano (reconstruible/en tsv)"

    for row in draft_subs:
        # Check potential paths
        found_p = None
        for base_dir in [SUB_DIR, REV_DIR]:
            matches = list(base_dir.rglob(row["name"]))
            if matches:
                found_p = matches[0]
                break
        if found_p and found_p.exists():
            row["sha256"] = get_file_sha256(found_p)
            row["found_path"] = str(found_p.relative_to(ROOT))
        else:
            row["sha256"] = "Archivo no encontrado en working tree"
            row["found_path"] = None

    df_pub = pd.DataFrame(public_subs)
    df_draft = pd.DataFrame(draft_subs)

    df_pub.to_csv(REV_DIR / "historical_public_submissions_complete_registry.csv", index=False)
    df_draft.to_csv(REV_DIR / "historical_draft_candidates_complete_registry.csv", index=False)

    print(f"Compiled {len(df_pub)} public submissions and {len(df_draft)} draft/offline submissions.")
    print(f"Saved registry CSVs to results/revision_submissions/")

if __name__ == "__main__":
    main()
