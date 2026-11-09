with open('scripts/train_high_capacity_lgbm_5fold_20261109.py', 'r', encoding='utf-8') as f:
    code = f.read()

feedback_code = '''
    # 12b. Deconvolved Pattern Feedback Features
    feedback_cues = {
        "cue_impecable_nuevo": (r"\\b(impecable sin uso|nuevo sellado|nuevo sin uso)\\b", -3.5),
        "cue_a_estrenar": (r"\\b(a estrenar)\\b", -3.0),
        "cue_artesanal": (r"\\b(artesanal|hech[oa] a mano|souvenirs?|cotill[oó]n|centro de mesa)\\b", -2.0),
        "cue_sellado": (r"\\b(sellad[oa]s?|en blister|cerrad[oa] en caja|caja cerrada)\\b", -1.9),
        "cue_0km": (r"\\b(0km|0 km)\\b", -1.5),
        "cue_disfraz": (r"\\b(disfraz|disfraces)\\b", -1.6),
        "cue_amiibo": (r"\\b(amiibo)\\b", -2.0),
        "cue_dummy": (r"\\b(dummy|ficticio)\\b", -3.0),
        "cue_detalles_uso": (r"\\b(detalles? de uso|leves detalles|buen estado|muy buen estado|impecable estado)\\b", 3.4),
        "cue_segunda_mano": (r"\\b(segunda mano|usad[oa]s?)\\b", 2.5),
        "cue_lote_revistas": (r"\\b(lote de|colecci[oó]n de|revistas?|fasc[ií]culos?)\\b", 2.2),
        "cue_antiguo_vintage": (r"\\b(antigu[oa]s?|antiguedades|vintage|retro)\\b", 1.7),
        "cue_desarme": (r"\\b(desarme|despiece|repuestos?|para repuesto)\\b", 0.5),
    }

    full_texts = [f"{t.lower()} {w.lower()}" for t, w in zip(titles, warranties)]
    composite_score = np.zeros(N, dtype=np.float32)

    for cname, (cpat, cweight) in feedback_cues.items():
        creg = re.compile(cpat, re.IGNORECASE)
        cmask = np.array([1.0 if creg.search(ft) else 0.0 for ft in full_texts], dtype=np.float32)
        feats[cname] = cmask
        composite_score += cmask * cweight

    feats["cue_composite_score"] = composite_score
    feats["cue_antiguo_x_nowarranty"] = feats["cue_antiguo_vintage"] * feats["warranty_null"]
    feats["cue_sellado_x_logprice"] = feats["cue_sellado"] * feats["log_price"]
    feats["cue_a_estrenar_x_logprice"] = feats["cue_a_estrenar"] * feats["log_price"]
'''

target_anchor = '    # 13. Categorical metadata frame'
assert target_anchor in code, 'Anchor not found'
new_code = code.replace(target_anchor, feedback_code + '\n' + target_anchor)
new_code = new_code.replace('results/high_capacity_lgbm_5fold', 'results/high_capacity_lgbm_feedback_5fold')
new_code = new_code.replace('results\\\\high_capacity_lgbm_5fold', 'results\\\\high_capacity_lgbm_feedback_5fold')

with open('scripts/train_high_capacity_lgbm_feedback_5fold_20261109.py', 'w', encoding='utf-8') as f:
    f.write(new_code)
print('Successfully generated scripts/train_high_capacity_lgbm_feedback_5fold_20261109.py!')
