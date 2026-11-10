import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

anchor_df = pd.read_csv('submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv')
quad_df = pd.read_csv('results/sparse_text_seller_group_cv_20261110/test_probabilities_quad_titan.csv')
quad_df['ID'] = 1 + np.arange(len(quad_df))
merged = pd.merge(anchor_df, quad_df, on='ID')
dis = merged[merged['condition'] != merged['pred_label_quad_titan']].copy()

with open('data/ap_2026_q_2/raw/test_data.jsonlines', encoding='utf-8') as f:
    test_rows = [json.loads(line) for line in f]

re_0km = re.compile(r'\b0\s*km\b|\bzero\s*km\b|\bsin\s+rodar\b|\b0km\b', re.I)
re_sellado = re.compile(r'\bcaja\s+sellada\b|\bcerrad[oa]\s+de\s+fabrica\b|\bprecintad[oa]\b|\ben\s+blister\b', re.I)
re_repuesto = re.compile(r'\bpara\s+repuestos?\b|\bpara\s+desarme\b|\bno\s+funciona\b|\brot[oa]s?\b|\bcon\s+falla\b', re.I)
re_usado = re.compile(r'\busad[oa]s?\b|\bsegunda\s+mano\b', re.I)
re_antiguo = re.compile(r'\bantigu[oa]s?\b|\bvintage\b|\bretro\b', re.I)
re_impecable = re.compile(r'\bimpecable\b|\bexcelente\s+estado\b|\bcomo\s+nuev[oa]\b|\bmuy\s+buen\s+estado\b', re.I)
re_digital = re.compile(r'\blibro\s+digital\b|\bformato\s+digital\b|\bdescarga\s+digital\b|\bdigital\s+oferta\b|\bebook\b|\bpdf\b', re.I)

certified_rows = []

for idx, row in dis.iterrows():
    item_id = int(row['ID'])
    r = test_rows[item_id - 1]
    title = str(r.get('title') or '').strip()
    cat = str(r.get('category_id') or '')
    price = float(r.get('price') or 0.0)
    attrs = r.get('attributes') or []
    
    p_quad = float(row['p_used_quad_titan'])
    p_tree = float(row['p_used_tri_titan_trees'])
    p_txt = float(row['p_used_sparse_text'])
    curr = str(row['condition'])
    prop = str(row['pred_label_quad_titan'])
    
    # Check vehicle trap
    is_vehicle_trap = False
    for a in attrs:
        if isinstance(a, dict) and 'KILOMETERS' in str(a.get('id', '')):
            try:
                val = float(str(a.get('value_name', '0')).replace('km', '').replace('.', '').strip())
                if val > 100 and prop == 'new':
                    is_vehicle_trap = True
            except:
                pass
    if is_vehicle_trap:
        continue

    # Check pet stud trap
    if ('servicio' in title.lower() or 'monta' in title.lower() or 'stud' in title.lower()) and ('perro' in title.lower() or 'caniche' in title.lower() or 'bulldog' in title.lower()):
        if prop == 'used':
            continue

    tree_text_agree = (p_tree >= 0.5 and p_txt >= 0.5) or (p_tree < 0.5 and p_txt < 0.5)
    margin = abs(p_quad - 0.5)

    lex_used = bool(re_repuesto.search(title) or re_usado.search(title) or re_antiguo.search(title) or re_impecable.search(title))
    lex_new = bool(re_0km.search(title) or re_sellado.search(title) or re_digital.search(title))

    is_certified = False
    reason = ''

    if prop == 'used' and curr == 'new':
        if lex_used and p_quad > 0.75 and tree_text_agree:
            is_certified = True
            reason = f'Lexical cue used + Quad p_used={p_quad:.3f} + Tree/Text consensus'
        elif margin > 0.40 and tree_text_agree and abs(p_tree - p_txt) < 0.10:
            is_certified = True
            reason = f'Ultra-high margin used ({p_quad:.3f}) + tight Tree/Text consensus ({abs(p_tree - p_txt):.3f})'

    elif prop == 'new' and curr == 'used':
        if lex_new and p_quad < 0.25 and tree_text_agree:
            is_certified = True
            reason = f'Lexical cue new + Quad p_new={1-p_quad:.3f} + Tree/Text consensus'
        elif margin > 0.40 and tree_text_agree and abs(p_tree - p_txt) < 0.10:
            is_certified = True
            reason = f'Ultra-high margin new ({1-p_quad:.3f}) + tight Tree/Text consensus ({abs(p_tree - p_txt):.3f})'

    if is_certified:
        certified_rows.append({
            'ID': item_id,
            'title': title,
            'category_id': cat,
            'price': price,
            'anchor_evt143_condition': curr,
            'certified_condition': prop,
            'p_used_quad_titan': p_quad,
            'p_used_tri_titan_trees': p_tree,
            'p_used_sparse_text': p_txt,
            'tree_text_discrepancy': abs(p_tree - p_txt),
            'margin': margin,
            'justification': reason,
        })

df_cert = pd.DataFrame(certified_rows).sort_values(by='margin', ascending=False)
out_csv = Path('results/certified_quad_titan_flips_20261110.csv')
df_cert.to_csv(out_csv, index=False)
print(f'Wrote {len(df_cert)} certified flips to {out_csv}')
