"""CL-0036 text/attribute model T (no external data). Fit on DATA/train_data.jsonlines, predict DATA/test_data.jsonlines.
Sources: title (word 1-2 + char_wb 2-5), warranty (word 1-3), attributes+tags+sub_status tokens, explicit Spanish condition lexicon
with negation, light numeric meta (n_pictures, n_variations, has_warranty). Logistic regression (liblinear), C given.
usage: python text_model.py DATA OUT.npy C [sources: title,warranty,attr,lex,meta]"""
import sys, re, json, unicodedata, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
D, OUT, C = sys.argv[1], sys.argv[2], float(sys.argv[3]); SRC = (sys.argv[4] if len(sys.argv) > 4 else 'title,warranty,attr,lex,meta').split(',')
def norm(s):
    s = '' if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = unicodedata.normalize('NFKD', s.lower()); s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9ñ ]', ' ', s)).strip()
LEX = {'nuevo': r'\bnuev[oa]s?\b', 'usado': r'\busad[oa]s?\b', 'sin_uso': r'\bsin (uso|usar|estrenar)\b', 'nunca_usado': r'\bnunca (usad|se uso)', 'no_usado': r'\bno (usad|se uso)',
       'como_nuevo': r'\bcomo nuev', 'impecable': r'\bimpecabl', 'sellado': r'\bsellad', 'en_caja': r'\ben (su )?caja\b|\bcaja cerrada\b', 'original': r'\boriginal', 'repuesto': r'\brepuesto',
       'accesorio': r'\baccesori', 'antiguo': r'\bantigu|\bvintage\b|\bretro\b|\bcoleccion', 'garantia': r'\bgarantia\b', 'oferta': r'\boferta|\bpromo', 'importado': r'\bimportad',
       'excelente_estado': r'\bexcelente estado|\bmuy buen estado|\bbuen estado', 'detalle': r'\bdetalle|\bfalla|\broto|\brayad', 'libro': r'\blibro|\brevista', 'lote': r'\blote\b'}
def frame(path):
    df = pd.read_json(path, lines=True)
    t = df.title.map(norm); w = df.warranty.map(norm)
    a = df.attributes.map(lambda x: ' '.join(f"{norm(d.get('id',''))}_{norm(d.get('value_name',''))}".replace(' ', '_') for d in (x or [])))
    tg = df.tags.map(lambda x: ' '.join('tag_' + str(v) for v in (x or []))) + ' ' + df.sub_status.map(lambda x: ' '.join('sub_' + str(v) for v in (x or [])))
    full = t + ' ' + w
    lex = np.column_stack([full.str.contains(p, regex=True).values for p in LEX.values()]).astype(float)
    meta = np.column_stack([np.log1p(df.pictures.map(len)), np.log1p(df.variations.map(len)), (w.str.len() > 0).astype(float), np.log1p(t.str.len())])
    return df, t, w, a + ' ' + tg, lex, meta
tr, ttr, wtr, atr, ltr, mtr = frame(f'{D}/train_data.jsonlines'); te, tte, wte, ate, lte, mte = frame(f'{D}/test_data.jsonlines')
y = (tr.condition == 'used').astype(int).values; Xtr, Xte = [], []
def add(vec, a, b): Xtr.append(vec.fit_transform(a)); Xte.append(vec.transform(b))
if 'title' in SRC:
    add(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), ttr, tte)
    add(TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=300000, sublinear_tf=True), ttr, tte)
if 'warranty' in SRC: add(TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True), wtr, wte)
if 'attr' in SRC: add(TfidfVectorizer(token_pattern=r'\S+', min_df=2), atr, ate)
if 'lex' in SRC: Xtr.append(sp.csr_matrix(ltr)); Xte.append(sp.csr_matrix(lte))
if 'meta' in SRC:
    mu, sd = mtr.mean(0), mtr.std(0) + 1e-9; Xtr.append(sp.csr_matrix((mtr - mu) / sd)); Xte.append(sp.csr_matrix((mte - mu) / sd))
Xtr, Xte = sp.hstack(Xtr).tocsr(), sp.hstack(Xte).tocsr()
m = LogisticRegression(C=C, solver='liblinear', max_iter=2000).fit(Xtr, y); p = m.predict_proba(Xte)[:, 1]
np.save(OUT, p); print(json.dumps(dict(out=OUT, C=C, src=SRC, n_features=int(Xtr.shape[1]), n_train=len(y), mean_p=float(p.mean()))))
