t1 = open('src/revision_submissions/deep_catboost_nbsvm_ablation_20261029.py', encoding='utf-8').read()
t2 = open('src/revision_submissions/strong_ensemble_audit_and_refit_20261029.py', encoding='utf-8').read()

# Let's find TfidfVectorizer in both
import re
print("TfidfVectorizer in deep ablation:")
for m in re.finditer(r'TfidfVectorizer\([^)]+\)', t1):
    print(m.group(0))

print("\nTfidfVectorizer in strong ensemble:")
for m in re.finditer(r'TfidfVectorizer\([^)]+\)', t2):
    print(m.group(0))
