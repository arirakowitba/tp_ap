import pickle
import pandas as pd
from collections import Counter

def main():
    print("=== Inspecting Auto Spares & Parts ===")
    with open("data/ap_2026_q_2/raw/tr.pkl", "rb") as f:
        df_tr = pickle.load(f)

    # Filter by auto parts categories
    # Common auto parts categories: MLA1744, MLA24723, MLA60606, MLA370638, etc.
    titles = df_tr["title"].fillna("").astype(str).str.lower()
    auto_mask = titles.str.contains(r"\brepuesto\b|\bdespiece\b|\bautoparte\b|\boriginal\b|\bfiat\b|\bford\b|\bchevrolet\b|\brenault\b|\bvolkswagen\b|\bpeugeot\b|\btoyota\b", regex=True)
    print(f"Total items with auto keywords: {auto_mask.sum()}")
    
    auto_df = df_tr[auto_mask]
    print(f"Condition breakdown:")
    print(auto_df["condition"].value_counts())

    # Words for NEW vs USED in auto
    new_titles = auto_df[auto_df["condition"] == "new"]["title"].str.lower().str.split()
    used_titles = auto_df[auto_df["condition"] == "used"]["title"].str.lower().str.split()

    new_words = Counter(w for t in new_titles for w in t if len(w) > 3)
    used_words = Counter(w for t in used_titles for w in t if len(w) > 3)

    print("\nTop distinctive words for NEW in auto:")
    for w, c in new_words.most_common(20):
        u_c = used_words.get(w, 0)
        ratio = c / (c + u_c + 1)
        if ratio > 0.7 and c > 20:
            print(f"  '{w}': new={c}, used={u_c} (p_new={ratio:.2f})")

    print("\nTop distinctive words for USED in auto:")
    for w, c in used_words.most_common(20):
        n_c = new_words.get(w, 0)
        ratio = c / (c + n_c + 1)
        if ratio > 0.7 and c > 20:
            print(f"  '{w}': used={c}, new={n_c} (p_used={ratio:.2f})")

if __name__ == "__main__":
    main()
