import pickle
import pandas as pd
from collections import Counter

def main():
    print("=== Inspecting Numismatics & Collectibles (MLA31838) ===")
    with open("data/ap_2026_q_2/raw/tr.pkl", "rb") as f:
        df_tr = pickle.load(f)

    # Search for categories matching coins, medals, stamps, collectibles
    cat_counts = df_tr["category_id"].value_counts()
    
    # Check titles containing moneda, billete, numismatica, conmemorativa
    titles = df_tr["title"].fillna("").astype(str).str.lower()
    coin_mask = titles.str.contains(r"\bmoneda\b|\bbillete\b|\bnumismatica\b|\bconmemorativa\b|\bproof\b|\bblister\b", regex=True)
    print(f"Total items with coin/banknote keywords: {coin_mask.sum()}")
    
    coin_df = df_tr[coin_mask]
    print(f"Condition breakdown:")
    print(coin_df["condition"].value_counts())
    print("\nTop categories for coin/banknote items:")
    print(coin_df["category_id"].value_counts().head(10))

    # Inspect words associated with NEW vs USED in this subset
    new_titles = coin_df[coin_df["condition"] == "new"]["title"].str.lower().str.split()
    used_titles = coin_df[coin_df["condition"] == "used"]["title"].str.lower().str.split()

    new_words = Counter(w for t in new_titles for w in t if len(w) > 2)
    used_words = Counter(w for t in used_titles for w in t if len(w) > 2)

    print("\nTop distinctive words for NEW in coins:")
    for w, c in new_words.most_common(20):
        u_c = used_words.get(w, 0)
        ratio = c / (c + u_c + 1)
        if ratio > 0.6 and c > 5:
            print(f"  '{w}': new={c}, used={u_c} (p_new={ratio:.2f})")

    print("\nTop distinctive words for USED in coins:")
    for w, c in used_words.most_common(20):
        n_c = new_words.get(w, 0)
        ratio = c / (c + n_c + 1)
        if ratio > 0.6 and c > 5:
            print(f"  '{w}': used={c}, new={n_c} (p_used={ratio:.2f})")

if __name__ == "__main__":
    main()
