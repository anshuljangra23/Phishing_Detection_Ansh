import pandas as pd
from difflib import SequenceMatcher


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "data/processed/clean_urls.csv"
)

target = "https://www.rezoni.com/"

target_clean = target.lower().strip()


# =====================================================
# SIMILARITY
# =====================================================

def similarity(a, b):
    return SequenceMatcher(
        None,
        a.lower(),
        b.lower()
    ).ratio()


# =====================================================
# CALCULATE SIMILARITY
# =====================================================

results = []

for _, row in df.iterrows():

    domain = str(row["domain"])

    score = similarity(
        target_clean,
        domain
    )

    results.append({
        "domain": domain,
        "label": int(row["label"]),
        "similarity": score
    })


result_df = pd.DataFrame(results)


# =====================================================
# SORT
# =====================================================

result_df = result_df.sort_values(
    "similarity",
    ascending=False
)


# =====================================================
# DISPLAY
# =====================================================

print("=" * 70)
print("MOST SIMILAR URLS TO REZONI")
print("=" * 70)

print("\nTarget:")
print(target)

print("\nTop 30 similar URLs:\n")

print(
    result_df.head(30).to_string(
        index=False
    )
)


# =====================================================
# LABEL SUMMARY
# =====================================================

top100 = result_df.head(100)

print("\n" + "=" * 70)
print("TOP 100 LABEL DISTRIBUTION")
print("=" * 70)

print(
    top100["label"].value_counts()
)

print("\n0 = LEGITIMATE")
print("1 = PHISHING")