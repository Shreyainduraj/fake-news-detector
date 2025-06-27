import pandas as pd

# Load both datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add label columns
fake["label"] = 0  # fake news
real["label"] = 1  # real news

# Combine and keep only needed columns
df = pd.concat([fake[["title", "text", "label"]], real[["title", "text", "label"]]], ignore_index=True)

# Shuffle the rows
df = df.sample(frac=1, random_state=42)

# Save as combined_news.csv
df.to_csv("combined_news.csv", index=False)

print("✅ combined_news.csv created successfully!")
