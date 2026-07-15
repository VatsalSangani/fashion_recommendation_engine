import pandas as pd
import json
from pathlib import Path

print("Loading articles with descriptions...")

# Load only the columns we need from the raw Kaggle dataset
articles = pd.read_csv(
    "data/articles.csv", 
    dtype={"article_id": str}, 
    usecols=["article_id", "prod_name", "colour_group_name", "detail_desc"]
)

metadata = {}
for _, row in articles.iterrows():
    # Keep the leading zero!
    art_id = str(row["article_id"]).zfill(10) 
    
    # Handle any items that might have missing descriptions
    desc = str(row["detail_desc"]) if pd.notna(row["detail_desc"]) else "No description available."
    
    # Map the ID to the human-readable text
    metadata[art_id] = {
        "name": str(row["prod_name"]),
        "color": str(row["colour_group_name"]),
        "description": desc
    }

# Save it to your processed folder so the API can read it later
output_path = Path("processed") / "article_metadata.json"
with open(output_path, "w") as f:
    json.dump(metadata, f)
    
print(f"✅ Saved metadata with descriptions to {output_path}!")