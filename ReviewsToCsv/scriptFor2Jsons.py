#!/usr/bin/env python3
import json
import pandas as pd
from dateutil import parser

# ——— CONFIG ———
JSON_PATHS = [
    "/Users/jawadali/Desktop/rndsProjects/GoogleCloud/RagAda/ReviewsToCsv/FGAmsterdam93.json",
    "/Users/jawadali/Desktop/rndsProjects/GoogleCloud/RagAda/ReviewsToCsv/FGBerlin94.json"
]
CSV_PATH  = "/Users/jawadali/Desktop/rndsProjects/GoogleCloud/RagAda/ReviewsToCsv/FinalReviews.csv"
# ————————

def load_reviews(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("data", {}).get("results", [])

def transform_review(r):
    review_id   = r.get("id")
    google_id   = r.get("google_review_id") or ""
    third_src   = r.get("thirdPartyReviewSourcesId")
    location_id = r.get("locationId") or r.get("location", {}).get("id")

    # Build rag_id
    short_gid = google_id[:8] if google_id else ""
    rag_id = f"REV_{review_id}_G_{short_gid}" if short_gid else f"REV_{review_id}"

    # Normalize date
    raw_date = r.get("date")
    if raw_date:
        dt = parser.isoparse(raw_date)
        dt_utc = dt.astimezone(tz=dt.tzinfo)
        date_str = dt_utc.isoformat().replace("T", " ")
    else:
        date_str = ""

    return {
        "rag_id":                    rag_id,
        "id":                        review_id,
        "date":                      date_str,
        "reviewerTitle":             r.get("reviewerTitle", ""),
        "ratingValue":               r.get("ratingValue"),
        "ratingText":                r.get("ratingText") or "",
        "reviewReply":               r.get("reviewReply") or "",
        "reviewReplyType":           r.get("reviewReplyType") or "",
        "sentimentAnalysis":         r.get("sentimentAnalysis") or "",
        "satisfactoryLevel":         r.get("satisfactoryLevel"),
        "locationId":                location_id,
        "google_review_id":          google_id,
        "thirdPartyReviewSourcesId": third_src,
        "url":                       r.get("url") or "",
        "full_review_json":          json.dumps(r, ensure_ascii=False)
    }

def main():
    all_reviews = []
    for path in JSON_PATHS:
        reviews = load_reviews(path)
        print(f"Loaded {len(reviews)} from {path}")
        all_reviews.extend(reviews)

    records = [transform_review(r) for r in all_reviews]
    df = pd.DataFrame(records, columns=[
        "rag_id",
        "id",
        "date",
        "reviewerTitle",
        "ratingValue",
        "ratingText",
        "reviewReply",
        "reviewReplyType",
        "sentimentAnalysis",
        "satisfactoryLevel",
        "locationId",
        "google_review_id",
        "thirdPartyReviewSourcesId",
        "url",
        "full_review_json"
    ])

    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {len(df)} total reviews → {CSV_PATH}")

if __name__ == "__main__":
    main()
