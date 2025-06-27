#!/usr/bin/env python3
import csv

# ——— CONFIG ———
CSV_PATH = "/Users/jawadali/Desktop/rndsProjects/GoogleCloud/RagAda/ReviewsToCsv/FinalReviews.csv"
# ————————

def count_reviews(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # skip header
        next(reader, None)
        # count real rows
        return sum(1 for _ in reader)

def main():
    try:
        total = count_reviews(CSV_PATH)
        print(f"Total reviews: {total}")
    except FileNotFoundError:
        print(f"Error: File not found: {CSV_PATH}")
    except Exception as e:
        print(f"Error reading '{CSV_PATH}': {e}")

if __name__ == "__main__":
    main()
