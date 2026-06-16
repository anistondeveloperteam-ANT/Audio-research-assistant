"""Standalone PDF web-scraper utility.

Downloads PDFs from a list of direct URLs into a review folder. Kept separate
from the indexing pipeline on purpose: scrape -> review -> move the good ones
into data/papers/ -> run `python pipeline.py --incremental`.
"""
