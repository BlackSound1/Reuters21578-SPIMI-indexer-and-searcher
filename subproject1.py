import time
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple

from bs4 import Tag

from utilities import RunMode, process_document, save_to_file, create_pairs, create_index, compute_doc_stats, get_texts


def SPIMI(ALL_TEXTS: list) -> timedelta:
    """
    Implement a SPIMI-style indexer.

    :param ALL_TEXTS: The list of documents to index
    :return: How long the process took, according to `full_timing`
    """

    print('\n---------- SPIMI Indexer ----------')

    print("\nCreating inverted index...")
    index = defaultdict(list)

    tick = time.perf_counter()  # Start timing
    tock = None  # Variable for when timing ends

    # Go through each text in the corpus and update the index based on it
    for text in ALL_TEXTS:
        # Find the docID for this document
        DOC_ID = int(text.attrs['newid'])

        # Create list of tokens without removing duplicates. Will use duplicates to compute term frequencies
        tokens = process_document(text, duplicates=True)

        # For each token in this document, add it's token to the index if it doesn't exist and add this
        # (docID, term frequency) to its postings list as a tuple. If it does exist, simply update its postings list
        for token in tokens:
            tf = tokens.count(token)  # Get the term frequency for this term in the tokens

            # Append (docID, term frequency) to the postings list for this token
            index[token] += [(DOC_ID, tf)]

            # After processing 10,000th term, stop timing
            if len(index) == 10_000:
                tock = time.perf_counter()

    # Remove duplicates in postings lists
    index = {k: list(set(v)) for k, v in index.items()}

    # Sort the index by term
    index = dict(sorted(index.items()))

    # Sort each postings list
    for term in index:
        index[term] = sorted(index[term])

    # Save results to file
    save_to_file(index, mode=RunMode.SPIMI)

    return timedelta(seconds=(tock - tick))


def naive(ALL_TEXTS: list) -> timedelta:
    """
    Recreate the Project 2 Subproject 1 indexer.

    :param ALL_TEXTS: The list of documents to index
    :return: How long the process took, according to `full_timing`
    """

    print('\n---------- Naive Indexer ----------')

    # Create a list of (term, docID) pairs
    F: List[Tuple] = []

    print(f"\nCreating (term, docID) pairs for all articles...")

    # Go through each text in the corpus and create (term, docID) pairs, and add them to the existing list
    for text in ALL_TEXTS:
        # Find the docID for this document
        DOC_ID = int(text.attrs['newid'])

        # Create list of tokens
        tokens = process_document(text)

        # Create (term, docID) pairs from those tokens, and add to existing list
        F.extend(create_pairs(tokens, DOC_ID))

    # Sort the list of tuples by term
    F = sorted(F)

    # Create an index for the list of (term, docID) pairs
    print("\nCreating inverted index...")
    index, duration = create_index(F)

    # Save results to file
    save_to_file(index, mode=RunMode.NAIVE)

    return duration


def main():
    # Get all reuters objects in the corpus
    print('\n---------- Getting Articles ----------')
    ALL_TEXTS: List[Tag] = get_texts()

    # Compute the stats for all documents, necessary for BM25
    print('\n---------- Computing Statistics for all Articles ----------')
    compute_doc_stats(ALL_TEXTS)

    duration_n = naive(ALL_TEXTS)
    duration_s = SPIMI(ALL_TEXTS)

    print('\n---------- Timing Results ----------')

    print(f"\nNaive: Time taken to index first 10,000 terms: {duration_n}")
    print(f"\nSPIMI: Time taken to index first 10,000 terms: {duration_s}")

    diff = -((duration_s - duration_n) / duration_n) * 100
    print(f"\nThere is a time difference of {diff:.2f}%")


if __name__ == '__main__':
    main()
