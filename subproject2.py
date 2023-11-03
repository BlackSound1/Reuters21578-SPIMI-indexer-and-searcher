import json
from math import log

naive: dict
SPIMI: dict


def single(query: str) -> None:
    """
    Performs a single-term search for the given query on the naive and SPIMI indexes. Compares results.

    :param query: The single-term query to search for
    """

    # OPERATE ON NAIVE INDEX

    # Search the naive index for the single-term query
    naive_postings = naive[query]

    print(f"\nGiven the query \"{query}\": for the naive indexer, found postings: {naive_postings}")

    # OPERATE ON SPIMI INDEX

    # Search the SPIMI index for the single-term query
    spimi_postings = [posting[0] for posting in SPIMI[query]]

    print(f"\nGiven the query \"{query}\": for the SPIMI indexer, found postings: {spimi_postings}")


def unranked(query: str) -> None:
    """
    Performs a search on the naive and SPIMI indexes with a given multi-keyword query, each keyword separated by `AND`.
    Compare the results.

    :param query: The multi-keyword query to search for
    """

    # Turn query into a list of keywords, stripping AND
    query_clean = [w.strip() for w in query.split('AND')]

    # Start with all documents. Since we will find intersections, need to start with all documents in a set, so first
    # keyword can intersect with it
    spimi_postings = set(i for i in range(1, 10_001))  # Start with all documents

    # Go through each keyword in the query
    for q in query_clean:
        # Get the postings list for the query term in the SPIMI index. Intersect it with what we have above
        spimi_postings = spimi_postings.intersection({posting[0] for posting in SPIMI[q]})

    print(f"\nGiven the query \"{query}\": for the SPIMI indexer, found postings: {list(spimi_postings)}")


def ranked(query: str, top_k: int = 10) -> None:
    """
    Performed ranked search on naive and SPIMI index, given a multi-keyword OR query. Compare results.

    :param query: The multi-keyword query to search for
    :param top_k: The number of best results to return e.g. top 10 results only.
    """

    # Turn query into a list of keywords, stripping OR
    query_clean = [w.strip() for w in query.split('OR')]

    # Create an empty list for all postings found in this query
    spimi_postings = []

    # Go through all query terms
    for q in query_clean:

        # Add the found postings to the main list
        spimi_postings += [posting[0] for posting in SPIMI[q]]

    # Sort the postings list found by frequency of documents, such that postings with higher frequency appear first
    spimi_postings = sorted(spimi_postings, key=spimi_postings.count, reverse=True)

    # Create a dict to associate documents with how many query terms in that document
    spimi_result = {}

    # Go through each posting that was found
    for posting in spimi_postings:

        # If we've reached the top_k documents, stop
        if len(spimi_result.keys()) == top_k:
            break

        # If we don't already have this posting in the dictionary, add it and its frequency in the main postings list
        if posting not in spimi_result.keys():
            spimi_result[posting] = spimi_postings.count(posting)

    print(f"\nGiven the query \"{query}\": for the SPIMI indexer, found the top {top_k} postings " +
          "({posting: count}): " + f"{spimi_result}")


def BM25(query: str, k_1: float = 1.5, b: float = 0.75, top_k: int = 10) -> None:
    """
    Compute BM25 ranking for a given query.

    Based off the first formula in the book that has k_1 and b parameters (p. 233, fig. 11.32).

    :param query: The query to find rankings for
    :param k_1: A free parameter, typically between 1.2 and 2.0
    :param b: A free parameter, typically 0.75
    :param top_k: Return the top k results to avoid overwhelming the user
    """

    # For each document in collection:
    # Use the terms in the query to compute the RSV for that document
    # Need N (# docs in collection), tf_td (count of term t in document d), L_d (length of doc d),
    # L_ave (avg. doc length in collection),

    RSV = {}

    # Turn query into a list of keywords
    query_clean = [w.strip() for w in query.split(' ')]

    # The number of total documents
    N = 21578

    # Get average document length
    with open('stats/avg_size.txt', 'rt') as f:
        L_ave = float(f.read())

    # Get individual document length
    with open('stats/doc_sizes.txt', 'rt') as f:
        doc_sizes = json.load(f)

    # Loop through each docID
    for d in range(1, N + 1):
        # Get the length of this document
        L_d = doc_sizes[str(d)]

        # Start RSV_d value for this document at 0
        RSV_d = 0

        # Sum over each term in the query
        for t in query_clean:
            # Find the document frequency for this term. This is represented by the postings list length
            df_t = len(SPIMI[t])

            # Find the term frequency for this term. This is represented by number of times the term t
            # appears in document d. The 2nd term in each of the tuples in the postings list.
            # Go through each tuple in the postings list, and the tf_td is the 2nd value in the tuple whose
            # first value is d
            try:
                tf_td = [tup[1] for tup in SPIMI[t] if tup[0] == d][0]
            except IndexError:
                tf_td = 0

            # Compute the log factor of the BM25 formula
            log_factor = log(N / df_t, 10)

            # Compute the rational factor of the BM25 formula
            numerator = (k_1 + 1) * tf_td
            denominator = k_1 * ((1 - b) + b * (L_d / L_ave)) + tf_td
            rational_factor = numerator / denominator

            # Add to the RSV_d value for this document, the result of multiplying the two factors
            RSV_d += round(log_factor * rational_factor, 2)

        # Add the RSV_d value for this document to the dictionary associating each document with its RSV
        RSV[d] = RSV_d

    # Sort the documents by highest RSV values
    RSV = {k: v for k, v in sorted(RSV.items(), key=lambda item: item[1], reverse=True)}

    # Get only top k results
    RSV_top_k = list(RSV.items())[: top_k]

    print(f"\nGiven the query \"{query}\": for the SPIMI indexer, found the top {top_k} postings " +
          "({posting: RSV score}): " + f"{RSV_top_k}")


def main():
    test1 = "Bush"  # Single word query
    test2 = "drug AND bankruptcy"  # Multiple keyword query (Unranked)
    test3 = "Democrat OR welfare OR healthcare OR reform OR policy"  # Multiple keyword query (Ranked)
    test4 = "Democrat welfare healthcare reform policy"  # BM25 query

    # Load indexes
    with open('index/naive_index.txt', 'rt') as f:
        global naive
        naive = json.load(f)

    with open('index/spimi_index.txt', 'rt') as f:
        global SPIMI
        SPIMI = json.load(f)

    print(f'\n---------- Test Query (a): "{test1}" ----------')
    single(test1)

    print(f'\n---------- Test Query (b): "{test2}" ----------')
    unranked(test2)

    print(f'\n---------- Test Query (c): "{test3}" ----------')
    ranked(test3)

    print(f'\n---------- Test Query (d): "{test4}" ----------')
    BM25(test4)


if __name__ == '__main__':
    main()
