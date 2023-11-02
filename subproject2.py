import json

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

    print(f"\nGiven the query {query}: for the naive indexer, found postings: {naive_postings}")

    # OPERATE ON SPIMI INDEX

    # Search the SPIMI index for the single-term query
    spimi_postings = SPIMI[query]

    print(f"\nGiven the query {query}: for the SPIMI indexer, found postings: {spimi_postings}")


def unranked(query: str) -> None:
    """
    Performs a search on the naive and SPIMI indexes with a given multi-keyword query, each keyword separated by `AND`.
    Compare the results.

    :param query: The multi-keyword query to search for
    """

    # Turn query into a list of keywords, stripping AND
    query_clean = [w.strip() for w in query.split('AND')]

    # OPERATE ON NAIVE INDEX

    # Start with all documents. Since we will find intersections, need to start with all documents in a set, so first
    # keyword can intersect with it
    naive_postings = set(i for i in range(1, 10_001))

    # Go through each keyword in the query
    for q in query_clean:
        # Get the postings list for the query term in the naive index. Intersect it with what we have above
        naive_postings = naive_postings.intersection(naive[q])

    print(f"\nGiven the query {query}: for the naive indexer, found postings: {list(naive_postings)}")

    # OPERATE ON SPIMI INDEX

    # Start with all documents. Since we will find intersections, need to start with all documents in a set, so first
    # keyword can intersect with it
    spimi_postings = set(i for i in range(1, 10_001))  # Start with all documents

    # Go through each keyword in the query
    for q in query_clean:
        # Get the postings list for the query term in the SPIMI index. Intersect it with what we have above
        spimi_postings = spimi_postings.intersection(SPIMI[q])

    print(f"\nGiven the query {query}: for the SPIMI indexer, found postings: {list(spimi_postings)}")


def ranked(query: str, top_k: int = 10) -> None:
    """
    Performed ranked search on naive and SPIMI index, given a multi-keyword OR query. Compare results.

    :param query: The multi-keyword query to search for
    :param top_k: The number of best results to return e.g. top 10 results only.
    """

    # Turn query into a list of keywords, stripping OR
    query_clean = [w.strip() for w in query.split('OR')]

    # OPERATE ON NAIVE INDEX

    # Create an empty list for all postings found in this query
    naive_postings = []

    # Go through all query terms
    for q in query_clean:

        # Add the found postings to the main list
        naive_postings += naive[q]

    # Sort the postings list found by frequency of documents, such that postings with higher frequency appear first
    naive_postings = sorted(naive_postings, key=naive_postings.count, reverse=True)

    # Create a dict to associate documents with how many query terms in that document
    naive_result = {}

    # Go through each posting that was found
    for posting in naive_postings:

        # If we've reached the top_k documents, stop
        if len(naive_result.keys()) == top_k:
            break

        # If we don't already have this posting in the dictionary, add it and its frequency in the main postings list
        if posting not in naive_result.keys():
            naive_result[posting] = naive_postings.count(posting)

    print(f"\nGiven the query {query}: for the naive indexer, found postings " +
          "({posting: count}): " + f"{naive_result}")

    # OPERATE ON SPIMI INDEX

    # Create an empty list for all postings found in this query
    spimi_postings = []

    # Go through all query terms
    for q in query_clean:

        # Add the found postings to the main list
        spimi_postings += SPIMI[q]

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

    print(f"\nGiven the query {query}: for the SPIMI indexer, found postings " +
          "({posting: count}): " + f"{spimi_result}")


def BM25(query: str):
    pass


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

    single(test1)
    unranked(test2)
    ranked(test3)


if __name__ == '__main__':
    main()
