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


def ranked(query: str):
    pass


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


if __name__ == '__main__':
    main()
