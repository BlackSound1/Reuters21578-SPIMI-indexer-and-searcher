import json
from math import log
from pathlib import Path

from nltk.stem import PorterStemmer

# Define global variables for the indexes
naive: dict
SPIMI: dict


def single(query: str, dir: str) -> None:
    """
    Performs a single-term search for the given query on the naive and SPIMI indexes. Compares results.

    :param query: The single-term query to search for
    :param dir: The subdirectory to save to
    """

    # OPERATE ON NAIVE INDEX

    # Search the naive index for the single-term query
    naive_postings = naive[query]

    print(f"\nGiven the query \"{query}\": for the naive indexer, found postings: {naive_postings}")

    # Save results to file
    Path(f'query_results/{dir}').mkdir(exist_ok=True, parents=True)
    with open(f'query_results/{dir}/{query}-naive.txt', 'wt') as f:
        json.dump(naive_postings, f)

    # OPERATE ON SPIMI INDEX

    # Search the SPIMI index for the single-term query
    spimi_postings = [posting[0] for posting in SPIMI[query]]

    print(f"\nGiven the query \"{query}\": for the SPIMI indexer, found postings: {spimi_postings}")

    # Save results to file
    Path(f'query_results/{dir}').mkdir(exist_ok=True, parents=True)
    with open(f'query_results/{dir}/{query}-SPIMI.txt', 'wt') as f:
        json.dump(spimi_postings, f)


def unranked(query: str, dir: str) -> None:
    """
    Performs a search on the naive and SPIMI indexes with a given multi-keyword query, each keyword separated by `AND`.
    Compare the results.

    :param query: The multi-keyword query to search for
    :param dir: The subdirectory to save to
    """

    # Turn query into a list of keywords, stripping AND
    query_clean = [w.strip() for w in query.split('AND')]

    # Start with all documents. Since we will find intersections, need to start with all documents in a set, so first
    # keyword can intersect with it
    spimi_postings = set(i for i in range(1, 21_579))  # Start with all documents

    # Go through each keyword in the query
    for q in query_clean:
        # Get the postings list for the query term in the SPIMI index. Intersect it with what we have above
        spimi_postings = spimi_postings.intersection({posting[0] for posting in SPIMI[q]})

    spimi_postings = list(spimi_postings)

    print(f"\nGiven the query \"{query}\": for the SPIMI indexer, found postings: {spimi_postings}")

    # Write to file
    Path(f'query_results/{dir}').mkdir(exist_ok=True, parents=True)
    with open(f'query_results/{dir}/{query}.txt', 'wt') as f:
        json.dump(spimi_postings, f)


def ranked(query: str, dir: str, top_k: int = 10) -> None:
    """
    Performed ranked search on naive and SPIMI index, given a multi-keyword OR query. Compare results.

    :param query: The multi-keyword query to search for
    :param dir: The subdirectory to save to
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

    # Write to file
    Path(f'query_results/{dir}').mkdir(exist_ok=True, parents=True)
    with open(f'query_results/{dir}/{query}.txt', 'wt') as f:
        json.dump(spimi_result, f)


def BM25(query: str, dir: str, k_1: float = 1.5, b: float = 0.75, top_k: int = 10) -> None:
    """
    Compute BM25 ranking for a given query.

    Based off the first formula in the book that has k_1 and b parameters (p. 233, fig. 11.32).

    :param query: The query to find rankings for
    :param dir: The subdirectory to save to
    :param k_1: A free parameter, typically between 1.2 and 2.0
    :param b: A free parameter, typically 0.75
    :param top_k: Return the top k results to avoid overwhelming the user
    """

    # Create the dict associating each document ID to its RSV
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
            # first value is d. This code may fail if this term is not found in this document. In that case,
            # I make sure tf_td is 0.
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
            RSV_d += log_factor * rational_factor

        # Add the RSV_d value for this document to the dictionary associating each document with its RSV
        RSV[d] = RSV_d

    # Sort the documents by highest RSV values
    RSV = {k: v for k, v in sorted(RSV.items(), key=lambda item: item[1], reverse=True)}

    # Round RSV to 2 decimal places
    RSV = {k: round(v, 2) for k, v in RSV.items()}

    # Get only top k results
    RSV_top_k = list(RSV.items())[: top_k]

    print(f"\nGiven the query \"{query}\": for the SPIMI indexer, found the top {top_k} postings " +
          "({posting: RSV score}): " + f"{RSV_top_k}")

    # Write to file
    Path(f'query_results/{dir}').mkdir(exist_ok=True, parents=True)
    with open(f'query_results/{dir}/{query}.txt', 'wt') as f:
        json.dump(RSV_top_k, f)


def main():
    # Make sure the query_results/ folder exists
    Path('query_results/').mkdir(exist_ok=True, parents=True)

    # Load indexes
    with open('index/naive_index.txt', 'rt') as f:
        global naive
        naive = json.load(f)

    with open('index/spimi_index.txt', 'rt') as f:
        global SPIMI
        SPIMI = json.load(f)

    print(f'\n---------- Development Queries ----------')

    test1 = "Bush"  # Single word query
    test2 = "drug AND company AND bankruptcy"  # Multiple keyword query (Unranked)
    test3 = "Democrat OR welfare OR healthcare OR reform OR policy"  # Multiple keyword query (Ranked)
    test4 = "Democrat welfare healthcare reform policy"  # BM25 query

    print(f'\n======= (a): "{test1}" =======')
    single(test1, '1. development')

    print(f'\n======= (b): "{test2}" =======')
    unranked(test2, '1. development')

    print(f'\n======= (c): "{test3}" =======')
    ranked(test3, '1. development')

    print(f'\n======= (d): "{test4}" =======')
    BM25(test4, '1. development')

    print(f'\n---------- Test Queries ----------')

    # Treat the information needs as literal queries
    BM25("Democrats welfare and healthcare reform policies", '2. test')
    BM25("Drug company bankruptcies", '2. test')
    BM25("George Bush", '2. test')

    print(f'\n---------- Other Queries ----------')

    print(f'\n======= Single-Term Queries from Project 2 =======')

    print(f'\n~~~ Project 2 Test Queries ~~~')

    # Project 2 single-term queries (test queries)
    single('abnormally', '3. other/P2-test')
    single('017', '3. other/P2-test')
    single('Zweig', '3. other/P2-test')

    print(f'\n~~~ Project 2 Sample Queries ~~~')
    # Create Porter stemmer
    stemmer = PorterStemmer()

    # Project 2 single-term queries (sample queries). Perform stemming and case-folding like in Project 2
    single(stemmer.stem('males').lower(), '3. other/P2-sample')
    single(stemmer.stem('CORRECTED').lower(), '3. other/P2-sample')
    single(stemmer.stem('texts').lower(), '3. other/P2-sample')

    print(f'\n======= Other Unranked Boolean "AND" Queries =======')
    unranked("Queen AND King", '3. other/AND')
    unranked("Queen AND Royal", '3. other/AND')
    unranked("King AND Royal", '3. other/AND')
    unranked("Queen AND King AND Royal", '3. other/AND')
    unranked("Royal AND family", '3. other/AND')

    print(f'\n======= Other Ranked Boolean "OR" Queries =======')
    ranked("Cancer OR treatment", "3. other/OR")
    ranked("Cancer OR treatment OR doctor", "3. other/OR")
    ranked("Cancer OR treatment OR doctor OR hospital", "3. other/OR")

    print(f'\n======= Other Ranked BM25 Queries =======')
    BM25("Ecuador in Latin America", '3. other/BM25')
    BM25("history of Europe", '3. other/BM25')
    BM25("great movies and books", '3. other/BM25')
    BM25("India Bangladesh and Pakistan", '3. other/BM25')


if __name__ == '__main__':
    main()
