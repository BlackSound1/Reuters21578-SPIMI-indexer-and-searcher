# Reuters21578 SPIMI Indexer and Searcher

## Installation

Download and unzip the Reuters 21578 corpus from <http://www.daviddlewis.com/resources/testcollections/reuters21578/>.

Download the dependencies in `requirements.txt`. (This step is unnecessary if using UV).

## Running

Run the first subproject with: `$ python subproject1.py`.

By default, Subproject 1 assumes that the unzipped corpus is on the same level as
this repository and is called `reuters21578`. If this is not the case, you can
specify the path to the corpus using the `-c` or `--corpus` flag when running the
program. Use it like `$ python subproject1.py -c /path/to/corpus/`.

This subproject:

- Gets all articles in `reuters21578`.
- Compute certain statistics about them for use later.
- Creates the naive and SPIMI indexes for the files.
- Computes the difference in how long each index took to create their first 10,000 dictionary terms.

Run the second subproject with: `$ python subproject2.py`. This subproject:

- Loads the two indexes.
- Runs four different search functions on 25 queries. One of these is BM25.
