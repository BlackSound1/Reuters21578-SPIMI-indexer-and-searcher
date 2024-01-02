# COMP479P3

## Installation

Download the Reuters 21578 corpus from http://www.daviddlewis.com/resources/testcollections/reuters21578/. Make sure the unzipped folder is at the same level as this repository and is called `reuters21578`.

Download the dependencies in `requirements.txt`.

## Running

Run the first subproject with: `$ python subproject1.py`. This subproject:
  - Gets all articles in `reuters21578`.
  - Compute certain statistics about them for use later.
  - Creates the naive and SPIMI indexes for the files.
  - Computes the difference in how long each index took to create their first 10,000 dictionary terms.

Run the second subproject with: `$ python subproject2.py`. This subproject: 
  - Loads the two indexes.
  - Runs four different search functions on 25 queries. One of these is BM25.
