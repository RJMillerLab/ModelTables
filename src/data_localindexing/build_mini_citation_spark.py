
#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Created: 2025-03-09
Last Modified: 2025-03-20 (Added Spark query)

Usage:
    Build the graph database:
        python build_mini_citation_spark.py --directory ./ --node_id 248811336

Description:
    This script processes all files in the specified directory whose names match "step*_file".
    Each file is assumed to be NDJSON (one JSON object per line).

    Each JSON object has the following fields:
        - citationid (INT64)  ######## changed to INT64 ########
        - citingcorpusid (INT64) ######## changed to INT64 ########
        - citedcorpusid  (INT64) ######## changed to INT64 ########
        - isinfluential (BOOL)
        - contexts (JSON array)
        - intents (JSON array, e.g. list of lists)

    We create a minimal node "Corpus" that only stores the id,
    and a relationship "CITES" (from Corpus to Corpus) that stores the citation edge.
    This approach minimizes node storage while focusing on citation edges.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, BooleanType, ArrayType, StringType
import os
import argparse

# Define the schema matching your NDJSON structure.
schema = StructType([
    StructField("citationid", LongType(), True),
    StructField("citingcorpusid", LongType(), True),
    StructField("citedcorpusid", LongType(), True),
    StructField("isinfluential", BooleanType(), True),
    # Assuming contexts is an array of strings.
    StructField("contexts", ArrayType(StringType()), True),
    # Assuming intents is an array of arrays of strings.
    StructField("intents", ArrayType(ArrayType(StringType())), True)
])

def main():
    parser = argparse.ArgumentParser(description="Query NDJSON citation data directly with Spark (without rewriting).")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing NDJSON files matching 'step*_file'")
    parser.add_argument("--node_id", type=int, required=True, help="Node ID to query")
    args = parser.parse_args()
    # Create SparkSession.
    spark = SparkSession.builder.appName("NDJSONBatchQuery").getOrCreate()
    # Build the input file pattern.
    input_path = os.path.join(args.directory, "step*_file")
    print("Reading NDJSON files from:", input_path)
    # Read NDJSON files as a DataFrame.
    df = spark.read.schema(schema).option("multiLine", "false").json(input_path)
    print("Total records read:", df.count())
    node_id = args.node_id
    # Outgoing edges: where the node is the citing paper.
    cites_other = df.filter(df.citingcorpusid == node_id)
    # Incoming edges: where the node is cited by other papers.
    cited_by_other = df.filter(df.citedcorpusid == node_id)
    print("Papers that node {} cites:".format(node_id))
    cites_other.show(truncate=False)
    print("Papers that cite node {}:".format(node_id))
    cited_by_other.show(truncate=False)
    spark.stop()

if __name__ == "__main__":
    main()

