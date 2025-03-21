#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-03-20
Description: 
    Read NDJSON files (each line a JSON object with fields: 
      citationid, citingcorpusid, citedcorpusid, isinfluential, contexts, intents)
    and write data to Neo4j using Spark Connector.
    
    Nodes: use citingcorpusid and citedcorpusid (unique)
    Edges: use citationid as edge index, along with attributes: isinfluential, contexts, intents.
    
    Note: This script only writes necessary fields to Neo4j to avoid duplication.
    
Usage:
    spark-submit --packages org.neo4j:neo4j-connector-apache-spark_2.12:4.2.2 spark_to_neo4j.py --input_dir /path/to/ndjson --mode all
    --mode all # write nodes and relationships at same time.
"""

import os
import argparse
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, BooleanType, ArrayType, StringType

# Define schema based on NDJSON structure.
schema = StructType([
    StructField("citationid", LongType(), True),
    StructField("citingcorpusid", LongType(), True),
    StructField("citedcorpusid", LongType(), True),
    StructField("isinfluential", BooleanType(), True),
    # Assume contexts is an array of strings.
    StructField("contexts", ArrayType(StringType()), True),
    # Assume intents is an array of arrays of strings.
    StructField("intents", ArrayType(ArrayType(StringType())), True)
])

def main():
    parser = argparse.ArgumentParser(description="Write NDJSON citation data to Neo4j using Spark Connector.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing NDJSON files matching 'step*_file'")
    parser.add_argument("--mode", type=str, choices=["nodes", "edges", "all"], default="all",
                        help="Write mode: nodes, edges or all")
    args = parser.parse_args()

    # Create Spark session.
    spark = SparkSession.builder \
        .appName("NDJSONToNeo4j") \
        .getOrCreate()

    # Build file pattern, e.g., step1_file, step2_file, etc.
    input_path = os.path.join(args.input_dir, "step*_file")
    print("Reading NDJSON files from:", input_path)

    # Read NDJSON files as DataFrame.
    df = spark.read.schema(schema).option("multiLine", "false").json(input_path)
    print("Total records read:", df.count())

    # -------------------------------
    # Write Nodes (Corpus)
    # -------------------------------
    if args.mode in ["nodes", "all"]:
        # Create nodes DataFrame by taking union of citing and cited IDs.
        citing_nodes = df.selectExpr("citingcorpusid as id")
        cited_nodes  = df.selectExpr("citedcorpusid as id")
        nodes_df = citing_nodes.union(cited_nodes).distinct()
        print("Total unique nodes:", nodes_df.count())

        # Write nodes to Neo4j.
        nodes_df.write \
            .format("org.neo4j.spark.DataSource") \
            .mode("Overwrite") \
            .option("url", "bolt://localhost:7687") \
            .option("authentication.basic.username", "neo4j") \
            .option("authentication.basic.password", "11111111") \
            .option("labels", "Corpus") \
            .option("node.keys", "id") \
            .save()

        print("Nodes written to Neo4j.")

    # -------------------------------
    # Write Relationships (CITES)
    # -------------------------------
    if args.mode in ["edges", "all"]:
        # Preprocess edge attributes: transform contexts and intents to JSON strings to reduce memory overhead.
        # Here we add two new columns: contexts_str and intents_str.
        from pyspark.sql.functions import to_json, col
        df2 = df.withColumn("contexts_str", to_json(col("contexts"))) \
                .withColumn("intents_str", to_json(col("intents")))
        # Select necessary fields.
        edges_df = df2.selectExpr("citationid", "citingcorpusid", "citedcorpusid", "isinfluential", "contexts_str as contexts", "intents_str as intents")
        print("Total edges:", edges_df.count())

        # Write relationships to Neo4j.
        edges_df.write \
            .format("org.neo4j.spark.DataSource") \
            .mode("Overwrite") \
            .option("url", "bolt://localhost:7687") \
            .option("authentication.basic.username", "neo4j") \
            .option("authentication.basic.password", "11111111") \
            .option("relationship", "CITES") \
            .option("relationship.save.strategy", "keys") \
            .option("relationship.source.labels", "Corpus") \
            .option("relationship.source.node.keys", "citingcorpusid:id") \
            .option("relationship.target.labels", "Corpus") \
            .option("relationship.target.node.keys", "citedcorpusid:id") \
            .option("relationship.properties", "citationid,isinfluential,contexts,intents") \
            .save()

        print("Relationships written to Neo4j.")

    spark.stop()

if __name__ == "__main__":
    main()

