#!/bin/bash
#SBATCH --job-name=neo4j_load_job
#SBATCH --output=neo4j_load.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


echo "Starting Neo4j..."
/u4/z6dong/shared_data/neo4j-community-2025.02.0/bin/neo4j console

echo "Waiting 10s for Neo4j to be ready..."
sleep 10

echo "Running Python script to build the citation graph..."
python build_mini_citation_neo4j.py \
    --mode build \
    --directory /u4/z6dong/shared_data/se_citations_250218/ \
    --fields minimal

#echo "Stopping Neo4j..."
#/u4/z6dong/shared_data/neo4j-community-2025.02.0/bin/neo4j stop

echo "All done!"

