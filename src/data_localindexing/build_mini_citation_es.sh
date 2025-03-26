#!/bin/bash
#SBATCH --job-name=bulk_import
#SBATCH --output=bulk_import.out
#SBATCH --time=01:00:00

echo "========Starting Elasticsearch..."
pkill -f elasticsearch || true
killall -9 elasticsearch
sleep 5
ES_PATH="/u4/z6dong/shared_data/elasticsearch-8.11.1/bin/elasticsearch"
#nohup $ES_PATH > es.log 2>&1 &

rm -f /u4/z6dong/shared_data/elasticsearch-8.11.1/data/_state/write.lock
rm -f /u4/z6dong/shared_data/elasticsearch-8.11.1/data/snapshot_cache/write.lock
ES_DATA_DIR="/tmp/elasticsearch_data_$SLURM_JOB_ID"
rm -rf $ES_DATA_DIR
mkdir -p $ES_DATA_DIR
rm -f $ES_DATA_DIR/node.lock
nohup $ES_PATH -Epath.data=$ES_DATA_DIR -Expack.security.enabled=false > es.log 2>&1 &
ES_PID=$!
echo "========Elasticsearch started with PID ${ES_PID}"

echo "========Waiting for Elasticsearch to initialize..."
max_wait=120
waited=0
while ! curl -k -u elastic:"6KdUGb=SifNeWOy__lEz" https://localhost:9200 >/dev/null 2>&1; do
    sleep 5
    waited=$((waited+5))
    echo "Waiting... (${waited}s)"
    if [ $waited -ge $max_wait ]; then
        echo "Elasticsearch did not start in time. Exiting."
        exit 1
    fi
done
echo "========Elasticsearch is up!"

echo "========Running bulk import..."
#python run_es.py --mode build --directory /u4/z6dong/shared_data/se_s2orc_250218 --index_name papers_index
#python run_es.py --mode test --directory /u4/z6dong/shared_data/se_s2orc_250218 --index_name papers_index
#python run_es.py --mode query --directory /u4/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --query "BioMANIA: Simplifying bioinformatics data analysis through conversation"
python run_citations_es.py --mode build --directory /u4/z6dong/shared_data/se_citations_250218 --index_name citations_index --fields minimal

echo "========Bulk import completed."

echo "========Killing Elasticsearch process with PID ${ES_PID}..."
kill ${ES_PID}
echo "========Elasticsearch process killed."


