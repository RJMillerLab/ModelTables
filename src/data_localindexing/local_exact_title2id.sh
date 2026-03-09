#!/bin/bash
#
# Local exact title->id batch query. Same ES setup as build_mini_citation_es.sh.
# Requires: papers_index already built (run build_mini_s2orc_es --mode build first).
#
# Usage: bash src/data_localindexing/local_exact_title2id.sh [TAG]
# Example: bash src/data_localindexing/local_exact_title2id.sh 251117
# Output: data/processed/s2orc_titles2ids_local_{TAG}.parquet

TAG="${1:-251117}"

########
echo "Running on node: $(hostname)"
echo "Allocated nodes: ${SLURM_NODELIST:-N/A}"
########

echo "========Removing stale lock files..."
rm -f /u501/z6dong/shared_data/elasticsearch-8.11.1/data/_state/write.lock
rm -rf /u501/z6dong/shared_data/elasticsearch-8.11.1/data/snapshot_cache

# Reduce heap if node has limited memory (avoid OOM). Increase if you have more RAM.
export ES_JAVA_OPTS="-Xms2g -Xmx2g"
ES_DATA_DIR="/u501/z6dong/shared_data/es_data_persistent"
rm -f ${ES_DATA_DIR}/node.lock
rm -rf ${ES_DATA_DIR}/snapshot_cache/write.lock
chmod 700 ${ES_DATA_DIR}

NODE_IP=$(hostname -I | awk '{print $1}')
echo "Node IP: ${NODE_IP}"

echo "========Starting Elasticsearch..."
ES_PATH="/u501/z6dong/shared_data/elasticsearch-8.11.1/bin/elasticsearch"
nohup ${ES_PATH} \
  -Epath.data=${ES_DATA_DIR} \
  -Ediscovery.type=single-node \
  -Ehttp.host=0.0.0.0 \
  -Etransport.host=${NODE_IP} \
  -Expack.security.enabled=false \
  -Ecluster.routing.allocation.disk.threshold_enabled=false \
  -Ecluster.routing.allocation.enable=all \
  > es.log 2>&1 &
ES_PID=$!

echo "========Waiting for Elasticsearch..."
max_wait=120
waited=0
while ! curl -s http://${NODE_IP}:9200 >/dev/null 2>&1; do
    sleep 5
    waited=$((waited+5))
    echo "Waiting... (${waited}s)"
    if [ $waited -ge $max_wait ]; then
        echo "Elasticsearch did not start in time. Exiting."
        exit 1
    fi
done
echo "========Elasticsearch is up!"

curl -XPUT "http://${NODE_IP}:9200/_all/_settings" -H 'Content-Type: application/json' -d'{"index":{"number_of_replicas":0}}'
sleep 5
curl -XGET "http://${NODE_IP}:9200/_cluster/health?wait_for_status=yellow&timeout=120s"

# Python must connect to same host as curl (NODE_IP). localhost may fail on some clusters.
export ES_HOST="http://${NODE_IP}:9200"
echo "========Running local exact title2id batch query..."
python -m local_exact_title2id --tag "${TAG}" --index_name papers_index

echo "========Done. Stopping Elasticsearch..."
kill ${ES_PID}
echo "========Elasticsearch stopped."
