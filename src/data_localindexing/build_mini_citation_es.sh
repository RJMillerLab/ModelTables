#!/bin/bash
#SBATCH --job-name=bulk_import
#SBATCH --output=bulk_import.out
#SBATCH --time=01:00:00
#SBATCH --mem=32G

########
# Print current node information and allocated nodes
echo "Running on node: $(hostname)"
echo "Allocated nodes: $SLURM_NODELIST"
########

echo "========Stopping any existing Elasticsearch processes on this node..."
pkill -f elasticsearch || true
killall -9 elasticsearch
sleep 5

echo "========Removing stale lock files from shared directories..."
rm -f /u4/z6dong/shared_data/elasticsearch-8.11.1/data/_state/write.lock
rm -f /u4/z6dong/shared_data/elasticsearch-8.11.1/data/snapshot_cache/write.lock

# Set lower JVM heap settings to avoid excessive memory usage
export ES_JAVA_OPTS="-Xms4g -Xmx4g"

# Set up a local data directory to ensure exclusive usage
ES_DATA_DIR="/tmp/elasticsearch_data_${SLURM_JOB_ID}"
rm -rf ${ES_DATA_DIR}
mkdir -p ${ES_DATA_DIR} && chmod 700 ${ES_DATA_DIR}
rm -f ${ES_DATA_DIR}/node.lock

# Get the real IP address of this node (assume the first IP is usable)
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Node IP: ${NODE_IP}"

echo "========Starting Elasticsearch..."
ES_PATH="/u4/z6dong/shared_data/elasticsearch-8.11.1/bin/elasticsearch"
nohup ${ES_PATH} \
  -Epath.data=${ES_DATA_DIR} \
  -Ediscovery.type=single-node \
  -Ehttp.host=0.0.0.0 \
  -Etransport.host=${NODE_IP} \
  -Expack.security.enabled=false \
  > es.log 2>&1 &
ES_PID=$!
echo "========Elasticsearch started with PID ${ES_PID}"

echo "========Waiting for Elasticsearch to initialize..."
max_wait=120
waited=0
# Using HTTP since security is disabled
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

echo "========Running bulk import..."
python run_citations_es.py --mode build \
    --directory /u4/z6dong/shared_data/se_citations_250218 \
    --index_name citations_index \
    --fields minimal 

echo "========Bulk import completed."

echo "========Killing Elasticsearch process with PID ${ES_PID}..."
kill ${ES_PID}
echo "========Elasticsearch process killed."
