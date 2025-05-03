#!/bin/bash

########
# Print current node information and allocated nodes
echo "Running on node: $(hostname)"
echo "Allocated nodes: $SLURM_NODELIST"
########

echo "========Stopping any existing Elasticsearch processes on this node..."
#pkill -f elasticsearch || true
#killall -9 elasticsearch
#sleep 5

echo "========Removing stale lock files from shared directories..."
rm -f /u4/z6dong/shared_data/elasticsearch-8.11.1/data/_state/write.lock
rm -f /u4/z6dong/shared_data/elasticsearch-8.11.1/data/snapshot_cache/write.lock
rm -f /u4/z6dong/shared_data/es_data_persistent/node.lock

# Set lower JVM heap settings to avoid excessive memory usage
export ES_JAVA_OPTS="-Xms4g -Xmx4g"

# Set up a local data directory to ensure exclusive usage
#ES_DATA_DIR="/tmp/elasticsearch_data_${SLURM_JOB_ID}"
ES_DATA_DIR="/u4/z6dong/shared_data/es_data_persistent"
#rm -rf ${ES_DATA_DIR}
#mkdir -p ${ES_DATA_DIR} && 
chmod 700 ${ES_DATA_DIR}

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
  -Ecluster.routing.allocation.disk.threshold_enabled=false \
  -Ecluster.routing.allocation.enable=all \
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

echo "========Adjusting index settings..."
curl -XPUT "http://${NODE_IP}:9200/_all/_settings" -H 'Content-Type: application/json' -d'
{
    "index": {
        "number_of_replicas" : 0
    }
}'
sleep 10

echo "========Checking cluster health..."
curl -XGET "http://${NODE_IP}:9200/_cluster/health?wait_for_status=yellow&timeout=120s"

echo "========Running bulk import..."

#python build_mini_citation_es.py --mode build --directory /u4/z6dong/shared_data/se_citations_250218 --index_name citations_index --fields minimal
#python build_mini_citation_es.py --mode build --directory /u4/z6dong/shared_data/se_citations_250218 --index_name citations_index_full --fields full
#python build_mini_citation_es.py --mode query --index_name citations_index --id 150223110
#python build_mini_citation_es.py --mode test --index_name citations_index
#python build_mini_citation_es.py --mode update --directory /u4/z6dong/shared_data/se_citations_250218 --index_name citations_index # update from minimal to full
python build_mini_citation_es.py --mode prepare_ids
python build_mini_citation_es.py --mode batch --index_name citations_index --input_file tmp_local_ids.txt --output_file batch_results.parquet


echo "========Bulk import completed."

echo "========Killing Elasticsearch process with PID ${ES_PID}..."
kill ${ES_PID}
echo "========Elasticsearch process killed."

