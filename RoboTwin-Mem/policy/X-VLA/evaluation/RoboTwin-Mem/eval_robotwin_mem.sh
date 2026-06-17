# Define your log directory here:
eval_log_dir=logs
port=4567

# Start your RoboTwin-Mem client
python client.py \
    --host 0.0.0.0 \
    --port $port \
    --eval_log_dir $eval_log_dir \
    --num_episodes 100 \
    --device 0 \
    --seed 100000 \
    --task_name pick_the_unhidden_block \
    --output_path $eval_log_dir \
    --task_config demo_clean
    
# Kill the server
PID=$(lsof -i :$port -t)
if [[ -n "$PID" ]]; then
    kill -9 $PID
fi
