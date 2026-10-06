# Shared helpers for "one job per GPU" fan-out launchers.  Source, don't execute.
#
# Why this exists (incident 2026-09-04, two host OOMs at 13:58 and 15:15):
# sglang picks a shard's NUMA node in numa_utils._query_numa_node_for_gpu(gpu_id)
# via pynvml.nvmlDeviceGetHandleByIndex(gpu_id).  NVML indexes physically and
# ignores CUDA_VISIBLE_DEVICES, while gpu_id inside a single-GPU job is always 0
# (base_gpu_id + tp_rank, tp=1).  So every shard resolved to physical GPU 0 and
# sglang spawned its scheduler under `numactl --cpunodebind=0 --membind=0`.
# All 8 shards were then hard-capped at node 0's 1 TB instead of the box's 2 TB,
# and crammed onto node 0's 112 CPUs.  Binding each shard to the node that really
# owns its GPU fixes the split; the resulting constrained cpu affinity also makes
# sglang skip its own (wrong) binding on both code paths -- see
# numa_utils._is_numa_available() and scheduler.run_scheduler_process.

# numa_for_gpu <physical gpu index>  ->  numa node id, or "" if undeterminable
numa_for_gpu() {
    local g="$1" bus dev node
    bus=$(nvidia-smi -i "$g" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null \
          | tr -d '[:space:]' | tr 'A-Z' 'a-z')
    [ -n "$bus" ] || return 0
    dev="${bus:4}"                       # 00000000:19:00.0 -> 0000:19:00.0
    node=$(cat "/sys/bus/pci/devices/$dev/numa_node" 2>/dev/null)
    case "$node" in ''|-1) return 0 ;; esac
    echo "$node"
}

# numa_prefix <physical gpu index>  ->  "numactl --cpunodebind=N --preferred=N" or ""
# --preferred (not --membind) keeps allocations node-local for speed but still lets
# a shard spill to the other node rather than being OOM-killed at the node wall.
numa_prefix() {
    local node
    node=$(numa_for_gpu "$1")
    [ -n "$node" ] && command -v numactl >/dev/null 2>&1 \
        && echo "numactl --cpunodebind=$node --preferred=$node"
}

# oom_shield <pid>...  -- make these workers the OOM killer's first victims so a
# runaway job can never take down the ssh/Claude session it was launched from
# (everything here lives in the same /system.slice/teleport.service cgroup).
oom_shield() {
    local p
    for p in "$@"; do echo 1000 > "/proc/$p/oom_score_adj" 2>/dev/null; done
}

# mem_watchdog <floor_gb> <log> <pid>...  -- background guard.  Gracefully SIGTERMs
# the workers if system MemAvailable or either node's MemFree drops below the floor,
# so we lose one checkpointed sub-batch instead of having the kernel shoot the box.
mem_watchdog() {
    local floor_gb="$1" log="$2"; shift 2
    local pids="$*"
    # stdout MUST be redirected: a backgrounded subshell inherits the caller's
    # command-substitution pipe and holds it open, so `WD=$(mem_watchdog ...)`
    # would block until the watchdog exited.  All messages go to $log explicitly.
    (
        while :; do
            local alive=0 p
            for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
            [ "$alive" = 1 ] || exit 0
            local avail_gb min_node_gb=999999 n free_gb
            avail_gb=$(( $(awk '/^MemAvailable:/{print $2}' /proc/meminfo) / 1048576 ))
            # Per-node headroom must count reclaimable memory, not just MemFree: streaming
            # 136k images fills Inactive(file) with page cache the kernel hands back on
            # demand, so a MemFree-only floor false-positives and shoots a healthy run.
            # Shmem is deliberately NOT counted -- sglang's IPC segments are unswappable
            # (and swap is full), so they are real pressure.
            for n in /sys/devices/system/node/node[0-9]*/meminfo; do
                free_gb=$(awk '/MemFree:/{f=$4} /Inactive\(file\):/{i=$4} /SReclaimable:/{s=$4}
                               END{printf "%d", (f+i+s)/1048576}' "$n")
                [ "$free_gb" -lt "$min_node_gb" ] && min_node_gb=$free_gb
            done
            if [ "$avail_gb" -lt "$floor_gb" ] || [ "$min_node_gb" -lt "$floor_gb" ]; then
                echo "[watchdog $(date +%T)] LOW MEM avail=${avail_gb}G min_node_free=${min_node_gb}G" \
                     "< floor=${floor_gb}G -- SIGTERM to shards (they checkpoint; rerun to resume)" >> "$log"
                for p in $pids; do kill -TERM "$p" 2>/dev/null; done
                exit 1
            fi
            echo "[watchdog $(date +%T)] avail=${avail_gb}G min_node_free=${min_node_gb}G" >> "$log"
            sleep 30
        done
    ) >/dev/null 2>&1 &
    echo $!
}
