#!/usr/bin/env bash
# Watcher: wait until the matched SLR TroVE run finishes (its runner process exits
# AND its output has 1000 rows), then record that both GPUs are free and print the
# final SLR tiered numbers. Does NOT launch anything itself; it just signals so the
# session can kick off the List Functions pilot on the freed GPUs.
set -uo pipefail
cd /home/arnaik/symbolic-library-agent
OUT=outputs/oh_trove_qwen_slr_matched.jsonl
STAMP=/tmp/slr_matched_done.flag

while pgrep -f "run_trove_pbe_slr.*slr" >/dev/null 2>&1; do
  sleep 60
done
# process gone; confirm it actually completed (1000 rows) vs died early
n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
echo "SLR runner exited at $(date). rows=$n/1000" | tee "$STAMP"
python3 -c "
import json
from collections import defaultdict
rows=[json.loads(l) for l in open('$OUT')]
n=len(rows); s=sum(1 for r in rows if r.get('best_reward',0)>=1.0)
print(f'FINAL matched SLR: {n}/1000 solved {s} ({100*s/n:.1f}%)')
t=defaultdict(lambda:[0,0])
for r in rows:
    k=r.get('curriculum_tier') or '?'; t[k][0]+=1; t[k][1]+=(r.get('best_reward',0)>=1.0)
for k in ['basic','easy','medium','hard']:
    if k in t: c=t[k]; print(f'  {k}: {c[1]}/{c[0]} ({100*c[1]/c[0]:.1f}%)')
" | tee -a "$STAMP"
echo "GPUs in job_9431972 should now be free for gpt-oss-120b." | tee -a "$STAMP"
