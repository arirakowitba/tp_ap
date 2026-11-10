"""Master Continuous Overnight Pipeline Orchestrator.
Sequentially coordinates:
Phase 1: Transductive Cycle 2 Ensemble (already running or refitted)
Phase 2: Multi-Seed Tree Bagging (5 seeds x 5 folds = 50 models)
Phase 3: Granular Category Specialist Suite (10 verticals)
Phase 4: Synthesis of 5-Day / 5-Slot Podium Ladders for tomorrow 21:00 ART
Outputs comprehensive timestamps and progress checkpoints.
"""
import subprocess
import time
import os
import sys

print("=== STARTING CONTINUOUS OVERNIGHT PIPELINE ORCHESTRATOR ===")
start_pipeline = time.time()

def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

# Step 1: Wait for Phase 1 if running
log("Checking Phase 1 (Transductive Cycle 2)...")
phase1_out = "results/transductive_cycle2_20261110/transductive_cycle2_predictions.npz"
while not os.path.exists(phase1_out):
    log("Phase 1 is actively training in background. Waiting 30s...")
    time.sleep(30)
log("Phase 1 Complete! Artifact confirmed.")

# Step 2: Run Phase 2: Multi-Seed Tree Bagging
log("Launching Phase 2: Multi-Seed Deep Tree Bagging (50 models)...")
ret = subprocess.run([sys.executable, "scripts/train_multiseed_tree_bagging.py"])
if ret.returncode != 0:
    log(f"WARNING: Phase 2 exited with code {ret.returncode}")
else:
    log("Phase 2 Complete!")

# Step 3: Run Phase 3: Granular Category Specialists
log("Launching Phase 3: Granular Category Specialists (Top 10 Verticals)...")
ret = subprocess.run([sys.executable, "scripts/train_granular_category_specialists.py"])
if ret.returncode != 0:
    log(f"WARNING: Phase 3 exited with code {ret.returncode}")
else:
    log("Phase 3 Complete!")

# Step 4: Run Phase 4: Podium Ladder Synthesis
log("Launching Phase 4: Podium Ladder Synthesis...")
ret = subprocess.run([sys.executable, "scripts/synthesize_final_podium_ladders.py"])
if ret.returncode != 0:
    log(f"WARNING: Phase 4 exited with code {ret.returncode}")
else:
    log("Phase 4 Complete!")

log(f"=== FULL OVERNIGHT PIPELINE COMPLETE IN {time.time()-start_pipeline:.1f}s ===")
