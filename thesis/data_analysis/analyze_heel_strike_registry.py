import json
from collections import defaultdict
from pathlib import Path

def analyze_alternation(hs_path, labels_path):
    print(f"Loading heel strike data from: {hs_path}")
    with open(hs_path, 'r') as f:
        hs_data = json.load(f)
        
    print(f"Loading labels from: {labels_path}\n")
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)["key_to_severity"]

    L_ANKLE = 6
    R_ANKLE = 3

    class_totals = defaultdict(int)
    class_violations = defaultdict(int)
    violating_seqs = []
    
    total_violations = 0
    total_seqs = len(hs_data)

    for clip_id, strikes in hs_data.items():
        base_key = clip_id.split('_down')[0] if '_down' in clip_id else clip_id
        base_key = base_key.replace('generated_walk_', '')
        severity = labels_data.get(base_key, "Unknown")
        
        class_totals[severity] += 1

        # at least 2 strikes for foot alternation check
        if len(strikes) < 2:
            continue
            
        # chronologically sort strikes by frame
        strikes = sorted(strikes, key=lambda x: x["frame"])
        
        seq_violations = 0
        violation_details = []
        
        for i in range(1, len(strikes)):
            prev_joint = strikes[i-1]["joint_idx"]
            curr_joint = strikes[i]["joint_idx"]
            
            if prev_joint == curr_joint:
                seq_violations += 1
                total_violations += 1
                
                leg = "Left" if curr_joint == L_ANKLE else "Right"
                f1 = strikes[i-1]["frame"]
                f2 = strikes[i]["frame"]
                violation_details.append(f"{leg}-{leg} (Frames {f1} & {f2})")
                
        if seq_violations > 0:
            class_violations[severity] += 1
            violating_seqs.append({
                "clip_id": clip_id,
                "severity": severity,
                "total_strikes": len(strikes),
                "violations": seq_violations,
                "details": violation_details
            })

    # Give summary stats
    print(f"Total Sequences Analyzed: {total_seqs}")
    
    overall_violation_rate = (len(violating_seqs) / total_seqs) * 100 if total_seqs > 0 else 0
    print(f"Sequences with >= 1 Violation: {len(violating_seqs)} ({overall_violation_rate:.2f}%)\n")
    
    print("--- Breakdown by Severity Class ---")
    for sev in sorted(class_totals.keys(), key=lambda x: str(x)):
        tot = class_totals[sev]
        bad = class_violations[sev]
        rate = (bad / tot) * 100 if tot > 0 else 0
        print(f"  Class {sev:<7} : {bad:>4} / {tot:<4} sequences affected ({rate:>5.1f}%)")

    print("\n--- Sequences Requiring Review (Sorted by # of Violations) ---")
    
    if not violating_seqs:
        print("No alternation violations found.")
        return

    # Sort descending by number of violations
    violating_seqs.sort(key=lambda x: x["violations"], reverse=True)
    
    print(f"{'Clip ID':<15} | {'Class':<5} | {'Strikes':<7} | {'Errors':<6} | {'Details'}")
    print("-" * 60)
    
    # Print the top 10 sequences with most violations
    for v in violating_seqs[:10]:
        # details_str = ", ".join(v["details"])
        print(f"{v['clip_id']:<15} | {v['severity']:<5} | {v['total_strikes']:<7} | {v['violations']:<6} | ")
        
    if len(violating_seqs) > 10:
        print(f"... and {len(violating_seqs) - 10} more sequences.")

    v_class_3 = [v for v in violating_seqs if v["severity"] == 3]
    print(f"Class 3 violation sequences:")
    for v in v_class_3:
        print(f"{v['clip_id']:<15} | {v['severity']:<5} | {v['total_strikes']:<7} | {v['violations']:<6} | ")

if __name__ == "__main__":
    hs_path = Path("thesis/data/processed/baseline_model_v2_epochs100/h36m/heel_strikes.json")
    labels_path = Path("thesis/data/processed/baseline_model_v2_epochs100/h36m/gen_labels.json")
    
    if not hs_path.exists():
        print(f"Error: Could not find {hs_path}")
    else:
        analyze_alternation(hs_path, labels_path)