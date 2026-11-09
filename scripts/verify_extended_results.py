#!/usr/bin/env python3
"""Verify locked PAMPA calibration anchors, raw-evidence reconstruction status, and full-InChIKey identity-audit counts in the consolidated manuscript reproducibility package."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def close(a,b,tol=5e-6):
    return abs(float(a)-float(b))<=tol

def main():
    p=ROOT/"results/stage6/pampa_calibration/crossfit_summary_mean_sd.csv"
    if not p.is_file(): raise SystemExit("Missing Stage VI calibration summary")
    d=pd.read_csv(p).set_index(["target","method"])
    anchors={
        ("H","platt_oof"):{"brier_mean":0.200611,"ece10_mean":0.049097},
        ("V","platt_oof"):{"brier_mean":0.201006,"ece10_mean":0.051000},
        ("I","platt_oof"):{"brier_mean":0.198961,"ece10_mean":0.040904},
        ("H","raw_clipped"):{"brier_mean":0.263517,"ece10_mean":0.255298},
        ("I","raw_clipped"):{"brier_mean":0.249760,"ece10_mean":0.231852},
    }
    for key,vals in anchors.items():
        if key not in d.index: raise SystemExit(f"Missing calibration row {key}")
        for col,expected in vals.items():
            if not close(d.loc[key,col],expected):
                raise SystemExit(f"Calibration anchor mismatch {key} {col}: {d.loc[key,col]} vs {expected}")
    q=ROOT/"results/audits/evidence_reconstruction/qc_report.json"
    if not q.is_file(): raise SystemExit("Missing evidence reconstruction qc_report.json")
    qc=json.loads(q.read_text(encoding="utf-8"))
    status=str(qc.get("status","")).upper()
    if status not in {"PASS","OK","COMPLETE"} and not bool(qc.get("all_checks_passed",False)):
        raise SystemExit(f"Evidence reconstruction not PASS: {qc}")
    m=ROOT/"results/audits/structure_identity/mixed15_full_inchikey_audit_metadata.json"
    if not m.is_file(): raise SystemExit("Missing mixed-label identity audit metadata")
    a=json.loads(m.read_text(encoding="utf-8"))
    expected={"n_parent_groups":15,"n_raw_records":65,"n_distinct_full_isomeric_InChIKey":65,"n_full_InChIKey_with_both_labels":0}
    for k,v in expected.items():
        if int(a.get(k,-1))!=v: raise SystemExit(f"Identity audit mismatch {k}: {a.get(k)} vs {v}")
    print(json.dumps({"status":"PASS","stage6_calibration_rows":int(len(pd.read_csv(p))),"evidence_reconstruction":"PASS","mixed_label_identity_audit":"PASS"},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
