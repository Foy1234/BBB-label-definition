# eval_audit.py — Audits identity consistency and overlap among BBBP test, scaffold test, reference-out, and independent experimental evaluation sets and reports cross-endpoint provenance.




from __future__ import annotations
import argparse
import hashlib
import json
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE5_PROJECT = ROOT/'BBB_AutoDL_Stage5_Chemspace_Final_v1'
STAGE3_PROJECT = ROOT/'BBB_AutoDL_Stage3_Controlled_Training_v1'
STAGE4_PROJECT = ROOT/'BBB_AutoDL_Stage4_Final_Stats_v1'
EXPECTED = {'train_pool':1234, 'train_common':1157, 'BBBP_test':291,
            'scaffold_test_raw':152, 'scaffold_test':137, 'reference_out':2607}
EVALS = ('BBBP_test','scaffold_test','reference_out')
ALLSETS = ('train_pool','train_common',*EVALS)
PROTOCOL = 'BBB_eval_identity_overlap_audit_v1'


def fail(reason):
    raise RuntimeError('[EVAL SET AUDIT ERROR] '+str(reason))


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()


def require_file(p):
    p=Path(p).expanduser().resolve()
    if not p.is_file():fail(f'Missing file: {p}')
    return p


def json_read(p):
    return json.loads(require_file(p).read_text(encoding='utf-8'))


def json_write(p, obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def read_csv(p):
    return pd.read_csv(require_file(p),dtype=str,keep_default_na=False,encoding='utf-8-sig',low_memory=False)


def require_fields(df, fields, name):
    miss=set(fields)-set(df.columns)
    if miss:fail(f'{name}: missing fields {sorted(miss)}')


def ensure_unique(df,field,name):
    if (df[field].eq('').any() or df[field].duplicated().any()):
        fail(f'{name}: missing or duplicate {field}; audit refuses to deduplicate identities')


def valid_binary(s):
    return pd.to_numeric(s,errors='coerce').isin([0,1])


def count_exact(n,expect,name):
    if n!=expect:fail(f'{name}: expected {expect}, got {n}; no rows altered')


def resolve_sources(stage5_run=None,stage3_run=None,stage4_run=None,scaffold_csv=None):
    if stage5_run is None:
        ptr=require_file(STAGE5_PROJECT/'CURRENT_RUN.txt')
        stage5_run=Path(ptr.read_text(encoding='utf-8').strip())
    s5=Path(stage5_run).resolve()
    p5=json_read(s5/'precheck.json')
    if p5.get('status')!='PASS_STAGE5_PRECHECK':fail('Stage5 precheck has not passed')
    s3=Path(stage3_run or p5['stage3_run']).resolve()
    s4=Path(stage4_run or p5['stage4_run']).resolve()
    if s3!=Path(p5['stage3_run']).resolve() or s4!=Path(p5['stage4_run']).resolve():
        fail('Requested Stage3/Stage4 run differs from verified Stage5 lineage')
    if not (s3/'PASS_PRECHECK.txt').is_file():fail('Stage3 PASS_PRECHECK.txt missing')
    if not (s4/'STATS_READY.txt').is_file():fail('Stage4 STATS_READY.txt missing')
    if not (s4/'RESULT_STATUS.txt').is_file() or s4.joinpath('RESULT_STATUS.txt').read_text().strip()!='COMPLETE_STATS_AND_PAMPA':
        fail('Stage4 is not COMPLETE_STATS_AND_PAMPA')
    config_path=require_file(s3/'run_config.json'); config=json_read(config_path)
    if config.get('protocol')!='BBB_Stage3_controlled_v1':fail('Unexpected Stage3 protocol')
    if file_sha256(config_path)!=p5.get('stage3_run_config_sha256'):
        fail('Stage3 config differs from frozen Stage5 audited config')
    con=require_file(config['source']['consensus'])
    if file_sha256(con)!=config['source']['hashes']['consensus'] or file_sha256(con)!=p5['frozen_consensus_sha256']:
        fail('Frozen consensus SHA256 mismatch; do not use another data version')
    s4mol=require_file(s4/'analysis/molecule_target_sensitivity.csv')
    if file_sha256(s4mol)!=p5['stage4_molecule_sha256']:
        fail('Stage4 molecule-level output SHA256 differs from verified Stage5')
    source_s2=Path(config['source']['stage2']).resolve()
    vec=require_file(source_s2/'stage2_target_vectors_common1157.csv')
    if file_sha256(vec)!=config['source']['hashes']['stage2_target_vectors_common1157.csv']:
        fail('Stage2 training target vectors SHA256 mismatch')
    sc=require_file(scaffold_csv or s3/'scaffold_split_assignments.csv')
    return {
      'stage5_run':s5,'stage3_run':s3,'stage4_run':s4,'consensus':con,
      'scaffold':sc,'stage4_molecule':s4mol,'stage2_target_vectors':vec,
      'stage3_config':config_path,'stage5_precheck':s5/'precheck.json',
      'stage3_cfg':config,
    }


def compute_audit(con,sc,stage4,vec):
    require_fields(con,('parent_ik','split','in_reference_out','bbbp_label','b3db_label'),'Frozen consensus')
    require_fields(sc,('parent_ik','scaffold_split','b3db_label'),'Stage3 scaffold assignments')
    require_fields(vec,('parent_ik',),'Stage2 target vectors')
    require_fields(stage4,('parent_ik','encoder','evaluation'),'Stage4 molecule output')
    ensure_unique(con,'parent_ik','Frozen consensus')
    ensure_unique(sc,'parent_ik','Stage3 scaffold assignments')
    ensure_unique(vec,'parent_ik','Stage2 vectors')
    if stage4.duplicated(['parent_ik','encoder','evaluation']).any() or stage4.parent_ik.eq('').any():
        fail('Stage4 duplicate/missing parent identity within an encoder and evaluation')
    count_exact(len(con),4131,'Frozen consensus parent count')
    if set(stage4.encoder)!= {'xgb','gine'} or set(stage4.evaluation)!=set(EVALS):
        fail('Stage4 contains unexpected encoder or evaluation names')
    ref_flag=pd.to_numeric(con.in_reference_out,errors='coerce')
    if (ref_flag.isna() | ~ref_flag.isin([0,1])).any():fail('in_reference_out must contain only literal 0/1')
    tr_pool=con.loc[con.split.eq('train') & ref_flag.eq(0)].copy()
    count_exact(len(tr_pool),EXPECTED['train_pool'],'Training pool')
    common=set(vec.parent_ik)
    count_exact(len(common),EXPECTED['train_common'],'True model training set')
    if not common.issubset(set(tr_pool.parent_ik)):
        fail('Stage2 1157 parents not a subset of frozen train pool 1234')
    
    bb_raw=con.loc[con.split.eq('test')].copy()
    count_exact(len(bb_raw),291,'BBBP split==test before label filtering')
    bb=bb_raw.loc[valid_binary(bb_raw.bbbp_label)].copy()
    sc_raw=sc.loc[sc.scaffold_split.eq('test')].copy()
    count_exact(len(sc_raw),152,'Scaffold raw test assignments')
    sf=sc_raw.loc[valid_binary(sc_raw.b3db_label)].copy()
    rf_raw=con.loc[ref_flag.eq(1)].copy()
    rf=rf_raw.loc[valid_binary(rf_raw.b3db_label)].copy()
    eval_frames={'BBBP_test':bb,'scaffold_test':sf,'reference_out':rf}
    sets={'train_pool':set(tr_pool.parent_ik),'train_common':common}
    for e in EVALS:
        frame=eval_frames[e]
        ensure_unique(frame,'parent_ik',e)
        count_exact(len(frame),EXPECTED[e],e)
        if not set(frame.parent_ik).issubset(set(con.parent_ik)):
            fail(f'{e}: parent ID absent from frozen consensus')
        sets[e]=set(frame.parent_ik)
    
    idx=con.set_index('parent_ik')
    aligned=idx.reindex(sc.parent_ik)
    if aligned.split.isna().any():fail('Scaffold assignment contains unknown parent IDs')
    a=pd.to_numeric(sc.b3db_label,errors='coerce').reset_index(drop=True)
    b=pd.to_numeric(aligned.b3db_label,errors='coerce').reset_index(drop=True)
    label_matches=(a.isna() & b.isna()) | (a==b)
    if not bool(label_matches.all()):fail('Scaffold B3DB labels mismatch frozen consensus')
    
    if not (pd.to_numeric(aligned.in_reference_out,errors='coerce')==0).all():
        fail('Scaffold assignments include reference-out identities contrary to original Stage3 split procedure')
    checks=[]
    for enc in ('xgb','gine'):
        for ev in EVALS:
            observed=set(stage4.loc[(stage4.encoder==enc)&(stage4.evaluation==ev),'parent_ik'])
            expected=sets[ev]
            checks.append({'encoder':enc,'evaluation':ev,'stage3_frozen_n':len(expected),
                           'stage4_rows':int(((stage4.encoder==enc)&(stage4.evaluation==ev)).sum()),
                           'stage4_distinct_n':len(observed),
                           'only_stage3_n':len(expected-observed),'only_stage4_n':len(observed-expected),
                           'exact_parent_set_match':observed==expected})
            if observed!=expected or len(observed)!=len(expected):
                fail(f'Stage4 {enc}/{ev} identity differs from Stage3 frozen evaluation')
    overlaps=[]; matrix=[]
    for a_name in ALLSETS:
        row={'set':a_name}
        for b_name in ALLSETS:
            row[b_name]=len(sets[a_name]&sets[b_name])
        matrix.append(row)
        for b_name in ALLSETS:
            if ALLSETS.index(a_name)>=ALLSETS.index(b_name):continue
            a=sets[a_name]; b=sets[b_name]; i=len(a&b)
            overlaps.append({'set_a':a_name,'set_b':b_name,'n_a':len(a),'n_b':len(b),
                 'overlap_parent_n':i,'pct_of_a':round(100*i/len(a),5),
                 'pct_of_b':round(100*i/len(b),5),
                 'jaccard':round(i/len(a|b),7)})
    train_overlap={t:{ev:len(sets[t]&sets[ev]) for ev in EVALS} for t in ('train_pool','train_common')}
    if any(v for row in train_overlap.values() for v in row.values()):
        fail('TRAINING-TO-EVALUATION PARENT LEAKAGE detected; do not declare audit pass')
    union=set.union(*(sets[e] for e in EVALS))
    patterns=[]
    membership=[]
    for key in sorted(union):
        row={'parent_ik':key,'BBBP_test':int(key in sets['BBBP_test']),
             'scaffold_test':int(key in sets['scaffold_test']),
             'reference_out':int(key in sets['reference_out']),
             'split':idx.at[key,'split'],'in_reference_out':idx.at[key,'in_reference_out']}
        membership.append(row)
    mdf=pd.DataFrame(membership)
    for mask in range(1,8):
        bflags=[int(bool(mask & (1<<i))) for i in range(3)]
        q=(mdf.BBBP_test==bflags[0])&(mdf.scaffold_test==bflags[1])&(mdf.reference_out==bflags[2])
        patterns.append({'BBBP_test':bflags[0],'scaffold_test':bflags[1],'reference_out':bflags[2],
                         'n_distinct_parents':int(q.sum())})
    both=sorted(sets['BBBP_test']&sets['reference_out'])
    brow=idx.reindex(both)
    comparable=valid_binary(brow.bbbp_label)&valid_binary(brow.b3db_label)
    agree=(brow.bbbp_label==brow.b3db_label)&comparable
    rows=[]
    for parent,br,comp,eq in zip(both,brow.itertuples(),comparable.to_numpy(),agree.to_numpy()):
        rows.append({'parent_ik':parent,'BBBP_label':br.bbbp_label,'B3DB_label':br.b3db_label,
                     'both_labels_binary':bool(comp),
                     'labels_agree_if_comparable':bool(eq) if comp else ''})
    agreement={'shared_parent_n':len(both),'both_labels_binary_n':int(comparable.sum()),
               'labels_agree_n':int(agree.sum()),'labels_disagree_n':int((comparable&~agree).sum()),
               'not_comparable_n':int((~comparable).sum()),
               'note':'BBBP and B3DB are different label sources; mismatch is not identity mismatch.'}
    return {'sets':sets,'matrix':pd.DataFrame(matrix),'pairwise':pd.DataFrame(overlaps),
            'membership':mdf,'patterns':pd.DataFrame(patterns),'checks':pd.DataFrame(checks),
            'shared_labels':pd.DataFrame(rows),'label_agreement':agreement,
            'train_eval':train_overlap,
            'scaffold_test_raw_n':len(sc_raw),'scaffold_test_valid_n':len(sf),
            'bbbp_test_old_stage5_filter_would_exclude_n':int((ref_flag.loc[bb_raw.index]==1).sum()),
            'evaluation_union_n':len(union)}


def make_summary(o):
    pairs={(r.set_a,r.set_b):r.overlap_parent_n for r in o['pairwise'].itertuples()}
    def value(a,b):return int(pairs.get((a,b),pairs.get((b,a),0)))
    la=o['label_agreement'];tr=o['train_eval']
    lines=[
      '# Evaluation-set identity and overlap audit','',
      '**Status: PASS_WITH_EVALUATION_OVERLAP** — Molecular identity and source checks pass, with no parent-identity overlap between training and evaluation sets. Evaluation sets can overlap with one another.','',
      '## 1. Frozen set definitions','',
      '- Common supervised training set: 1,157 parent IDs from the Stage II target vectors.',
      '- Split-train pool: 1,234 parent IDs with `split=train AND in_reference_out=0` in the frozen consensus; 77 are not included in the common supervised training set.',
      '- BBBP test: 291 parents with `split=test` and a valid `bbbp_label`; no additional reference-out exclusion is applied.',
      '- Scaffold test: 137 parents with a valid `b3db_label` in the saved Stage III scaffold assignments (152 raw assignments).',
      '- Reference-out: 2,607 parents with `in_reference_out=1` and a valid `b3db_label`.','',
      '## 2. Evaluation-set overlap by parent identity','',
      '| Set intersection | Shared parents |', '|---|---:|',
      f'| BBBP test ∩ Reference-out | {value("BBBP_test","reference_out")} |',
      f'| BBBP test ∩ Scaffold test | {value("BBBP_test","scaffold_test")} |',
      f'| Scaffold test ∩ Reference-out | {value("scaffold_test","reference_out")} |',
      f'| Three-way evaluation intersection | {len(o["sets"]["BBBP_test"] & o["sets"]["scaffold_test"] & o["sets"]["reference_out"])} |',
      f'| Distinct parents in the evaluation-set union | {o["evaluation_union_n"]} |','',
      '## 3. Training/evaluation overlap by parent identity','',
      '| Training set | BBBP test | Scaffold test | Reference-out |', '|---|---:|---:|---:|',
      f'| 1,157 common supervised training parents | {tr["train_common"]["BBBP_test"]} | {tr["train_common"]["scaffold_test"]} | {tr["train_common"]["reference_out"]} |',
      f'| 1,234 split-train pool parents | {tr["train_pool"]["BBBP_test"]} | {tr["train_pool"]["scaffold_test"]} | {tr["train_pool"]["reference_out"]} |','',
      '## 4. Cross-database binary labels among shared parents','',
      f'- Parents shared by BBBP test and Reference-out: {la["shared_parent_n"]}.',
      f'- Parents with valid binary labels from both BBBP and B3DB: {la["both_labels_binary_n"]}.',
      f'- Source-label agreement: {la["labels_agree_n"]}; disagreement: {la["labels_disagree_n"]}; not comparable: {la["not_comparable_n"]}.',
      '- These counts describe agreement between two label sources; source-label disagreement is separate from identity mismatch or training leakage.','',
      '## 5. Molecule-level cross-check against Stage IV predictions','',
      '- XGBoost and GINE parent IDs match the frozen Stage III identity sets in all three evaluation settings.',
      f'- Applying an additional reference-out exclusion to BBBP test would remove {o["bbbp_test_old_stage5_filter_would_exclude_n"]} parents and is not part of the frozen BBBP test definition.',
      '- Audit inputs are read-only; the audit does not modify frozen partitions or remove shared molecules across evaluation definitions.','',
      '## 6. Interpretation','',
      'BBBP test, scaffold test, and reference-out are constructed from different evaluation definitions and label-availability rules, so their parent memberships are not forced to be mutually exclusive. Metrics are computed within each evaluation set and endpoint. The training sets have zero parent-identity overlap with all three evaluation sets.','',
      'Parent-identity overlap between evaluation sets does not imply duplicated experimental measurements, and it does not imply training leakage. Evaluation labels can originate from different sources and are not assumed to represent an identical biological measurement.',''
    ]
    return '\n'.join(lines)


def audit(args):
    out=Path(args.out).expanduser().resolve()
    if out.exists() and any(f.name != 'audit.log' for f in out.iterdir()):fail('Output directory must be empty/new: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    sources=resolve_sources(args.stage5_run,args.stage3_run,args.stage4_run,args.scaffold)
    
    for k in ('consensus','scaffold','stage4_molecule','stage2_target_vectors','stage3_config','stage5_precheck'):
        if out==sources[k] or out in sources[k].parents:fail('--out must not be an input directory')
    con=read_csv(sources['consensus']);sc=read_csv(sources['scaffold'])
    stage4=read_csv(sources['stage4_molecule']);vec=read_csv(sources['stage2_target_vectors'])
    result=compute_audit(con,sc,stage4,vec)
    mapping={
      'overlap_matrix.csv':result['matrix'],
      'overlap_pairwise.csv':result['pairwise'],
      'evaluation_overlap_patterns.csv':result['patterns'],
      'evaluation_membership_by_parent.csv':result['membership'],
      'stage4_prediction_identity_checks.csv':result['checks'],
      'BBBP_reference_out_shared_parent_label_comparison.csv':result['shared_labels'],
    }
    for fn,df in mapping.items():df.to_csv(out/fn,index=False,encoding='utf-8-sig')
    (out/'EVAL_SET_AUDIT_SUMMARY.md').write_text(make_summary(result),encoding='utf-8')
    manifest={'protocol':PROTOCOL,'status':'PASS_WITH_EVALUATION_OVERLAP',
       'generated_utc':datetime.now(timezone.utc).isoformat(),
       'counts':{k:len(v) for k,v in result['sets'].items()},
       'scaffold_test_raw_n':result['scaffold_test_raw_n'],
       'evaluation_union_n':result['evaluation_union_n'],
       'stage5_old_filter_excluded_n':result['bbbp_test_old_stage5_filter_would_exclude_n'],
       'train_eval_intersections':result['train_eval'],
       'cross_source_label_comparison':result['label_agreement'],
       'input_provenance':{k:{'path':str(sources[k]),'sha256':file_sha256(sources[k])} for k in
          ('consensus','scaffold','stage4_molecule','stage2_target_vectors','stage3_config','stage5_precheck')},
       'stage3_run':str(sources['stage3_run']),'stage4_run':str(sources['stage4_run']),
       'stage5_run':str(sources['stage5_run']),
       'out_files':{k:{'n_rows':len(v),'sha256':file_sha256(out/k)} for k,v in mapping.items()},
       'note':'No source file edited; no models trained; evaluation overlaps are retained.'}
    json_write(out/'EVAL_SET_AUDIT_REPORT.json',manifest)
    (out/'EVAL_SET_AUDIT_READY.txt').write_text('PASS_WITH_EVALUATION_OVERLAP\n',encoding='utf-8')
    print('[AUDIT PASS] Frozen source and Stage4 XGB/GINE parents match.',flush=True)
    print('[TRAIN/EVAL] All training-to-evaluation parent intersections are 0.',flush=True)
    print('[OVERLAP] BBBP vs reference-out =',len(result['sets']['BBBP_test']&result['sets']['reference_out']),flush=True)
    print('[OVERLAP] BBBP vs scaffold =',len(result['sets']['BBBP_test']&result['sets']['scaffold_test']),flush=True)
    print('[OVERLAP] scaffold vs reference-out =',len(result['sets']['scaffold_test']&result['sets']['reference_out']),flush=True)
    print('[SAVED]',out,flush=True)


def precheck(args):
    s=resolve_sources(args.stage5_run,args.stage3_run,args.stage4_run,args.scaffold)
    print('[PASS_EVAL_AUDIT_INPUTS] Verified Stage5 lineage and SHA256 of frozen files.',flush=True)
    for k in ('consensus','scaffold','stage4_molecule','stage2_target_vectors'):
        print(k,':',s[k],flush=True)


def pack(args):
    o=Path(args.out).expanduser().resolve()
    if not (o/'EVAL_SET_AUDIT_READY.txt').is_file():fail('Audit success marker missing; refuse to pack incomplete result')
    p=o/'BBB_EVAL_SET_AUDIT_REPORT_TO_SEND.zip'
    files=['EVAL_SET_AUDIT_SUMMARY.md','EVAL_SET_AUDIT_REPORT.json','overlap_matrix.csv','overlap_pairwise.csv',
           'evaluation_overlap_patterns.csv','evaluation_membership_by_parent.csv',
           'stage4_prediction_identity_checks.csv',
           'BBBP_reference_out_shared_parent_label_comparison.csv','EVAL_SET_AUDIT_READY.txt','audit.log']
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:
            f=o/name
            if f.is_file():z.write(f,arcname=name)
    print('[REPORT]',p,p.stat().st_size,'bytes',flush=True)


def main():
    ap=argparse.ArgumentParser(description='Frozen Stage3/Stage4 evaluation identity overlap audit; read-only.')
    ap.add_argument('action',choices=['precheck','audit','pack'])
    ap.add_argument('--out',default='')
    ap.add_argument('--stage5-run',default=None)
    ap.add_argument('--stage3-run',default=None)
    ap.add_argument('--stage4-run',default=None)
    ap.add_argument('--scaffold',default=None)
    args=ap.parse_args()
    try:
        if args.action=='precheck':precheck(args)
        elif args.action=='audit':
            if not args.out:fail('audit requires --out directory')
            audit(args)
        elif args.action=='pack':
            if not args.out:fail('pack requires --out directory')
            pack(args)
    except Exception as e:
        print('[AUDIT FAILED]',str(e),file=sys.stderr,flush=True)
        traceback.print_exc()
        if args.out and args.action=='audit':
            Path(args.out).mkdir(parents=True,exist_ok=True)
            Path(args.out,'FAILURE.txt').write_text(str(e)+'\n',encoding='utf-8')
        return 2
    return 0


if __name__=='__main__':sys.exit(main())
