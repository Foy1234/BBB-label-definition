# legacy_gine_cpu.py — Defines atom-bond graph encoding and the three-layer GINE encoder, prediction head, training loop and early-stopping protocol used in the published controlled model comparisons.
from __future__ import annotations
import argparse, json, platform, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem, RDLogger, rdBase
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
SEEDS10 = [42,52,62,72,82,92,102,112,122,132]
RDLogger.DisableLog('rdApp.*')
HIDDEN_DIM=64; NUM_LAYERS=3; DROPOUT=.20; LR=1e-3; BATCH_SIZE=64; MAX_EPOCHS=120; PATIENCE=12

def atom_values(a): return {'atomic_num':str(a.GetAtomicNum()),'degree':str(a.GetDegree()),'formal_charge':str(a.GetFormalCharge()),'hybridization':str(a.GetHybridization()),'chirality':str(a.GetChiralTag()),'aromatic':str(int(a.GetIsAromatic())),'total_h':str(a.GetTotalNumHs())}
def bond_values(b): return {'bond_type':str(b.GetBondType()),'conjugated':str(int(b.GetIsConjugated())),'in_ring':str(int(b.IsInRing()))}
def make_vocabularies(smiles):
    av={k:set() for k in atom_values(Chem.MolFromSmiles('C').GetAtomWithIdx(0))}; bv={k:set() for k in bond_values(Chem.MolFromSmiles('CC').GetBondWithIdx(0))}
    for s in smiles:
      m=Chem.MolFromSmiles(str(s));
      if m is None: continue
      for a in m.GetAtoms():
        for k,v in atom_values(a).items(): av[k].add(v)
      for b in m.GetBonds():
        for k,v in bond_values(b).items(): bv[k].add(v)
    return ({k:{v:i for i,v in enumerate(sorted(vals))} for k,vals in av.items()},{k:{v:i for i,v in enumerate(sorted(vals))} for k,vals in bv.items()})
def one_hot(v,m):
    z=np.zeros(len(m),dtype=np.float32); key=str(v); z[m[key] if key in m else 0]=1.; return z
def build_graph(smiles,am,bm):
    mol=Chem.MolFromSmiles(str(smiles));
    if mol is None: raise ValueError(smiles)
    nodes=[np.concatenate([one_hot(atom_values(a)[k],am[k]) for k in am]) for a in mol.GetAtoms()]
    ep=[]; er=[]; ed=sum(len(v) for v in bm.values())
    for b in mol.GetBonds():
      f=np.concatenate([one_hot(bond_values(b)[k],bm[k]) for k in bm]); i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ep += [[i,j],[j,i]]; er += [f,f]
    x=torch.tensor(np.asarray(nodes),dtype=torch.float32); ei=torch.tensor(ep,dtype=torch.long).t().contiguous() if ep else torch.empty((2,0),dtype=torch.long); ea=torch.tensor(np.asarray(er),dtype=torch.float32) if er else torch.empty((0,ed),dtype=torch.float32)
    return Data(x=x,edge_index=ei,edge_attr=ea)
class Model(nn.Module):
    def __init__(self,nd,ed):
      super().__init__(); self.convs=nn.ModuleList(); self.norms=nn.ModuleList()
      for i in range(NUM_LAYERS):
        ind=nd if i==0 else HIDDEN_DIM; mlp=nn.Sequential(nn.Linear(ind,HIDDEN_DIM),nn.ReLU(),nn.Linear(HIDDEN_DIM,HIDDEN_DIM)); self.convs.append(GINEConv(mlp,edge_dim=ed)); self.norms.append(nn.BatchNorm1d(HIDDEN_DIM))
      self.head=nn.Sequential(nn.Linear(HIDDEN_DIM,32),nn.ReLU(),nn.Dropout(DROPOUT),nn.Linear(32,1))
    def forward(self,d):
      x=d.x
      for c,n in zip(self.convs,self.norms): x=F.dropout(F.relu(n(c(x,d.edge_index,d.edge_attr))),p=DROPOUT,training=self.training)
      return torch.sigmoid(self.head(global_mean_pool(x,d.batch)).view(-1))
def train_model(tg,vg,seed):
    
    torch.manual_seed(seed); np.random.seed(seed); model=Model(tg[0].x.shape[1],tg[0].edge_attr.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4); loader=DataLoader(tg,batch_size=BATCH_SIZE,shuffle=True); best=(float('inf'),None); patience=0; best_epoch=-1
    for epoch in range(MAX_EPOCHS):
      model.train()
      for b in loader:
        opt.zero_grad(); p=model(b); loss=F.mse_loss(p,b.y); loss.backward(); opt.step()
      model.eval(); pp=[]; yy=[]
      with torch.no_grad():
        for b in DataLoader(vg,batch_size=BATCH_SIZE): pp.append(model(b).cpu().numpy()); yy.append(b.y.cpu().numpy())
      vl=float(np.mean((np.concatenate(pp)-np.concatenate(yy))**2))
      if vl<best[0]: best=(vl,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); patience=0; best_epoch=epoch
      else:
        patience+=1
        if patience>=PATIENCE: break
    model.load_state_dict(best[1]); return model,best_epoch,float(best[0])
def predict(model,graphs):
    model.eval(); out=[]
    with torch.no_grad():
      for b in DataLoader(graphs,batch_size=BATCH_SIZE): out.append(model(b).cpu().numpy())
    return np.concatenate(out)

