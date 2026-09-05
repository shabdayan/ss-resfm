#!/usr/bin/env python
"""Add GT-triangulation outlier labels (RESfM Appendix-C style, 4px) to the
Olsson npz, writing datasets/olsson_id/. Olsson ships GT poses but our npz
carry no outliers2; this generates them exactly as build_1dsfm does for
datasets with real GT."""
import glob, os
import numpy as np
OUT="datasets/olsson_id"; os.makedirs(OUT, exist_ok=True)
for f in sorted(glob.glob("datasets/olsson/*.npz")):
    d=dict(np.load(f, allow_pickle=True))
    M=d["M"]; Ps=d["Ps_gt"]
    m=M.shape[0]//2; X=M.reshape(m,2,-1); vis=(X[:,0]!=0)|(X[:,1]!=0)
    n=M.shape[1]; out=np.zeros((m,n),dtype=bool)
    for j in range(n):
        cams=np.where(vis[:,j])[0]
        if len(cams)<2: continue
        A=[]
        for i in cams:
            x,y=X[i,0,j],X[i,1,j]; P=Ps[i]
            A.append(x*P[2]-P[0]); A.append(y*P[2]-P[1])
        A=np.asarray(A)
        _,_,Vt=np.linalg.svd(A, full_matrices=False)
        Xh=Vt[-1]
        for i in cams:
            p=Ps[i]@Xh
            if abs(p[2])<1e-12: out[i,j]=True; continue
            err=np.hypot(p[0]/p[2]-X[i,0,j], p[1]/p[2]-X[i,1,j])
            if err>4.0: out[i,j]=True
    d["outliers2"]=out
    d["outlier_pct"]=np.float64(100.0*(out&vis).sum()/max(1,vis.sum()))
    np.savez(os.path.join(OUT, os.path.basename(f)), **d)
    print(f"{os.path.basename(f)}: {d['outlier_pct']:.2f}% outliers", flush=True)
print("ALL DONE", flush=True)
