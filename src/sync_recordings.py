"""Align independent feeds using team-labelled player geometry, never ground truth."""
import csv,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]


def load(source):
    base=ROOT/'data'/source
    r=ET.parse(base/f'BAR-ATH_{source}_atd.xml').getroot()
    home=r.find('.//Score').attrib['idLocalTeam']
    teams={p.attrib['id']:0 if p.attrib['teamId']==home else 1 for p in r.findall('.//Players/Player')}
    channels={p.attrib['id']:p.attrib['playerId'] for p in r.findall('.//PlayerChannel')}
    slots=[teams[channels[p.attrib['playerChannelId']]] for p in r.findall('.//DataFormatSpecification//PlayerChannelRef') if p.attrib['playerChannelId'].endswith('_x')]
    raw=base/f'BAR-ATH_{source}_atd.txt'
    frames=[];balls=[]
    with raw.open() as f:
        for line in f:
            _,players,ball=line.rstrip().split(':')
            arr=np.fromstring(players.replace(';',','),sep=',').reshape(-1,3)[:,:2]
            frames.append(arr*np.array([105.,68.]))
            balls.append(np.fromstring(ball,sep=',')[:2]*[105.,68.])
    return np.array(frames),np.array(balls),np.array(slots)


def features(players,teams,step=25):
    ids=[];rows=[]
    for i in range(0,len(players),step):
        p=players[i];row=[];valid=True
        for team in (0,1):
            a=p[(teams==team)&np.isfinite(p).all(axis=1)]
            if len(a)<5:valid=False;break
            row.extend(np.quantile(a,[.1,.3,.5,.7,.9],axis=0).ravel())
        if valid:ids.append(i);rows.append(row)
    return np.array(ids),np.array(rows)


def geometry(a,b,ta,tb):
    distances=[]
    for team in (0,1):
        aa=a[(ta==team)&np.isfinite(a).all(axis=1)]
        bb=b[(tb==team)&np.isfinite(b).all(axis=1)]
        if min(len(aa),len(bb))<5:return 100.,0
        d=np.linalg.norm(aa[:,None,:]-bb[None,:,:],axis=2)
        distances.extend(d.min(axis=1));distances.extend(d.min(axis=0))
    d=np.array(distances)
    return float(np.quantile(d,.65)),int((d<2).sum())


def main():
    tp,tball,tt=load('tactical');bp,bball,bt=load('broadcast')
    ti,tf=features(tp,tt);bi,bf=features(bp,bt)
    print('Feature frames',len(ti),len(bi),flush=True)
    tree=cKDTree(tf)
    _,neighbors=tree.query(bf,k=16)
    anchors=[]
    for j,b in enumerate(bi):
        candidates=[]
        for ix in neighbors[j]:
            t=ti[ix]
            error,n=geometry(tp[t],bp[b],tt,bt)
            candidates.append((error,int(t),n))
        candidates.sort()
        err,t,n=candidates[0]
        # Distinct temporal alternative, rather than adjacent near-duplicate frames.
        alternative=min((e for e,t2,n2 in candidates if abs(t2-t)>125),default=100)
        if err<2.5 and n>=18 and alternative>err+1:
            anchors.append((int(b),t,err,n))
    np.savez_compressed(ROOT/'outputs/sync_cache.npz',tp=tp,tball=tball,tt=tt,bp=bp,bball=bball,bt=bt,anchors=np.array(anchors))
    print('Candidate anchors',len(anchors),flush=True)
    from collections import Counter
    print('Offset bins:',Counter(round((t-b)/25) for b,t,e,n in anchors).most_common(20),flush=True)
    with (ROOT/'outputs/sync_anchors.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['broadcast_frame','tactical_frame','geometry_error_m','matched_distances']);w.writerows(anchors)


if __name__=='__main__':main()
