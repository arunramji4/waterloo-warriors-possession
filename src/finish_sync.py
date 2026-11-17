"""Refine offsets and retain only geometry-validated cross-feed frame matches."""
import csv,json
from collections import Counter
import numpy as np
from sync_recordings import ROOT,geometry
HALVES=[(11185,90651),(110808,190197)]
c=np.load(ROOT/'outputs/sync_cache.npz');tp,tball,tt,bp,bball,bt=[c[k] for k in ('tp','tball','tt','bp','bball','bt')]
anchors=c['anchors']
offsets=[]
for lo,hi in [(3900,4100),(26800,27050)]:
 selected=[(int(b),int(t)) for b,t,e,n in anchors if lo<=t-b<=hi]
 refined=[]
 for b,t in selected[::max(1,len(selected)//100)]:
  choices=[]
  for d in range(-30,31):
   if 0<=t+d<len(tp):choices.append((geometry(tp[t+d],bp[b],tt,bt)[0],t+d-b))
  refined.append(min(choices)[1])
 offset=Counter(refined).most_common(1)[0][0]
 offsets.append(offset)
 print('Refined offset',offset,'seconds',offset/25,'votes',Counter(refined).most_common(8),flush=True)
valid_h=[]
with (ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv').open() as f:
 for row in csv.DictReader(f):valid_h.append(row['valid']=='1')
valid_h=np.array(valid_h)
mapping=[];supplement=[];overlap_errors=[]
for half,((first,last),offset) in enumerate(zip(HALVES,offsets),1):
 for b in range(max(0,first-offset),min(len(bp),last-offset+1)):
  if not valid_h[b]:continue
  t=b+offset
  err,n=geometry(tp[t],bp[b],tt,bt)
  if err>2.5 or n<18:continue
  tb,bb=tball[t],bball[b]
  tv=bool(np.isfinite(tb).all());bv=bool(np.isfinite(bb).all())
  inbounds=bv and 0<=bb[0]<=105 and 0<=bb[1]<=68
  delta=float(np.linalg.norm(tb-bb)) if tv and bv else None
  if delta is not None:overlap_errors.append(delta)
  # The player match is independent of ball visibility/position.
  row=dict(tactical_frame=t,broadcast_frame=b,half=half,offset_frames=offset,
   player_geometry_error_m=round(err,4),matched_distances=n,
   tactical_ball_visible=int(tv),broadcast_ball_visible=int(bv),
   broadcast_ball_in_bounds=int(inbounds),
   ball_disagreement_m=round(delta,4) if delta is not None else '')
  mapping.append(row)
  if not tv and bv:
   supplement.append(dict(**row,broadcast_ball_x_m=round(float(bb[0]),4),broadcast_ball_y_m=round(float(bb[1]),4)))
 for name,rows in [('recording_alignment',mapping),('broadcast_ball_supplement',supplement)]:
  if rows:
   with (ROOT/f'outputs/{name}.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('Half',half,'cumulative matched',len(mapping),'supplement',len(supplement),flush=True)
errors=np.array(overlap_errors)
summary=dict(offset_frames=offsets,offset_seconds=[o/25 for o in offsets],
 aligned_frames=len(mapping),aligned_seconds=len(mapping)/25,
 extra_ball_frames=len(supplement),extra_ball_seconds=len(supplement)/25,
 common_ball_frames=len(errors),common_ball_median_disagreement_m=float(np.median(errors)),
 common_ball_p90_disagreement_m=float(np.quantile(errors,.9)),
 common_ball_within_3m_fraction=float(np.mean(errors<=3)),
 method='Per-half offsets refined using team-labelled player geometry. Each accepted frame has valid broadcast homography, >=18 close directional player matches, and 65th-percentile symmetric nearest-player distance <=2.5 m. No ball positions or ground truth used to align.',
 warning='Provisional alignment, not a guarantee of correct ball tracking. Broadcast ball supplements are separate and do not overwrite tactical observations.')
(ROOT/'outputs/sync_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
