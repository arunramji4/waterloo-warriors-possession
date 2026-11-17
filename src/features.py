import csv, math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from goalkeeper_teams import goalkeeper_ids
ROOT=Path(__file__).resolve().parents[1]
STRIDE=5
HALVES=[(11185,90652),(110808,190198)]

def infer_keeper_ends(slots,teams,keepers):
    # Infer each half's defending ends from the majority of provider-labelled
    # goalkeeper observations. No human-set orientation or frame corrections.
    votes=np.zeros((2,2),dtype=np.int64)
    for frame,line in enumerate((ROOT/'data/tactical/BAR-ATH_tactical_atd.txt').open()):
        if frame%25:continue
        half=next((i for i,(a,b) in enumerate(HALVES) if a<=frame<b),None)
        if half is None:continue
        for slot,record in zip(slots,line.rstrip().split(':')[1].split(';')):
            if slot not in keepers:continue
            x,y,_=map(float,record.split(','))
            if math.isfinite(x) and 0<=x<=1 and .2<y<.8 and (x<.2 or x>.8):
                home_left=(teams[slot]=='H')==(x<.2)
                votes[half,0 if home_left else 1]+=1
    if any(a==b for a,b in votes):raise ValueError('Ambiguous defending-end evidence')
    result=[bool(a>b) for a,b in votes]
    print('Automatic home-defends-left by half:',result,flush=True)
    return result

def corrected_team(frame,x,y,metadata_team,is_goalkeeper,orientation):
    if not is_goalkeeper:return metadata_team
    half=next((i for i,(a,b) in enumerate(HALVES) if a<=frame<b),None)
    if half is None or not(math.isfinite(x) and math.isfinite(y)):return metadata_team
    if not(0<=x<=1 and .2<y<.8) or .2<=x<=.8:return metadata_team
    return 'H' if (x<.2)==orientation[half] else 'A'

def make_features(supplement_file='outputs/broadcast_ball_supplement.csv'):
    root=ET.parse(ROOT/'data/tactical/BAR-ATH_tactical_atd.xml').getroot()
    home=root.find('.//Score').attrib['idLocalTeam']
    teams={p.attrib['id']:'H' if p.attrib['teamId']==home else 'A' for p in root.findall('.//Players/Player')}
    channels={p.attrib['id']:p.attrib['playerId'] for p in root.findall('.//PlayerChannel')}
    slots=[channels[p.attrib['playerChannelId']] for p in root.findall('.//DataFormatSpecification//PlayerChannelRef') if p.attrib['playerChannelId'].endswith('_x')]
    keepers=goalkeeper_ids(root)
    orientation=infer_keeper_ends(slots,teams,keepers)
    extra={int(r['tactical_frame']):(float(r['broadcast_ball_x_m'])/105,float(r['broadcast_ball_y_m'])/68) for r in csv.DictReader((ROOT/supplement_file).open())}
    raw_players=[];raw_ball=[];team_ids=[];sources=[];missing_age=[];age=0
    for frame,line in enumerate((ROOT/'data/tactical/BAR-ATH_tactical_atd.txt').open()):
        if frame%STRIDE:continue
        _,players,ball=line.rstrip().split(':')
        p=np.fromstring(players.replace(';',','),sep=',',dtype=np.float32).reshape(-1,3)
        x,y,_=map(float,ball.split(','));source=0
        if math.isfinite(x) and math.isfinite(y):source=1
        elif frame in extra:x,y=extra[frame];source=2
        else:x=y=float('nan')
        valid=source!=0
        age=0 if valid else age+.2
        raw_players.append(p[:,:2]);raw_ball.append([x,y]);sources.append(source);missing_age.append(min(age,10)/10)
        team_ids.append([0 if corrected_team(frame,float(p[j,0]),float(p[j,1]),teams[s],s in keepers,orientation)=='H' else 1 for j,s in enumerate(slots)])
    p=np.array(raw_players,dtype=np.float32);b=np.array(raw_ball,dtype=np.float32);team=np.array(team_ids,dtype=np.int64)
    mask=np.isfinite(p).all(-1);bm=np.isfinite(b).all(-1)
    pv=np.zeros_like(p);bv=np.zeros_like(b)
    pv[1:]=(p[1:]-p[:-1])*np.array([105,68],dtype=np.float32)/.2
    bv[1:]=(b[1:]-b[:-1])*np.array([105,68],dtype=np.float32)/.2
    pmove=np.isfinite(pv).all(-1)&(np.linalg.norm(pv,axis=-1)<=15)
    bmove=np.isfinite(bv).all(-1)&(np.linalg.norm(bv,axis=-1)<=50)
    pv[~pmove]=0;bv[~bmove]=0
    rel=(p-b[:,None,:]);rel[~bm]=0
    distance=np.linalg.norm(rel*np.array([105,68],dtype=np.float32),axis=-1)/20
    gk=np.broadcast_to(np.array([s in keepers for s in slots],dtype=np.float32),mask.shape)
    player=np.concatenate([p*2-1,np.clip(pv/10,-3,3),gk[...,None],rel,
        np.clip(distance,0,6)[...,None],np.broadcast_to(bm[:,None],mask.shape)[...,None],
        pmove[...,None]],axis=-1).astype(np.float32)
    player=np.nan_to_num(player,nan=0,posinf=0,neginf=0);player[~mask]=0
    scene=np.concatenate([b*2-1,np.clip(bv/30,-3,3),bm[:,None],bmove[:,None],
        (np.array(sources)==2)[:,None],np.array(missing_age,dtype=np.float32)[:,None],
        np.minimum(b[:,1],1-b[:,1])[:,None],np.minimum(b[:,0],1-b[:,0])[:,None]],axis=-1).astype(np.float32)
    scene=np.nan_to_num(scene,nan=0,posinf=0,neginf=0)
    for i in np.flatnonzero(~bm):scene[i,[0,1,8,9]]=0
    scene[:,6]=0
    return player,mask,team,scene

if __name__=='__main__':
    arrays=make_features()
    np.savez_compressed(ROOT/'outputs/features.npz',**dict(zip(('player','mask','team','scene'),arrays)))
    print('Observation-only features built; no possession labels read.',flush=True)
