#!/usr/bin/env python3
"""End-to-end non-DL raw/infra selective hybrid pipeline with conservative infra coarsening.

Pipeline assumption:
- Upstream model gives large-block masks (category 'gra') on RAW images only.
- RAW and INFRA images are paired but not pixel-aligned.
- Final subparts are output in INFRA coordinates.

Main steps:
1. Split each raw large block into subparts with v3 raw graph-superpixel baseline.
2. Estimate raw->infra registration using raw gra mask and an automatically detected infra foreground mask.
3. Warp raw subparts to infra coordinates. This is the default final result.
4. Generate an infra-only candidate in the inferred infra gra region.
5. Selectively switch to infra candidate only when an infra-domain score and complexity checks say raw is likely unreliable.
6. If infra manual subparts exist in COCO, evaluate final masks against infra subparts.

This script intentionally does not use deep learning. It also does not use infra gra/subpart annotations for inference;
those are used only when --eval is set.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_multiotsu
from skimage.segmentation import felzenszwalb, find_boundaries, watershed
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

COLORS = np.array([
    [230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],
    [145,30,180],[70,240,240],[240,50,230],[210,245,60],[250,190,190],
    [0,128,128],[230,190,255],[170,110,40],[255,250,200],[128,0,0],
    [170,255,195],[128,128,0],[255,215,180],[0,0,128],[128,128,128]
], dtype=np.uint8)

@dataclass
class V3Params:
    gaussian_sigma_divisor: float = 7.0
    gaussian_sigma_min: float = 9.0
    contrast_low: float = 5.0
    contrast_high: float = 16.0
    felz_scale_small: float = 60.0
    felz_scale_large: float = 80.0
    felz_sigma: float = 0.4
    felz_min_size_area_divisor: float = 900.0
    felz_min_size_floor: int = 18
    large_area_threshold: int = 12000
    min_component_area_fraction: float = 0.012
    min_component_area_floor: int = 25

@dataclass
class Metrics:
    method: str
    raw_file: str
    infra_file: str
    raw_image_id: int
    infra_image_id: int
    raw_block: int
    infra_block: int
    reg_iou: float
    decision: str
    gt_regions: int
    pred_regions: int
    ari: float
    cover_gt: float
    cover_pred: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    eval_area: int
    raw_regions: int
    infra_candidate_regions: int
    raw_score: float
    infra_score: float
    auto_mask_iou: float

# ---------------- COCO / IO ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--input-zip', type=Path)
    src.add_argument('--dataset-dir', type=Path)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--pair-csv', type=Path, default=None, help='CSV with raw_file,infra_file columns. Recommended for production if acquisition system knows pairs.')
    ap.add_argument('--pair-mode', choices=['csv','filename','order','shape'], default='filename')
    ap.add_argument('--raw-prefix', default='raw_png')
    ap.add_argument('--infra-prefix', default='infra_png')
    ap.add_argument('--big-category', default='gra')
    ap.add_argument('--subpart-category', default='subparts')
    ap.add_argument('--eval', action='store_true', help='Evaluate against infra subparts if available. Evaluation annotations are never used for inference.')
    ap.add_argument('--overview-width', type=int, default=1350)
    ap.add_argument('--skip-overview', action='store_true', help='Do not save JPG panels/contact sheet; useful for fast batch validation.')
    ap.add_argument('--coarsen-infra-output', action='store_true', default=True, help='When selective logic chooses infra, coarsen the infra candidate before output to suppress texture/stacking-induced fragments.')
    ap.add_argument('--no-coarsen-infra-output', dest='coarsen_infra_output', action='store_false')
    return ap.parse_args()

def prepare_dataset(args):
    if args.dataset_dir:
        return args.dataset_dir, None
    tmp = tempfile.TemporaryDirectory(prefix='selective_hybrid_')
    with zipfile.ZipFile(args.input_zip) as z:
        z.extractall(tmp.name)
    return Path(tmp.name), tmp

def find_ann(root: Path) -> Path:
    hits = list(root.rglob('_annotations.coco.json'))
    if not hits: raise FileNotFoundError('No _annotations.coco.json found')
    return sorted(hits, key=lambda p: ('train' not in str(p.parent), len(str(p))))[0]

def load_coco(ann_path: Path):
    data = json.loads(ann_path.read_text())
    images = {int(im['id']): im for im in data['images']}
    anns_by: Dict[int,List[Dict]] = {}
    for a in data.get('annotations', []):
        anns_by.setdefault(int(a['image_id']), []).append(a)
    cats = {c['name']: int(c['id']) for c in data.get('categories', [])}
    return data, images, anns_by, cats

def image_path(root: Path, ann_path: Path, fn: str) -> Path:
    p = ann_path.parent / fn
    if p.exists(): return p
    hits = list(root.rglob(fn))
    if not hits: raise FileNotFoundError(fn)
    return hits[0]

def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))

def poly_mask(h:int, w:int, ann:Dict) -> np.ndarray:
    m = np.zeros((h,w), np.uint8)
    for seg in ann.get('segmentation', []):
        pts = np.array(seg, np.int32).reshape(-1,2)
        if len(pts) >= 3: cv2.fillPoly(m, [pts], 255)
    return m

def masks_for(im:Dict, anns_by:Dict[int,List[Dict]], cat_id:Optional[int]) -> List[np.ndarray]:
    if cat_id is None: return []
    out=[]
    for a in anns_by.get(int(im['id']), []):
        if int(a.get('category_id', -1)) == int(cat_id):
            out.append(poly_mask(im['height'], im['width'], a))
    return out

def union_masks(ms:Sequence[np.ndarray], shape:Tuple[int,int]) -> np.ndarray:
    out=np.zeros(shape, np.uint8)
    for m in ms: out[m>0]=255
    return out

# ---------------- Mask / segmentation utils ----------------
def relabel(label: np.ndarray) -> np.ndarray:
    out=np.zeros_like(label, np.int32)
    labs=[int(x) for x in np.unique(label) if x>0]
    for i,l in enumerate(labs,1): out[label==l]=i
    return out

def fill_zero_inside(label:np.ndarray, inside:np.ndarray)->np.ndarray:
    out=label.copy().astype(np.int32)
    need=(out==0)&inside
    if need.any() and (out>0).any():
        _,inds=distance_transform_edt(out==0, return_indices=True)
        near=out[tuple(inds)]
        out[need]=near[need]
    return out

def bbox(mask:np.ndarray)->Tuple[int,int,int,int]:
    ys,xs=np.nonzero(mask>0)
    if len(xs)==0: return 0,0,1,1
    return int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)

def detect_bad_raw(rgb:np.ndarray)->np.ndarray:
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    red=(r>130)&(g<105)&(b<125)&((r.astype(int)-g.astype(int))>35)&((r.astype(int)-b.astype(int))>30)
    dark=(r<36)&(g<36)&(b<36)
    bad=(red|dark).astype(np.uint8)*255
    bad=cv2.dilate(bad,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
    return bad>0

def normalize_channel(ch, valid):
    p1,p99=np.percentile(ch[valid],[1,99])
    return np.clip((ch-p1)/max(1.0,float(p99-p1)),0,1)

def choose_classes(vals, p:V3Params):
    if len(vals)<80: return 1
    c=float(np.percentile(vals,90)-np.percentile(vals,10))
    if c<p.contrast_low: return 1
    if c<p.contrast_high: return 2
    return 3

def cleanup_classmap(classmap:np.ndarray, inside:np.ndarray, min_area:int)->np.ndarray:
    cm=classmap.astype(np.uint8).copy(); cm[~inside]=0
    out=np.zeros_like(cm,np.int32); nxt=1
    for cls in [int(x) for x in np.unique(cm) if x>0]:
        n,cc,stats,_=cv2.connectedComponentsWithStats((cm==cls).astype(np.uint8),8)
        if n<=1: continue
        areas=stats[1:,cv2.CC_STAT_AREA]
        keep=list(np.where(areas>=min_area)[0]+1)
        if not keep and len(areas): keep=[int(np.argmax(areas)+1)]
        for k in keep:
            out[cc==k]=nxt; nxt+=1
    if out.max()==0: out[inside]=1
    else: out=fill_zero_inside(out,inside)
    return relabel(out)

def segment_raw_v3(rgb:np.ndarray, block_mask:np.ndarray, p:V3Params)->np.ndarray:
    inside=block_mask>0; H,W=inside.shape; area=int(inside.sum())
    out=np.zeros((H,W),np.int32)
    if area<80:
        out[inside]=1; return out
    er=max(1,int(round(math.sqrt(area)/85)))
    inner=cv2.erode(inside.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*er+1,2*er+1)),iterations=1).astype(bool)
    bad=detect_bad_raw(rgb)
    valid=inner&(~bad)
    if valid.sum()<80: valid=inside&(~bad)
    if valid.sum()<80: valid=inside
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    L=lab[:,:,0]
    sigma=max(p.gaussian_sigma_min, min(H,W)/p.gaussian_sigma_divisor)
    bg=cv2.GaussianBlur(L,(0,0),sigmaX=sigma,sigmaY=sigma)
    Lc=L-bg+np.median(L[valid])
    Ls=cv2.bilateralFilter(Lc.astype(np.float32),5,7,7)
    vals=Ls[valid]
    kcls=choose_classes(vals,p)
    if kcls==1:
        out[inside]=1; return out
    fL=normalize_channel(Ls,valid); fa=normalize_channel(lab[:,:,1],valid); fb=normalize_channel(lab[:,:,2],valid)
    feat=np.dstack([fL,0.25*fa,0.25*fb]).astype(np.float32)
    x1,y1,x2,y2=bbox(block_mask); pad=4
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(W,x2+pad); y2=min(H,y2+pad)
    fc=feat[y1:y2,x1:x2].copy(); mc=inside[y1:y2,x1:x2]
    med=np.median(fc[mc],axis=0); fc[~mc]=med
    scale=p.felz_scale_small if area<p.large_area_threshold else p.felz_scale_large
    minsz=max(p.felz_min_size_floor,int(area/p.felz_min_size_area_divisor))
    spc=felzenszwalb(fc,scale=scale,sigma=p.felz_sigma,min_size=minsz).astype(np.int32)+1
    sp=np.zeros((H,W),np.int32); sp[y1:y2,x1:x2]=spc; sp[~inside]=0; sp=relabel(sp)
    try: th=threshold_multiotsu(vals,classes=kcls)
    except Exception: th=np.array([np.median(vals)]) if kcls==2 else np.percentile(vals,[33,67])
    cm=np.zeros((H,W),np.int32)
    for sid in [int(x) for x in np.unique(sp) if x>0]:
        pix=(sp==sid)&inside; vp=pix&(~bad)
        if vp.sum()<3: vp=pix
        cm[pix]=int(np.digitize(float(np.median(Ls[vp])),th)+1)
    cm=fill_zero_inside(cm,inside)
    return cleanup_classmap(cm,inside,max(p.min_component_area_floor,int(p.min_component_area_fraction*area)))

def corrected_infra_intensity(rgb:np.ndarray, mask:np.ndarray)->np.ndarray:
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    valid=mask>0
    if valid.sum()<10: return gray
    sigma=max(10,min(gray.shape)/6)
    bg=cv2.GaussianBlur(gray,(0,0),sigmaX=sigma,sigmaY=sigma)
    corr=gray-bg+np.median(gray[valid])
    return cv2.bilateralFilter(corr.astype(np.float32),5,8,8)

def infra_kmeans_candidate(rgb:np.ndarray, mask:np.ndarray, max_k:int=7)->np.ndarray:
    # Historical name kept for compatibility. Implementation uses multi-Otsu intensity binning
    # plus connected-component cleanup because it is deterministic and faster than KMeans.
    inside=mask>0; H,W=inside.shape; lab=np.zeros((H,W),np.int32)
    area=int(inside.sum())
    if area<100:
        lab[inside]=1; return lab
    I=corrected_infra_intensity(rgb,mask)
    vals=I[inside]
    contrast=float(np.percentile(vals,90)-np.percentile(vals,10))
    if contrast<20:
        lab[inside]=1; return lab
    k=2
    if contrast>40: k=3
    if contrast>60: k=4
    if contrast>80: k=5
    k=min(max_k,k)
    # Use quantile thresholds for speed and stability. Multi-Otsu can be slow on larger masks.
    th=np.percentile(vals, np.linspace(0,100,k+1)[1:-1])
    tmp=np.zeros((H,W),np.int32)
    tmp[inside]=np.digitize(I[inside], th)+1
    min_area=max(35,int(0.015*area))
    return cleanup_classmap(tmp,inside,min_area)

def infra_quantile_candidate(rgb:np.ndarray, mask:np.ndarray, k:int=4, min_area_fraction:float=0.015)->np.ndarray:
    """Fixed-level infra intensity candidate used as evidence for raw-boundary refinement."""
    inside=mask>0; H,W=inside.shape; lab=np.zeros((H,W),np.int32)
    area=int(inside.sum())
    if area<100 or k<=1:
        lab[inside]=1; return lab
    I=corrected_infra_intensity(rgb,mask)
    vals=I[inside]
    th=np.percentile(vals, np.linspace(0,100,k+1)[1:-1])
    tmp=np.zeros((H,W),np.int32)
    tmp[inside]=np.digitize(I[inside], th)+1
    min_area=max(35,int(min_area_fraction*area))
    return cleanup_classmap(tmp,inside,min_area)

# ---------------- Infra gra inference and registration ----------------
def norm_mask(mask:np.ndarray, size:int=96)->np.ndarray:
    ys,xs=np.nonzero(mask>0); canvas=np.zeros((size,size),np.uint8)
    if len(xs)==0: return canvas
    x1,y1,x2,y2=xs.min(),ys.min(),xs.max()+1,ys.max()+1
    crop=mask[y1:y2,x1:x2]
    h,w=crop.shape; sc=(size-8)/max(h,w)
    nw=max(1,int(round(w*sc))); nh=max(1,int(round(h*sc)))
    res=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_NEAREST)
    x=(size-nw)//2; y=(size-nh)//2
    canvas[y:y+nh,x:x+nw]=res
    return canvas

def shape_iou_cost(a:np.ndarray,b:np.ndarray)->float:
    A=norm_mask(a)>0; B=norm_mask(b)>0
    return 1.0-float((A&B).sum()/(((A|B).sum())+1e-9))

def auto_infra_candidates(rgb:np.ndarray, max_cands:int=16)->List[np.ndarray]:
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    H,W=gray.shape
    bl=cv2.GaussianBlur(gray,(0,0),2)
    qs=[10,20,30,40,50,60,70,80,90]
    thresholds=list(np.percentile(bl.ravel(),qs))
    otsu,_img=cv2.threshold(bl,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresholds.append(float(otsu))
    out=[]
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    for t in thresholds:
        for pol in (0,1):
            m=((bl<t) if pol==0 else (bl>t)).astype(np.uint8)*255
            m=cv2.morphologyEx(m,cv2.MORPH_OPEN,k,iterations=1)
            m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k,iterations=2)
            n,cc,stats,_=cv2.connectedComponentsWithStats(m,8)
            comps=[]
            for i in range(1,n):
                area=int(stats[i,cv2.CC_STAT_AREA])
                if area<0.008*H*W or area>0.85*H*W: continue
                comps.append((area,i))
            comps=sorted(comps,reverse=True)[:3]
            for topn in (1,2):
                if len(comps)>=topn:
                    mm=np.zeros((H,W),np.uint8)
                    for _,ci in comps[:topn]: mm[cc==ci]=255
                    # cheap dedupe by Jaccard against stored masks.
                    duplicate=False
                    for u in out:
                        inter=np.logical_and(mm>0,u>0).sum(); union=np.logical_or(mm>0,u>0).sum()
                        if union and inter/union>0.93:
                            duplicate=True; break
                    if not duplicate: out.append(mm)
                    if len(out)>=max_cands: return out
    return out

def component_count(mask:np.ndarray)->int:
    n,cc,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
    return sum(stats[i,cv2.CC_STAT_AREA] > 0.02*mask.size for i in range(1,n))

def choose_best_infra_mask_for_raw(raw_union:np.ndarray, infra_rgb:np.ndarray)->Tuple[np.ndarray,float]:
    cands=auto_infra_candidates(infra_rgb)
    if not cands:
        return np.zeros(infra_rgb.shape[:2],np.uint8), 1e9
    rc=component_count(raw_union)
    best=None; best_cost=1e9
    for m in cands:
        cost=shape_iou_cost(raw_union,m)+0.12*abs(component_count(m)-rc)
        if cost<best_cost:
            best_cost=cost; best=m
    return best, float(best_cost)

def affine_from_bbox(src_mask:np.ndarray, dst_mask:np.ndarray)->np.ndarray:
    sx1,sy1,sx2,sy2=bbox(src_mask); dx1,dy1,dx2,dy2=bbox(dst_mask)
    sw=max(1,sx2-sx1); sh=max(1,sy2-sy1); dw=max(1,dx2-dx1); dh=max(1,dy2-dy1)
    ax=dw/sw; ay=dh/sh
    tx=dx1-ax*sx1; ty=dy1-ay*sy1
    return np.array([[ax,0,tx],[0,ay,ty]],np.float32)

def warp_label(label:np.ndarray, M:np.ndarray, out_shape:Tuple[int,int])->np.ndarray:
    return cv2.warpAffine(label.astype(np.int32),M,(out_shape[1],out_shape[0]),flags=cv2.INTER_NEAREST,borderValue=0).astype(np.int32)

def warp_mask(mask:np.ndarray, M:np.ndarray, out_shape:Tuple[int,int])->np.ndarray:
    return cv2.warpAffine((mask>0).astype(np.uint8)*255,M,(out_shape[1],out_shape[0]),flags=cv2.INTER_NEAREST,borderValue=0)

def registration_iou(pred_mask:np.ndarray, gt_mask:np.ndarray)->float:
    if gt_mask is None or gt_mask.sum()==0: return float('nan')
    a=pred_mask>0; b=gt_mask>0
    return float((a&b).sum()/(((a|b).sum())+1e-9))

# ---------------- Evaluation and scoring ----------------
def label_from_subparts(sub_masks:Sequence[np.ndarray], block_masks:Sequence[np.ndarray], block_idx:int)->Optional[np.ndarray]:
    inside=block_masks[block_idx]>0
    lab=np.zeros_like(block_masks[block_idx],np.int32); cur=1
    for sm in sub_masks:
        ovs=[int(((sm>0)&(bm>0)).sum()) for bm in block_masks]
        if ovs and int(np.argmax(ovs))==block_idx and max(ovs)>0:
            lab[(sm>0)&inside]=cur; cur+=1
    if lab[inside].max()==0: return None
    return relabel(fill_zero_inside(lab,inside))

def covering(gt,pred,inside,from_gt=True):
    A=gt if from_gt else pred; B=pred if from_gt else gt
    regs=[int(x) for x in np.unique(A[inside]) if x>0]
    labsB=[int(x) for x in np.unique(B[inside]) if x>0]
    total=0; acc=0.0
    for r in regs:
        ma=(A==r)&inside; a=int(ma.sum()); total+=a; best=0.0
        for s in labsB:
            mb=(B==s)&inside; inter=int((ma&mb).sum())
            if inter:
                best=max(best,inter/(int((ma|mb).sum())+1e-9))
        acc+=a*best
    return float(acc/(total+1e-9)) if total else float('nan')

def bf1(gt,pred,inside,tol=2):
    gb=find_boundaries(gt,mode='outer')&inside; pb=find_boundaries(pred,mode='outer')&inside
    if gb.sum()==0 and pb.sum()==0: return 1,1,1
    if gb.sum()==0 or pb.sum()==0: return 0,0,0
    dg=distance_transform_edt(~gb); dp=distance_transform_edt(~pb)
    mp=pb&(dg<=tol); mg=gb&(dp<=tol)
    p=float(mp.sum()/(pb.sum()+1e-9)); r=float(mg.sum()/(gb.sum()+1e-9)); f=float(2*p*r/(p+r+1e-9))
    return p,r,f

def eval_metrics(method, raw_file, infra_file, raw_id, infra_id, rb, ib, reg_iou, decision, gt, pred, inside, raw_regions, cand_regions, raw_score, cand_score, auto_iou):
    ytrue=gt[inside].ravel(); ypred=pred[inside].ravel()
    p,r,f=bf1(gt,pred,inside)
    return Metrics(method,raw_file,infra_file,raw_id,infra_id,rb,ib,reg_iou,decision,int(gt.max()),int(pred.max()),float(adjusted_rand_score(ytrue,ypred)),covering(gt,pred,inside,True),covering(gt,pred,inside,False),p,r,f,int(inside.sum()),raw_regions,cand_regions,raw_score,cand_score,auto_iou)

def score_segmentation_in_infra(rgb:np.ndarray, label:np.ndarray, mask:np.ndarray)->float:
    inside=mask>0
    if inside.sum()<50 or label.max()==0: return 1e9
    I=corrected_infra_intensity(rgb,mask)
    vals=I[inside]; global_var=float(np.var(vals)+1e-6)
    # within variance weighted
    within=0.0; small_pen=0.0
    labs=[int(x) for x in np.unique(label[inside]) if x>0]
    for l in labs:
        m=(label==l)&inside; frac=m.sum()/(inside.sum()+1e-9)
        within += frac*float(np.var(I[m])/(global_var))
        if frac<0.05: small_pen += (0.05-frac)*3
    # boundary contrast support
    boundary=find_boundaries(label,mode='outer')&inside
    grad=cv2.magnitude(cv2.Sobel(I,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(I,cv2.CV_32F,0,1,ksize=3))
    bg=float(np.median(grad[inside])+1e-6)
    bgrad=float(np.median(grad[boundary])/(bg)) if boundary.sum()>0 else 0.0
    # region count penalty makes infra conservative.
    count_pen=0.045*max(0,len(labs)-2)
    return within - 0.04*bgrad + small_pen + count_pen

def selective_decision(raw_score:float, cand_score:float, raw_regions:int, cand_regions:int, reg_iou:float, auto_cost:float=0.0)->str:
    # Conservative switching. Two cases trigger infra:
    # 1) raw predicts very few regions while infra has much lower energy;
    # 2) registration/shape matching was hard (high auto_cost), suggesting a complex block,
    #    and infra proposes substantially more structure.
    if raw_regions <= 2 and cand_regions > raw_regions and cand_score < raw_score - 0.25:
        return 'use_infra'
    if auto_cost > 0.18 and cand_regions >= raw_regions + 4:
        return 'use_infra'
    return 'keep_raw'



def pair_by_filename(raws,infras):
    import re
    raw_by={}; infra_by={}
    for r in raws:
        m=re.search(r'(pair[_-]?\d+)', r['file_name'])
        if m: raw_by[m.group(1)]=r['file_name']
    for i in infras:
        m=re.search(r'(pair[_-]?\d+)', i['file_name'])
        if m: infra_by[m.group(1)]=i['file_name']
    keys=sorted(set(raw_by)&set(infra_by))
    if not keys:
        raise ValueError('No pair_XXX tokens found. Use --pair-mode csv or shape.')
    return [(raw_by[k], infra_by[k]) for k in keys]

def boundary_gradient_support(rgb:np.ndarray, label:np.ndarray, mask:np.ndarray)->float:
    inside=mask>0
    if inside.sum()<50 or label.max()==0: return 0.0
    I=corrected_infra_intensity(rgb,mask)
    boundary=find_boundaries(label,mode='outer')&inside
    if boundary.sum()==0: return 0.0
    grad=cv2.magnitude(cv2.Sobel(I,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(I,cv2.CV_32F,0,1,ksize=3))
    return float(np.median(grad[boundary])/(np.median(grad[inside])+1e-6))



def adjacent_label_pairs(label: np.ndarray, inside: np.ndarray) -> set[tuple[int, int]]:
    """Return 4-neighbor adjacent label pairs inside mask."""
    pairs: set[tuple[int, int]] = set()
    a = label[:, :-1]; b = label[:, 1:]
    m = (a != b) & (a > 0) & (b > 0) & inside[:, :-1] & inside[:, 1:]
    for x, y in zip(a[m], b[m]):
        pairs.add(tuple(sorted((int(x), int(y)))))
    a = label[:-1, :]; b = label[1:, :]
    m = (a != b) & (a > 0) & (b > 0) & inside[:-1, :] & inside[1:, :]
    for x, y in zip(a[m], b[m]):
        pairs.add(tuple(sorted((int(x), int(y)))))
    return pairs

def coarsen_candidate_by_adjacency(label: np.ndarray, mask: np.ndarray, target_regions: Optional[int] = None) -> np.ndarray:
    """Reduce infra over-segmentation by merging the smallest adjacent regions.

    This is intentionally intensity-agnostic. The high-sensitivity infra candidate is still
    used as an evidence map for deciding whether the block should be handled by infra, but
    the output mask is coarsened so stacking/texture stripes are less likely to appear as
    separate thickness subparts.

    Default target is ceil(K/2), where K is the candidate region count.
    """
    inside = mask > 0
    lab = relabel(label.copy())
    k = int(lab.max())
    if k <= 1:
        return lab
    if target_regions is None:
        target_regions = max(1, (k + 1) // 2)
    target_regions = max(1, int(target_regions))
    while int(lab.max()) > target_regions:
        labs = [int(x) for x in np.unique(lab[inside]) if x > 0]
        areas = {l: int(((lab == l) & inside).sum()) for l in labs}
        pairs = adjacent_label_pairs(lab, inside)
        if not pairs:
            break
        best_src = None
        best_dst = None
        best_area = 10**18
        for l in labs:
            neigh = [b if a == l else a for a, b in pairs if a == l or b == l]
            if neigh and areas[l] < best_area:
                best_area = areas[l]
                best_src = l
                best_dst = max(neigh, key=lambda n: areas[n])
        if best_src is None or best_dst is None:
            break
        lab[lab == best_src] = best_dst
        lab = relabel(lab)
    return lab

def intersect_label_maps(anchor: np.ndarray, evidence: np.ndarray, mask: np.ndarray, min_area_fraction: float = 0.008) -> np.ndarray:
    """Split anchor labels only where an evidence map also has stable connected support."""
    inside = mask > 0
    out = np.zeros_like(anchor, np.int32)
    if anchor.max() <= 0 or evidence.max() <= 0 or inside.sum() == 0:
        out[inside] = 1
        return out
    cur = 1
    min_area = max(20, int(min_area_fraction * inside.sum()))
    anchor_labs = [int(x) for x in np.unique(anchor[inside]) if x > 0]
    evidence_labs = [int(x) for x in np.unique(evidence[inside]) if x > 0]
    for a in anchor_labs:
        for e in evidence_labs:
            m = (anchor == a) & (evidence == e) & inside
            n, cc, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
            for i in range(1, n):
                if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                    out[cc == i] = cur
                    cur += 1
    if out.max() == 0:
        out[inside] = 1
    else:
        out = fill_zero_inside(out, inside)
    return relabel(out)

def hybrid_raw_infra_candidate(infra_rgb: np.ndarray, raw_label: np.ndarray, mask: np.ndarray, evidence: Optional[np.ndarray] = None) -> np.ndarray:
    """Use infra intensity to move raw boundaries while preserving raw's conservative region count.

    The raw split count is treated as a prior. Infra contributes only local boundary evidence,
    then the result is coarsened back to the raw count so texture bands do not explode the output.
    """
    inside = mask > 0
    raw = relabel(fill_zero_inside(raw_label, inside)) if raw_label.max() > 0 else raw_label
    raw_regions = int(raw.max())
    if raw_regions <= 1:
        out = np.zeros_like(raw, np.int32)
        out[inside] = 1
        return out
    if evidence is None:
        evidence = infra_quantile_candidate(infra_rgb, mask, k=4)
    hybrid = intersect_label_maps(raw, evidence, mask)
    return coarsen_candidate_by_adjacency(hybrid, mask, target_regions=raw_regions)

def corrected_raw_l_channel(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inside = mask > 0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    if inside.sum() < 10:
        return L
    sigma = max(9.0, min(rgb.shape[:2]) / 7.0)
    bg = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)
    corr = L - bg + np.median(L[inside])
    return cv2.bilateralFilter(corr.astype(np.float32), 5, 7, 7)

def warp_float(ch: np.ndarray, M: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(ch.astype(np.float32), M, (out_shape[1], out_shape[0]), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)

def raw_evidence_channels_in_infra(raw_rgb: np.ndarray, raw_mask: np.ndarray, M: np.ndarray, out_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    lab = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    chans = {
        'R': raw_rgb[:, :, 0].astype(np.float32),
        'G': raw_rgb[:, :, 1].astype(np.float32),
        'B': raw_rgb[:, :, 2].astype(np.float32),
        'gray': gray,
        'LabL': lab[:, :, 0],
        'Lcorr': corrected_raw_l_channel(raw_rgb, raw_mask),
    }
    return {name: warp_float(ch, M, out_shape) for name, ch in chans.items()}

def robust_range(ch: np.ndarray, inside: np.ndarray) -> float:
    vals = ch[inside]
    if len(vals) < 10:
        return 1.0
    q10, q90 = np.percentile(vals, [10, 90])
    return max(1.0, float(q90 - q10))

def normalized_gradient_max(channels: Dict[str, np.ndarray], inside: np.ndarray) -> np.ndarray:
    out = np.zeros_like(next(iter(channels.values())), np.float32)
    for ch in channels.values():
        gx = cv2.Sobel(ch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        g = cv2.magnitude(gx, gy)
        denom = float(np.percentile(g[inside], 75)) if inside.sum() else 1.0
        out = np.maximum(out, g / max(1.0, denom))
    return out

def pair_boundary_pixels(label: np.ndarray, inside: np.ndarray) -> Dict[Tuple[int, int], List[Tuple[int, int, int, int]]]:
    pairs: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
    a = label[:, :-1]; b = label[:, 1:]
    m = (a != b) & (a > 0) & (b > 0) & inside[:, :-1] & inside[:, 1:]
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        p = tuple(sorted((int(label[y, x]), int(label[y, x + 1]))))
        pairs.setdefault(p, []).append((int(y), int(x), int(y), int(x + 1)))
    a = label[:-1, :]; b = label[1:, :]
    m = (a != b) & (a > 0) & (b > 0) & inside[:-1, :] & inside[1:, :]
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        p = tuple(sorted((int(label[y, x]), int(label[y + 1, x]))))
        pairs.setdefault(p, []).append((int(y), int(x), int(y + 1), int(x)))
    return pairs

def split_label_parents(split_label: np.ndarray, raw_label: np.ndarray, inside: np.ndarray) -> Dict[int, int]:
    parents: Dict[int, int] = {}
    for lab in [int(x) for x in np.unique(split_label[inside]) if x > 0]:
        vals = raw_label[(split_label == lab) & inside]
        vals = vals[vals > 0]
        if len(vals) == 0:
            parents[lab] = 0
            continue
        labs, counts = np.unique(vals, return_counts=True)
        parents[lab] = int(labs[int(np.argmax(counts))])
    return parents

def boundary_support_stats(
    label: np.ndarray,
    a: int,
    b: int,
    edge_pts: Sequence[Tuple[int, int, int, int]],
    inside: np.ndarray,
    raw_channels: Dict[str, np.ndarray],
    raw_grad_max: np.ndarray,
    infra_grad_norm: np.ndarray,
    band_radius: int = 5,
) -> Dict[str, float]:
    edge = np.zeros_like(label, np.uint8)
    y1 = np.array([p[0] for p in edge_pts], dtype=np.int32)
    x1 = np.array([p[1] for p in edge_pts], dtype=np.int32)
    y2 = np.array([p[2] for p in edge_pts], dtype=np.int32)
    x2 = np.array([p[3] for p in edge_pts], dtype=np.int32)
    edge[y1, x1] = 1
    edge[y2, x2] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_radius + 1, 2 * band_radius + 1))
    band = (cv2.dilate(edge, k, iterations=1) > 0) & inside
    side_a = band & (label == a)
    side_b = band & (label == b)
    if side_a.sum() < 8 or side_b.sum() < 8:
        return dict(valid=0, edge_count=float(len(edge_pts)))
    side_max = 0.0
    for ch in raw_channels.values():
        rr = robust_range(ch, inside)
        side = abs(float(np.median(ch[side_a])) - float(np.median(ch[side_b]))) / rr
        side_max = max(side_max, side)
    edge_ys = np.r_[y1, y2]
    edge_xs = np.r_[x1, x2]
    raw_grad = float(np.median(raw_grad_max[edge_ys, edge_xs]))
    infra_grad = float(np.median(infra_grad_norm[edge_ys, edge_xs]))
    support = side_max + 0.05 * min(4.0, max(0.0, raw_grad - 1.0))
    area_a = int(((label == a) & inside).sum())
    area_b = int(((label == b) & inside).sum())
    return dict(valid=1, edge_count=float(len(edge_pts)), side=float(side_max), raw_grad=raw_grad, infra_grad=infra_grad, support=support, area_a=float(area_a), area_b=float(area_b))

def raw_supported_split_boundary(stats: Dict[str, float], total_area: int) -> bool:
    if not stats.get('valid', 0):
        return False
    min_area = max(25.0, 0.006 * float(total_area))
    if min(stats.get('area_a', 0.0), stats.get('area_b', 0.0)) < min_area:
        return False
    if stats['edge_count'] < 21:
        return False
    weak_raw = stats['side'] >= 0.17 and stats['raw_grad'] >= 1.26
    medium_raw = stats['edge_count'] >= 38 and stats['side'] >= 0.23 and stats['support'] >= 0.29
    infra_supported = stats['infra_grad'] >= 1.30
    return bool(medium_raw or (weak_raw and infra_supported))

def raw_led_refine_candidate(
    infra_rgb: np.ndarray,
    raw_label: np.ndarray,
    raw_channels: Dict[str, np.ndarray],
    mask: np.ndarray,
    evidence: np.ndarray,
) -> np.ndarray:
    """Split raw regions only where infra proposes a boundary and RAW has calibrated support."""
    inside = mask > 0
    raw = relabel(fill_zero_inside(raw_label, inside)) if raw_label.max() > 0 else raw_label
    if raw.max() <= 0:
        out = np.zeros_like(raw, np.int32)
        out[inside] = 1
        return out
    split = intersect_label_maps(raw, evidence, mask, min_area_fraction=0.006)
    split = relabel(fill_zero_inside(split, inside))
    raw_grad_max = normalized_gradient_max(raw_channels, inside)
    I = corrected_infra_intensity(infra_rgb, mask)
    gx = cv2.Sobel(I.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    infra_grad = cv2.magnitude(gx, gy)
    denom = float(np.percentile(infra_grad[inside], 75)) if inside.sum() else 1.0
    infra_grad_norm = infra_grad / max(1.0, denom)
    total_area = int(inside.sum())

    for _ in range(80):
        parents = split_label_parents(split, raw, inside)
        pairs = pair_boundary_pixels(split, inside)
        unsupported: List[Tuple[float, int, int]] = []
        for (a, b), pts in pairs.items():
            if parents.get(a, 0) != parents.get(b, -1):
                continue
            stats = boundary_support_stats(split, a, b, pts, inside, raw_channels, raw_grad_max, infra_grad_norm)
            if not raw_supported_split_boundary(stats, total_area):
                priority = stats.get('support', 0.0) + 0.02 * stats.get('edge_count', 0.0) / max(1.0, math.sqrt(total_area))
                unsupported.append((priority, a, b))
        if not unsupported:
            break
        _, a, b = min(unsupported, key=lambda x: x[0])
        area_a = int(((split == a) & inside).sum())
        area_b = int(((split == b) & inside).sum())
        src, dst = (a, b) if area_a <= area_b else (b, a)
        split[split == src] = dst
        split = relabel(split)
    return relabel(fill_zero_inside(split, inside))

def eroded_seed_markers(label: np.ndarray, inside: np.ndarray) -> np.ndarray:
    markers = np.zeros_like(label, np.int32)
    for lab in [int(x) for x in np.unique(label[inside]) if x > 0]:
        m = (label == lab) & inside
        area = int(m.sum())
        if area == 0:
            continue
        radius = max(1, int(round(math.sqrt(area) / 80.0)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        core = cv2.erode(m.astype(np.uint8), k, iterations=1).astype(bool)
        if core.sum() < max(8, 0.03 * area):
            dist = distance_transform_edt(m)
            core = dist >= max(1.0, float(np.percentile(dist[m], 65)))
        if core.sum() == 0:
            core = m
        markers[core] = lab
    return markers

def normalized_gradient(ch: np.ndarray, inside: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(ch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    denom = float(np.percentile(g[inside], 75)) if inside.sum() else 1.0
    return g / max(1.0, denom)

def mean_shift_watershed_candidate(
    raw_rgb: np.ndarray,
    infra_rgb: np.ndarray,
    marker_label: np.ndarray,
    raw_mask: np.ndarray,
    infra_mask: np.ndarray,
    M: np.ndarray,
    sp: int = 12,
    sr: int = 18,
) -> np.ndarray:
    """Refine marker boundaries with raw mean-shift smoothing and seeded watershed."""
    inside = infra_mask > 0
    markers_src = relabel(fill_zero_inside(marker_label, inside)) if marker_label.max() > 0 else marker_label
    if markers_src.max() <= 1 or inside.sum() < 100:
        return markers_src
    raw_ms = cv2.pyrMeanShiftFiltering(raw_rgb, sp=sp, sr=sr, maxLevel=1)
    raw_ms_channels = raw_evidence_channels_in_infra(raw_ms, raw_mask, M, infra_rgb.shape[:2])
    raw_grad = normalized_gradient_max(raw_ms_channels, inside)
    infra_I = corrected_infra_intensity(infra_rgb, infra_mask)
    infra_grad = normalized_gradient(infra_I, inside)
    elevation = 0.72 * raw_grad + 0.28 * infra_grad
    elevation = cv2.GaussianBlur(elevation.astype(np.float32), (0, 0), sigmaX=0.7, sigmaY=0.7)
    elevation[~inside] = float(elevation[inside].max() if inside.any() else 0.0)
    markers = eroded_seed_markers(markers_src, inside)
    if markers.max() <= 1:
        return markers_src
    ws = watershed(elevation, markers=markers, mask=inside, watershed_line=False).astype(np.int32)
    ws = relabel(fill_zero_inside(ws, inside))
    return ws

def selective_decision_v3_fast(raw_score:float, cand_score:float, raw_regions:int, cand_regions:int, cand_bgrad:float)->str:
    # Merge-to-one anti-oversegmentation first: raw split the block, but infra boundaries are weak
    # and raw's infra-domain energy is already low. This prevents homogeneous infra blocks
    # from being overcut by either raw color noise or infra quantization.
    if raw_regions >= 3 and raw_score < 1.0 and cand_bgrad < 1.8:
        return 'merge_to_one'
    # Use infra when raw is low-confidence and infra has either much lower energy or strong supported boundaries.
    if cand_regions >= raw_regions + 3 and (cand_bgrad > 2.0 or cand_score < raw_score - 0.25):
        return 'use_infra'
    return 'keep_raw'

# ---------------- Visualization ----------------
def overlay_label(rgb,label,alpha=0.35):
    out=rgb.copy().astype(np.float32)
    for lab in [int(x) for x in np.unique(label) if x>0]:
        m=label==lab; out[m]=out[m]*(1-alpha)+COLORS[(lab-1)%len(COLORS)].astype(np.float32)*alpha
    b=find_boundaries(label,mode='outer'); out[b]=[255,255,0]
    return np.clip(out,0,255).astype(np.uint8)

def draw_contour(rgb,mask,color=(0,255,0),width=2):
    out=rgb.copy(); cs,_=cv2.findContours((mask>0).astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out,cs,-1,color,width); return out

def add_title(img,title,h=30):
    pil=Image.fromarray(img); can=Image.new('RGB',(pil.width,pil.height+h),'white'); can.paste(pil,(0,h)); d=ImageDraw.Draw(can)
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    except Exception: font=None
    d.text((5,7),title,fill='black',font=font); return np.array(can)

def resize_w(img,w):
    pil=Image.fromarray(img); h=max(1,int(pil.height*w/pil.width)); return np.array(pil.resize((w,h),Image.Resampling.LANCZOS))

def make_panel(raw_rgb, infra_rgb, raw_label, raw_gra, inferred_gra, auto_gra, gt_gra, gt_sub, raw_warp, cand, final, title):
    cols=[]
    cols.append(add_title(draw_contour(overlay_label(raw_rgb,raw_label),raw_gra),'raw v3'))
    if auto_gra is not None: cols.append(add_title(draw_contour(infra_rgb,auto_gra,(255,0,0)),'auto infra gra'))
    cols.append(add_title(draw_contour(infra_rgb,inferred_gra,(0,255,255)),'inferred gra'))
    if gt_gra is not None and gt_gra.sum()>0: cols.append(add_title(draw_contour(infra_rgb,gt_gra,(0,255,0)),'eval gra GT'))
    if gt_sub is not None: cols.append(add_title(overlay_label(infra_rgb,gt_sub),'infra GT subparts'))
    cols.append(add_title(overlay_label(infra_rgb,raw_warp),'raw warp'))
    cols.append(add_title(overlay_label(infra_rgb,cand),'infra candidate'))
    cols.append(add_title(overlay_label(infra_rgb,final),title))
    cols=[resize_w(c,260) for c in cols]
    H=max(c.shape[0] for c in cols); W=sum(c.shape[1] for c in cols)
    out=np.ones((H,W,3),np.uint8)*255; x=0
    for c in cols:
        out[:c.shape[0],x:x+c.shape[1]]=c; x+=c.shape[1]
    return out

def contact(panels,out,width):
    if not panels: return
    rows=[resize_w(p,width) for p in panels]
    W=max(r.shape[1] for r in rows); H=sum(r.shape[0]+8 for r in rows)
    sheet=np.ones((H,W,3),np.uint8)*255; y=0
    for r in rows:
        sheet[y:y+r.shape[0],:r.shape[1]]=r; y+=r.shape[0]+8
    Image.fromarray(sheet).save(out,quality=88)

# ---------------- Pairing ----------------
def load_pair_csv(path:Path)->List[Tuple[str,str]]:
    df=pd.read_csv(path)
    return list(zip(df['raw_file'].astype(str), df['infra_file'].astype(str)))

def pair_by_order(raws,infras):
    return [(r['file_name'], i['file_name']) for r,i in zip(sorted(raws,key=lambda x:x['file_name']), sorted(infras,key=lambda x:x['file_name']))]

def pair_by_shape(raws,infras,raw_masks,root,ann_path):
    # Lightweight fallback. For production, pair_csv is recommended because acquisition already knows pairs.
    infra_cands={}
    for im in infras:
        rgb=read_rgb(image_path(root,ann_path,im['file_name']))
        # use only first/best candidate per raw later to avoid expensive global descriptor search.
        infra_cands[im['file_name']]=auto_infra_candidates(rgb, max_cands=8)
    C=np.ones((len(raws),len(infras)),np.float32)*999
    for ri,r in enumerate(raws):
        rm=raw_masks[r['file_name']]
        rc=component_count(rm)
        for ii,inf in enumerate(infras):
            best=999.0
            for cm in infra_cands[inf['file_name']]:
                best=min(best, shape_iou_cost(rm,cm)+0.12*abs(component_count(cm)-rc))
            C[ri,ii]=best
    rr,cc=linear_sum_assignment(C)
    return [(raws[r]['file_name'], infras[c]['file_name']) for r,c in zip(rr,cc)]

# ---------------- Main ----------------
def main():
    args=parse_args()
    root,tmp=prepare_dataset(args)
    ann_path=find_ann(root)
    data,images,anns_by,cats=load_coco(ann_path)
    big_id=cats.get(args.big_category); sub_id=cats.get(args.subpart_category)
    if big_id is None: raise ValueError('big category not found')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    per_dir=args.out_dir/'per_pair'; per_dir.mkdir(exist_ok=True)
    p=V3Params()

    raws=[im for im in images.values() if args.raw_prefix in im['file_name']]
    infras=[im for im in images.values() if args.infra_prefix in im['file_name']]
    by_file={im['file_name']:im for im in images.values()}
    # raw gra unions from annotations only. This is the only upstream mask used for inference.
    raw_union={}
    raw_blocks={}
    for im in raws:
        ms=masks_for(im,anns_by,big_id)
        raw_blocks[im['file_name']]=ms
        raw_union[im['file_name']]=union_masks(ms,(im['height'],im['width']))
    if args.pair_csv is not None or args.pair_mode=='csv':
        if args.pair_csv is None:
            raise ValueError('--pair-mode csv requires --pair-csv')
        pairs=load_pair_csv(args.pair_csv)
    elif args.pair_mode=='filename':
        pairs=pair_by_filename(raws,infras)
    elif args.pair_mode=='order':
        pairs=pair_by_order(raws,infras)
    else:
        pairs=pair_by_shape(raws,infras,raw_union,root,ann_path)

    pair_rows=[]; metric_rows=[]; decision_rows=[]; panels=[]
    for raw_fn,infra_fn in pairs:
        print(f"[pipeline] processing {raw_fn} -> {infra_fn}", flush=True)
        if raw_fn not in by_file or infra_fn not in by_file: continue
        raw_im=by_file[raw_fn]; infra_im=by_file[infra_fn]
        raw_rgb=read_rgb(image_path(root,ann_path,raw_fn)); infra_rgb=read_rgb(image_path(root,ann_path,infra_fn))
        raw_gra_blocks=raw_blocks[raw_fn]
        if not raw_gra_blocks: continue
        raw_gra_union=raw_union[raw_fn]
        # Generate raw v3 labels over all raw gra blocks.
        raw_label_full=np.zeros(raw_gra_union.shape,np.int32); nxt=1
        for bidx,bm in enumerate(raw_gra_blocks):
            lab=segment_raw_v3(raw_rgb,bm,p)
            for l in [int(x) for x in np.unique(lab) if x>0]:
                raw_label_full[lab==l]=nxt; nxt+=1
        # Infer infra gra target mask automatically and register raw union to it.
        auto_infra_gra, auto_cost=choose_best_infra_mask_for_raw(raw_gra_union, infra_rgb)
        M=affine_from_bbox(raw_gra_union, auto_infra_gra) if auto_infra_gra.sum()>0 else np.array([[1,0,0],[0,1,0]],np.float32)
        inferred_gra=warp_mask(raw_gra_union,M,infra_rgb.shape[:2])
        raw_warp_label=warp_label(raw_label_full,M,infra_rgb.shape[:2])
        raw_warp_label[~(inferred_gra>0)]=0
        raw_warp_label=relabel(fill_zero_inside(raw_warp_label,inferred_gra>0)) if raw_warp_label.max()>0 else raw_warp_label
        # Infra candidate uses inferred gra, not manual infra gra.
        infra_candidate=infra_kmeans_candidate(infra_rgb,inferred_gra)
        infra_candidate[~(inferred_gra>0)]=0
        infra_candidate=relabel(fill_zero_inside(infra_candidate,inferred_gra>0)) if infra_candidate.max()>0 else infra_candidate
        # Use the original high-sensitivity infra candidate for evidence/decision,
        # but use a coarsened version for final output when infra is selected.
        infra_candidate_coarse = coarsen_candidate_by_adjacency(infra_candidate, inferred_gra) if args.coarsen_infra_output else infra_candidate
        one_label=np.zeros_like(raw_warp_label,np.int32); one_label[inferred_gra>0]=1
        raw_score=score_segmentation_in_infra(infra_rgb,raw_warp_label,inferred_gra)
        cand_score=score_segmentation_in_infra(infra_rgb,infra_candidate,inferred_gra)
        raw_regions=int(raw_warp_label.max()); cand_regions=int(infra_candidate.max())
        cand_bgrad=boundary_gradient_support(infra_rgb,infra_candidate,inferred_gra)
        decision=selective_decision_v3_fast(raw_score,cand_score,raw_regions,cand_regions,cand_bgrad)
        infra_q4_evidence=infra_quantile_candidate(infra_rgb,inferred_gra,k=4)
        hybrid_q4=hybrid_raw_infra_candidate(infra_rgb,raw_warp_label,inferred_gra,infra_q4_evidence)
        hybrid_from_candidate=hybrid_raw_infra_candidate(infra_rgb,raw_warp_label,inferred_gra,infra_candidate)
        hybrid_q4_score=score_segmentation_in_infra(infra_rgb,hybrid_q4,inferred_gra)
        hybrid_candidate_score=score_segmentation_in_infra(infra_rgb,hybrid_from_candidate,inferred_gra)
        if hybrid_candidate_score < hybrid_q4_score:
            hybrid_raw_infra=hybrid_from_candidate
            hybrid_score=hybrid_candidate_score
            hybrid_source='infra_candidate'
        else:
            hybrid_raw_infra=hybrid_q4
            hybrid_score=hybrid_q4_score
            hybrid_source='infra_q4'
        raw_channels_infra=raw_evidence_channels_in_infra(raw_rgb,raw_gra_union,M,infra_rgb.shape[:2])
        raw_led_q4=raw_led_refine_candidate(infra_rgb,raw_warp_label,raw_channels_infra,inferred_gra,infra_q4_evidence)
        raw_led_candidate=raw_led_refine_candidate(infra_rgb,raw_warp_label,raw_channels_infra,inferred_gra,infra_candidate)
        raw_led_q4_score=score_segmentation_in_infra(infra_rgb,raw_led_q4,inferred_gra)
        raw_led_candidate_score=score_segmentation_in_infra(infra_rgb,raw_led_candidate,inferred_gra)
        if raw_led_candidate_score < raw_led_q4_score:
            raw_led_refined=raw_led_candidate
            raw_led_score=raw_led_candidate_score
            raw_led_source='infra_candidate'
        else:
            raw_led_refined=raw_led_q4
            raw_led_score=raw_led_q4_score
            raw_led_source='infra_q4'
        raw_led_regions=int(raw_led_refined.max())
        watershed_raw=mean_shift_watershed_candidate(raw_rgb,infra_rgb,raw_warp_label,raw_gra_union,inferred_gra,M)
        watershed_raw_led=mean_shift_watershed_candidate(raw_rgb,infra_rgb,raw_led_refined,raw_gra_union,inferred_gra,M)
        watershed_raw_score=score_segmentation_in_infra(infra_rgb,watershed_raw,inferred_gra)
        watershed_raw_led_score=score_segmentation_in_infra(infra_rgb,watershed_raw_led,inferred_gra)
        if watershed_raw_led_score < watershed_raw_score:
            mean_shift_watershed=watershed_raw_led
            watershed_score=watershed_raw_led_score
            watershed_source='raw_led_seed'
        else:
            mean_shift_watershed=watershed_raw
            watershed_score=watershed_raw_score
            watershed_source='raw_seed'
        watershed_regions=int(mean_shift_watershed.max())
        raw_led_improves=(raw_led_regions > raw_regions and raw_led_regions <= max(raw_regions + 5, raw_regions * 2) and raw_led_score < raw_score - 0.08)
        raw_led_conservative_split=(raw_regions >= 5 and raw_led_regions > raw_regions and raw_led_regions <= raw_regions + 3 and cand_regions >= raw_regions + 3 and cand_bgrad < 1.8 and raw_led_score < raw_score + 0.25)
        watershed_improves=(watershed_regions >= raw_regions and watershed_regions <= max(raw_regions + 5, raw_regions * 2) and watershed_score < raw_score - 0.10 and watershed_score <= raw_led_score + 0.05)
        raw_led_split_ok=raw_regions > 2 and (raw_led_improves or raw_led_conservative_split)
        watershed_refines_raw_led=(raw_led_split_ok and int(watershed_raw_led.max()) == raw_led_regions and watershed_raw_led_score <= raw_led_score + 0.05)
        if decision=='merge_to_one':
            final=one_label
        elif watershed_refines_raw_led:
            decision='mean_shift_watershed'
            final=watershed_raw_led
        elif raw_led_split_ok:
            decision='raw_split_refined'
            final=raw_led_refined
        elif raw_regions > 2 and watershed_improves:
            decision='mean_shift_watershed'
            final=mean_shift_watershed
        elif decision=='use_infra':
            final=infra_candidate_coarse
        elif raw_regions == 2 and cand_regions >= 6 and cand_bgrad < 1.8 and hybrid_score < raw_score - 0.05:
            decision='hybrid_raw_infra'
            final=hybrid_raw_infra
        else:
            final=raw_warp_label
        # eval labels if available. Need match each warped raw block to infra GT block. For current data mostly one block; use union/subparts per image.
        gt_gra_blocks=masks_for(infra_im,anns_by,big_id) if args.eval else []
        gt_sub_masks=masks_for(infra_im,anns_by,sub_id) if args.eval else []
        gt_gra_union=union_masks(gt_gra_blocks,infra_rgb.shape[:2]) if gt_gra_blocks else np.zeros(infra_rgb.shape[:2],np.uint8)
        auto_iou=registration_iou(auto_infra_gra,gt_gra_union) if args.eval and gt_gra_union.sum()>0 else float('nan')
        reg_iou=registration_iou(inferred_gra,gt_gra_union) if args.eval and gt_gra_union.sum()>0 else float('nan')
        pair_rows.append(dict(raw_file=raw_fn,infra_file=infra_fn,raw_id=raw_im['id'],infra_id=infra_im['id'],auto_mask_cost=auto_cost,auto_mask_iou=auto_iou,reg_iou=reg_iou,decision=decision,raw_regions=raw_regions,infra_candidate_regions=cand_regions,hybrid_regions=int(hybrid_raw_infra.max()),hybrid_source=hybrid_source,raw_led_regions=raw_led_regions,raw_led_source=raw_led_source,watershed_regions=watershed_regions,watershed_source=watershed_source,raw_score=raw_score,infra_score=cand_score,hybrid_score=hybrid_score,raw_led_score=raw_led_score,watershed_score=watershed_score,watershed_raw_score=watershed_raw_score,watershed_raw_led_score=watershed_raw_led_score,hybrid_q4_score=hybrid_q4_score,hybrid_candidate_score=hybrid_candidate_score,raw_led_q4_score=raw_led_q4_score,raw_led_candidate_score=raw_led_candidate_score,infra_bgrad=cand_bgrad))
        if args.eval and gt_gra_blocks and gt_sub_masks:
            # eval inside gt gra union intersect inferred gra union to avoid penalizing slight target mask extrapolation too much.
            # If inferred/gra overlap is too small, still use gt union for visibility.
            inside_eval=(gt_gra_union>0) & (inferred_gra>0)
            if inside_eval.sum()<0.5*(gt_gra_union>0).sum(): inside_eval=(gt_gra_union>0)
            # label GT subparts assigned to infra union; for multiple blocks, just union all subparts in same image.
            gt_lab=np.zeros(infra_rgb.shape[:2],np.int32)
            cur=1
            for sm in gt_sub_masks:
                gt_lab[(sm>0)&inside_eval]=cur; cur+=1
            if gt_lab[inside_eval].max()>0:
                gt_lab=relabel(fill_zero_inside(gt_lab,inside_eval))
                # evaluate raw, infra, final for same pair.
                for method,label,dec in [('raw_v3_warp',raw_warp_label,'raw'),('merge_to_one',one_label,'one'),('infra_candidate',infra_candidate,'infra'),('infra_candidate_coarse',infra_candidate_coarse,'infra_coarse'),('hybrid_raw_infra',hybrid_raw_infra,'hybrid'),('raw_led_refined',raw_led_refined,'raw_led'),('mean_shift_watershed',mean_shift_watershed,'watershed'),('watershed_raw_seed',watershed_raw,'watershed_raw'),('watershed_raw_led_seed',watershed_raw_led,'watershed_raw_led'),('selective_hybrid_v4_smooth',final,decision)]:
                    pred=relabel(fill_zero_inside(label,inside_eval)) if label.max()>0 else label
                    metric_rows.append(asdict(eval_metrics(method,raw_fn,infra_fn,int(raw_im['id']),int(infra_im['id']),1,1,reg_iou,dec,gt_lab,pred,inside_eval,raw_regions,cand_regions,raw_score,cand_score,auto_iou)))
                if not args.skip_overview:
                    panel=make_panel(raw_rgb,infra_rgb,raw_label_full,raw_gra_union,inferred_gra,auto_infra_gra,gt_gra_union,gt_lab,raw_warp_label,infra_candidate,final,f"final {decision} | ARI see csv")
                    panels.append(panel)
                    Image.fromarray(panel).save(per_dir/f"raw{raw_im['id']}_infra{infra_im['id']}.jpg",quality=95)
        else:
            if not args.skip_overview:
                panel=make_panel(raw_rgb,infra_rgb,raw_label_full,raw_gra_union,inferred_gra,auto_infra_gra,None,None,raw_warp_label,infra_candidate,final,f"final {decision}")
                panels.append(panel)
                Image.fromarray(panel).save(per_dir/f"raw{raw_im['id']}_infra{infra_im['id']}.jpg",quality=95)
        # save masks
        np.savez_compressed(per_dir/f"raw{raw_im['id']}_infra{infra_im['id']}_masks.npz",raw_label=raw_label_full,raw_gra=raw_gra_union,auto_infra_gra=auto_infra_gra,inferred_gra=inferred_gra,raw_warp_label=raw_warp_label,infra_q4_evidence=infra_q4_evidence,infra_candidate=infra_candidate,infra_candidate_coarse=infra_candidate_coarse,hybrid_q4=hybrid_q4,hybrid_from_candidate=hybrid_from_candidate,hybrid_raw_infra=hybrid_raw_infra,raw_led_q4=raw_led_q4,raw_led_candidate=raw_led_candidate,raw_led_refined=raw_led_refined,watershed_raw=watershed_raw,watershed_raw_led=watershed_raw_led,mean_shift_watershed=mean_shift_watershed,one_label=one_label,final_label=final,M=M)
    pd.DataFrame(pair_rows).to_csv(args.out_dir/'pair_registration_decisions.csv',index=False)
    if metric_rows:
        df=pd.DataFrame(metric_rows)
        df.to_csv(args.out_dir/'per_block_metrics.csv',index=False)
        summary=df.groupby('method').agg(blocks=('ari','count'),ari_mean=('ari','mean'),ari_median=('ari','median'),boundary_f1_mean=('boundary_f1','mean'),cover_gt_mean=('cover_gt','mean'),pred_regions_total=('pred_regions','sum'),gt_regions_total=('gt_regions','sum')).reset_index()
        summary.to_csv(args.out_dir/'summary_metrics.csv',index=False)
        df_complex=df.copy()
        df_complex['complexity']=np.where(df_complex['gt_regions']<=1,'single_gt','complex_gt_gt1')
        all_rows=df_complex.copy()
        all_rows['complexity']='all'
        summary_by_complexity=pd.concat([all_rows,df_complex],ignore_index=True).groupby(['complexity','method']).agg(blocks=('ari','count'),ari_mean=('ari','mean'),ari_median=('ari','median'),boundary_f1_mean=('boundary_f1','mean'),cover_gt_mean=('cover_gt','mean'),pred_regions_total=('pred_regions','sum'),gt_regions_total=('gt_regions','sum')).reset_index()
        summary_by_complexity.to_csv(args.out_dir/'summary_metrics_by_complexity.csv',index=False)
    if not args.skip_overview:
        contact(panels,args.out_dir/'pipeline_overview.jpg',args.overview_width)
    # config json
    (args.out_dir/'run_config.json').write_text(json.dumps(dict(params=asdict(p),pair_mode=args.pair_mode,pair_csv=str(args.pair_csv) if args.pair_csv else None),indent=2))
    if tmp is not None: tmp.cleanup()

if __name__ == '__main__':
    main()
