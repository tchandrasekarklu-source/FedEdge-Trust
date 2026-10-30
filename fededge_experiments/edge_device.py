"""Run this script physically ON the deployment device (Jetson, Pi, bedside CPU).
No hardware evidence is claimed when executed on a developer laptop/cloud VM.
Energy optional and requires contemporaneous synchronized external power data.
"""
import argparse, hashlib, json, platform, socket, time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import torch
from .data import LinkedCohort
from .model import ITGFusion


def device_model():
    for name in ['/proc/device-tree/model','/sys/firmware/devicetree/base/model']:
        try:
            return Path(name).read_bytes().decode('utf8','replace').rstrip('\x00')
        except OSError:pass
    return None


def integrated_energy_joules(df,start,end,idle_power=0.):
    """CSV epoch_seconds,watts from trusted, separately calibrated external meter."""
    if not {'epoch_seconds','watts'}.issubset(df.columns):raise ValueError('Power CSV requires epoch_seconds,watts')
    t=df.epoch_seconds.to_numpy(float);w=df.watts.to_numpy(float)
    if not np.isfinite(t).all() or not np.isfinite(w).all() or (w<0).any():
        raise ValueError('Power CSV has nonfinite or negative readings')
    if not np.all(np.diff(t)>0) or start<t[0] or end>t[-1] or end<=start:
        raise ValueError('Power readings must cover full measured window on same system clock')
    inside=(t>start)&(t<end)
    grid_t=np.concatenate(([start],t[inside],[end]))
    grid_w=np.interp(grid_t,t,w)
    return max(0.,float(np.trapezoid(np.maximum(grid_w-idle_power,0),grid_t)))


def evaluate_on_device(model,inputs,device,warmup=25,repetitions=300,power_csv=None,idle_w=0.):
    if repetitions<10 or warmup<0:raise ValueError('At least 10 iterations required')
    model=model.to(device).eval();inputs=[t.to(device) for t in inputs]
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(warmup):model(*inputs)
        sync();start_epoch=time.time()
        start=time.perf_counter();elapsed=[]
        rss0=psutil.Process().memory_info().rss
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repetitions):
            sync();before=time.perf_counter()
            model(*inputs);sync()
            elapsed.append((time.perf_counter()-before)*1000)
        sync();end=time.perf_counter();end_epoch=time.time()
        rss1=psutil.Process().memory_info().rss
    m=device_model();is_edge=bool(m and any(t in m.lower() for t in ('raspberry','jetson','nvidia')))
    n_samples=repetitions*len(inputs[0])
    report={'measured_execution':True,'system_hostname':socket.gethostname(),
            'platform':platform.platform(),'board_model':m,'board_identified_as_edge':is_edge,
            'cpu':platform.processor(),'gpu':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            'torch_version':torch.__version__,'precision':str(inputs[0].dtype),
            'batch_size':len(inputs[0]),'warmup_iterations':warmup,'measured_iterations':repetitions,
            'mean_latency_ms':float(np.mean(elapsed)),'p50_latency_ms':float(np.percentile(elapsed,50)),
            'p95_latency_ms':float(np.percentile(elapsed,95)),'p99_latency_ms':float(np.percentile(elapsed,99)),
            'sequential_throughput_samples_s':n_samples/(end-start),
            'elapsed_seconds':end-start,'rss_before_MB':rss0/1e6,'rss_after_MB':rss1/1e6,
            'max_gpu_memory_MB':float(torch.cuda.max_memory_allocated(device)/1e6) if str(device).startswith('cuda') else None,
            'wall_start_epoch':start_epoch,'wall_end_epoch':end_epoch,
            'energy_mJ_per_sample':None,'energy_provenance':'unmeasured: external calibrated power CSV required'}
    if power_csv:
        j=integrated_energy_joules(pd.read_csv(power_csv),start_epoch,end_epoch,idle_w)
        report['energy_mJ_per_sample']=j*1000/n_samples
        report['energy_provenance']='external sampled watts integrated over precisely observed wall clock window; subtract idle watts'
        report['energy_idle_watts']=idle_w
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--cohort',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--warmup',type=int,default=25);p.add_argument('--repetitions',type=int,default=300)
    p.add_argument('--power-csv');p.add_argument('--idle-watts',type=float,default=0.)
    a=p.parse_args()
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=ITGFusion(**ck['model_kwargs']);model.load_state_dict(ck['state_dict'])
    cohort=LinkedCohort(a.cohort,require_primary=True)
    inputs,_=cohort.batch(cohort.indices('test')[:1])
    report=evaluate_on_device(model,inputs,a.device,a.warmup,a.repetitions,a.power_csv,a.idle_watts)
    report['checkpoint_sha256']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    report['model_parameters']=int(sum(p.numel() for p in model.parameters()))
    report['checkpoint_MB']=Path(a.checkpoint).stat().st_size/1e6
    report['verification_note']='Device identified from /proc/device-tree/model; cloud/developer CPU is NOT a physical edge evaluation.'
    Path(a.out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
