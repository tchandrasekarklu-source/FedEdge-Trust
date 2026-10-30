"""Actual target-device latency/peak memory instrumentation, no invented metrics.

Only single-input TorchScript models supported directly; wrap complex models.
Energy measurement is intentionally not inferred from elapsed CPU/GPU time.
"""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import psutil
import torch


def benchmark(model,input_tensor,warmup=50,repetitions=1000):
    if warmup<0 or repetitions<10:raise ValueError('at least 10 measured repetitions')
    device=input_tensor.device
    model.eval()
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(warmup):model(input_tensor)
        sync()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        rss_before=psutil.Process().memory_info().rss
        elapsed=[]
        for _ in range(repetitions):
            sync();t=time.perf_counter()
            model(input_tensor);sync()
            elapsed.append((time.perf_counter()-t)*1e3)
        rss_after=psutil.Process().memory_info().rss
    return {'observed_on_supplied_model':True,'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'torch_version':torch.__version__,'cpu':platform.processor(),'precision':str(input_tensor.dtype),
            'input_shape':list(input_tensor.shape),'batch_size':int(input_tensor.shape[0]),
            'warmup_iterations':warmup,'measured_iterations':repetitions,
            'latency_ms':{q:float(np.percentile(elapsed,q)) for q in (50,95,99)},
            'latency_mean_ms':float(np.mean(elapsed)), 'throughput_samples_per_s_sequential':float(input_tensor.shape[0]*1000/np.mean(elapsed)),
            'rss_MB_before':rss_before/1e6,'rss_MB_after':rss_after/1e6,
            'max_gpu_allocated_MB':float(torch.cuda.max_memory_allocated(device)/1e6) if device.type=='cuda' else None,
            'energy_mJ_per_sample':None,'energy_note':'NOT MEASURED; external power metering required'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--module',required=True);p.add_argument('--input-shape',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--warmup',type=int,default=50);p.add_argument('--repetitions',type=int,default=1000)
    a=p.parse_args()
    shape=tuple(int(z) for z in a.input_shape.split(','))
    if not shape or min(shape)<1:raise ValueError('shape must contain positive dimensions')
    module=torch.jit.load(a.module,map_location=a.device)
    z=benchmark(module,torch.randn(*shape,device=a.device),a.warmup,a.repetitions)
    Path(a.out).write_text(json.dumps(z,indent=2),encoding='utf8')
    print('Written measured device report',a.out)
if __name__=='__main__':main()
