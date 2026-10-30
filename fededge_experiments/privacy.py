"""Per-sample clipped Poisson-subsampled Gaussian DP-SGD; integer-order RDP.

When used on disjoint local client subjects, parallel composition bound is
max(client_epsilon). This does NOT implement secure aggregation and does NOT
account for private calibration/tuning outside DP training.
"""
import math
import numpy as np
import torch
from scipy.special import gammaln, logsumexp


def rdp_accountant(q, sigma, steps, delta, orders=tuple(range(2,129))):
    if not (0 < q <= 1 and sigma > 0 and steps >= 0 and 0 < delta < 1):
        raise ValueError('Invalid Poisson rate/noise/steps/delta')
    eps_candidates=[]
    for alpha in orders:
        k=np.arange(alpha+1,dtype=np.float64)
        logcomb=gammaln(alpha+1)-gammaln(k+1)-gammaln(alpha-k+1)
        logw=logcomb + k*np.log(q) + (alpha-k)*np.log1p(-q) if q<1 else np.where(k==alpha,0.,-np.inf)
        logm=logsumexp(logw+(k*k-k)/(2*sigma*sigma))
        rdp=steps*max(0.,logm/(alpha-1))
        eps_candidates.append(rdp+math.log(1/delta)/(alpha-1))
    i=int(np.argmin(eps_candidates))
    return {'epsilon_upper_bound':float(eps_candidates[i]),'delta':float(delta),
            'alpha':int(orders[i]),'q':float(q),'sigma':float(sigma),'steps':int(steps),
            'accountant':'integer-order RDP, Poisson-subsampled Gaussian, upper-bound conversion',
            'assumptions':'disjoint subjects, independently sampled examples each step, pre-specified DP hyperparameters'}


def private_step(model, inputs, labels, optimizer, q, noise_multiplier, max_grad_norm,
                 reference=None, prox_mu=0., rng=None):
    """One Poisson step. Individually clips per-example data gradients before noise.

    `inputs` is a tuple of per-client tensors with same leading dimension.
    Fixed public q*n scaling means a sampled empty batch still adds Gaussian noise.
    """
    if not inputs or q<=0 or q>1 or noise_multiplier<=0 or max_grad_norm<=0:
        raise ValueError('DP-SGD parameters invalid')
    n=len(labels)
    if n<2 or any(len(z)!=n for z in inputs): raise ValueError('client tensor length mismatch')
    device=labels.device
    included=(torch.rand(n,device=device,generator=rng)<q).nonzero(as_tuple=False).reshape(-1)
    params=[p for p in model.parameters() if p.requires_grad]
    buffers=[torch.zeros_like(p) for p in params]
    model.train()
    for idx in included.tolist():
        logits=model(*(x[idx:idx+1] for x in inputs)).reshape(-1)
        l=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels[idx:idx+1].float())
        grads=torch.autograd.grad(l,params,allow_unused=True)
        norm=torch.sqrt(sum(torch.sum(g.detach()**2) for g in grads if g is not None)).clamp_min(1e-12)
        scale=min(1.,max_grad_norm/float(norm))
        for dst,g in zip(buffers,grads):
            if g is not None: dst.add_(g.detach(),alpha=scale)
    optimizer.zero_grad(set_to_none=True)
    for p,g in zip(params,buffers):
        noise=torch.randn(g.shape,device=g.device,dtype=g.dtype,generator=rng)*noise_multiplier*max_grad_norm
        p.grad=(g+noise)/(q*n)
        if reference is not None and prox_mu>0:
            p.grad.add_(p.detach()-reference[id(p)],alpha=prox_mu)
    optimizer.step()
    return int(len(included))
