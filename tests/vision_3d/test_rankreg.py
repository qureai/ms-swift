"""CPU tests for the VICReg rank-reg loss (S5)."""
import torch
from swift.model.vision_3d.slowfast import rankreg_loss


def test_collapsed_is_penalised():
    Z = torch.ones(500, 16) * 3.0                      # all tokens identical -> zero variance
    loss = rankreg_loss(Z)
    assert loss.item() > 0.9, f'collapsed rep must incur ~gamma variance penalty, got {loss.item()}'
    print('ok collapsed_is_penalised')


def test_healthy_is_near_zero():
    torch.manual_seed(0)
    Z = torch.randn(8000, 16)                          # unit-variance, decorrelated
    loss = rankreg_loss(Z, max_tokens=8000)
    assert loss.item() < 0.05, f'healthy unit-variance decorrelated rep should be ~0, got {loss.item()}'
    print('ok healthy_is_near_zero')


def test_correlation_raises_cov_term():
    torch.manual_seed(0)
    base = torch.randn(4000, 2)
    W = torch.randn(2, 16)
    Z = (base @ W)                                     # rank-2 embedded in 16 dims (highly correlated dims)
    Z = Z / Z.std(0, keepdim=True)                     # unit per-dim variance -> var term ~0, isolate cov
    loss_corr = rankreg_loss(Z, var_w=0.0, cov_w=1.0, max_tokens=4000)
    loss_dec = rankreg_loss(torch.randn(4000, 16), var_w=0.0, cov_w=1.0, max_tokens=4000)
    assert loss_corr.item() > 10 * (loss_dec.item() + 1e-6), \
        f'correlated dims must have much higher cov term ({loss_corr.item():.3f} vs {loss_dec.item():.3f})'
    print('ok correlation_raises_cov_term')


def test_gradient_step_reduces_loss():
    torch.manual_seed(0)
    # a learnable map producing a nearly-collapsed Z; a few steps should reduce the rank-reg loss
    P = torch.nn.Linear(16, 16)
    with torch.no_grad():
        P.weight.mul_(0.01)                            # start near-collapsed
    opt = torch.optim.SGD(P.parameters(), lr=0.5)
    x = torch.randn(1000, 16)
    l0 = rankreg_loss(P(x)).item()
    for _ in range(20):
        opt.zero_grad(); loss = rankreg_loss(P(x)); loss.backward(); opt.step()
    l1 = rankreg_loss(P(x)).item()
    assert l1 < l0, f'rank-reg should decrease under optimisation: {l0:.3f} -> {l1:.3f}'
    print(f'ok gradient_step_reduces_loss ({l0:.3f} -> {l1:.3f})')


def test_tiny_batch_safe():
    assert rankreg_loss(torch.randn(1, 16)).item() == 0.0   # <2 tokens -> 0, no crash
    print('ok tiny_batch_safe')


if __name__ == '__main__':
    for fn in [test_collapsed_is_penalised, test_healthy_is_near_zero, test_correlation_raises_cov_term,
               test_gradient_step_reduces_loss, test_tiny_batch_safe]:
        fn()
    print('\nALL RANKREG TESTS PASSED')
