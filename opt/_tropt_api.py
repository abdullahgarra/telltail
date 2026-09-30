"""Thin wrapper over the private `tropt` library's GASLITE / RASLITE+ optimizers.

Ports the research `tropt_api.craft_fingerprint_query`. The target abstraction lives in
the caller (`opt.optimize.Target`): pass exactly one of `target_text` (passage) or
`target_vector` (centroid). White-box (HF encoders) is implemented; black-box (OpenAI /
API-only, RASLITE+) is stubbed for the next increment.

Requires the `tropt` env (the optimizer is not a dependency of the generic pipeline).
"""
from __future__ import annotations

from typing import List, Optional

from . import config as C


def craft_fingerprint_query(*, model, query_template: str, initial_trigger: str,
                            forbidden_tokens: Optional[List[int]],
                            target_text: Optional[str] = None,
                            target_vector=None, device: Optional[str] = None,
                            optim_method: str = "gaslite-whitebox") -> str:
    """Return the optimized trigger string for `query_template` (must contain the
    literal `{{OPTIMIZED_TRIGGER}}` placeholder) under `model`."""
    import torch  # noqa
    from tropt.loss.base import SimilarityLoss
    from tropt.optimizer.utils.token_constraints import TokenConstraints
    from tropt.optimizer.gaslite_optimizer import GASLITEOptimizer

    assert optim_method in ("gaslite-whitebox", "gaslite-blackbox"), optim_method
    assert "{{OPTIMIZED_TRIGGER}}" in query_template, "template needs {{OPTIMIZED_TRIGGER}}"
    assert (target_vector is not None) != (target_text is not None), \
        "specify exactly one of target_text / target_vector"

    dev = device or (model.device if hasattr(model, "device") else
                     ("cuda" if torch.cuda.is_available() else "cpu"))

    # target_text -> embed under the model to get the target vector
    if target_text is not None:
        target_vector = model([target_text])
    if not torch.is_tensor(target_vector):
        target_vector = torch.as_tensor(target_vector)

    loss = SimilarityLoss()
    token_constraints = TokenConstraints(
        disallow_non_ascii=True, disallow_special_tokens=True,
        disallow_custom_token_ids=forbidden_tokens or [],
    )

    if optim_method == "gaslite-blackbox":
        raise NotImplementedError(
            "Black-box (RASLITE+) OpenAI path is the next increment; see opt.config.BLACKBOX.")

    optimizer = GASLITEOptimizer(
        model=model, loss=loss,
        n_candidates=C.WHITEBOX["n_candidates"],
        num_steps=C.WHITEBOX["num_steps"],
        n_grad=C.WHITEBOX["n_grad"],
        n_flip=C.WHITEBOX["n_flip"],
        token_constraints=token_constraints,
        use_retokenize=C.WHITEBOX["use_retokenize"],
    )

    def _optimize():
        return optimizer.optimize_trigger(
            texts=[query_template],
            targets={loss.TARGET_KEY: target_vector.to(dev)},
            initial_trigger=initial_trigger,
        )

    try:
        result = _optimize()
    except RuntimeError as e:
        if "No token sequences are the same after decoding and re-encoding" in str(e):
            print("[warn] retokenize filtered all candidates; retrying use_retokenize=False")
            optimizer.use_retokenize = False
            result = _optimize()
        else:
            raise
    return result.best_trigger_str
