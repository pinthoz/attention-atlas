"""One lock for the shared transformer models.

``ModelManager`` caches models, so every panel that analyses the same model
is handed the SAME object. These analyses are not read-only:

* head ablation registers a forward hook, runs a pass, then removes it;
* AttnLRP (``lxt``) monkey-patches torch functions process-wide on entry
  and restores them on exit;
* LRP also calls ``zero_grad`` and ``backward`` on that shared model;
* the fast tokenizer is a Rust object that raises ``Already borrowed``
  when two threads reach it at once.

Overlapping any two of these corrupts the result of at least one, and the
ablation is the dangerous case because it fails silently: it returns
plausible numbers that are simply wrong.

Measured on the q64 pair with the MiniLM L-6 reranker:

* head ablation against itself, before this lock existed: wrong impacts in
  8 runs out of 30, with values up to 0.0114 where the true maximum is
  0.0052, and some heads reported at exactly 0.0;
* LRP against head ablation: the ablation came back wrong in 3 runs out of
  10, the LRP in 2, and 7 raised ``Already borrowed``;
* LRP against itself: the process died with ``0xC0000374``, heap corruption.

Integrated gradients and perturbation never returned wrong numbers on their
own, but they share the same model and tokenizer, so they take the lock too.

``RLock``, not ``Lock``: the ``batch_*`` helpers call the leaf helpers, and a
plain lock would deadlock on the nested call.
"""

import functools
import threading

MODEL_LOCK = threading.RLock()


def serialized(fn):
    """Hold :data:`MODEL_LOCK` for the whole call."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with MODEL_LOCK:
            return fn(*args, **kwargs)

    return wrapper


__all__ = ["MODEL_LOCK", "serialized"]
