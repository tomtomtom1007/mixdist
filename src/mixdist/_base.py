"""Minimal scikit-learn-compatible estimator plumbing.

``mixdist`` deliberately does not depend on scikit-learn: it only mimics the
parts of the estimator protocol (``get_params`` / ``set_params`` / ``_repr``)
that make objects usable inside ``Pipeline``, ``GridSearchCV`` and friends when
scikit-learn *is* installed.
"""

from __future__ import annotations

import inspect
from typing import Any


class NotFittedError(RuntimeError):
    """Raised when a method requiring fitted state is called before ``fit``."""


class BaseEstimator:
    """Duck-typed stand-in for ``sklearn.base.BaseEstimator``."""

    @classmethod
    def _param_names(cls) -> list[str]:
        init = cls.__init__
        if init is object.__init__:
            return []
        params = [
            p
            for p in inspect.signature(init).parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD and p.kind != p.VAR_POSITIONAL
        ]
        return sorted(p.name for p in params)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self._param_names():
            value = getattr(self, name)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                for key, sub in value.get_params().items():
                    out[f"{name}__{key}"] = sub
            out[name] = value
        return out

    def set_params(self, **params: Any):
        if not params:
            return self
        valid = self._param_names()
        nested: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            head, _, tail = key.partition("__")
            if head not in valid:
                raise ValueError(
                    f"Invalid parameter {head!r} for estimator {type(self).__name__}. "
                    f"Valid parameters are: {valid}."
                )
            if tail:
                nested.setdefault(head, {})[tail] = value
            else:
                setattr(self, head, value)
        for head, sub in nested.items():
            getattr(self, head).set_params(**sub)
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        defaults = {
            p.name: p.default
            for p in inspect.signature(type(self).__init__).parameters.values()
            if p.name != "self"
        }
        parts = []
        for name in self._param_names():
            value = getattr(self, name)
            if name in defaults and _eq(value, defaults[name]):
                continue
            parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def _check_fitted(self, attr: str = "_fitted") -> None:
        if not getattr(self, attr, False):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )


def _eq(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # pragma: no cover - exotic __eq__
        return a is b
