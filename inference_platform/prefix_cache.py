"""Reuse prefill work only across an exact token prefix and security scope.

Text that looks identical can tokenize differently, and the same token ids can mean
different activations under another model revision or adapter. Cache identity must
therefore include those execution inputs plus the tenant/security domain. Entries
below contain complete blocks only, matching paged runtimes that cannot safely claim
reuse for a partially populated block.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class CacheScope:
    model_revision: str
    tokenizer_revision: str
    adapter_revision: str
    tenant: str


@dataclass(frozen=True)
class PrefixDecision:
    reused_tokens: int
    recompute_tokens: int
    cache_key: str | None
    reason: str


class PrefixCache:
    """A deterministic content-addressed cache for complete token blocks."""

    def __init__(self, block_size: int = 4) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._entries: dict[tuple[CacheScope, str], tuple[int, ...]] = {}

    def store(self, scope: CacheScope, tokens: tuple[int, ...]) -> str | None:
        """Store the complete-block portion of a computed prefix.

        Partial trailing blocks are omitted because later tokens would change their
        contents. Empty or shorter-than-one-block inputs produce no entry. The key is
        derived from scope and actual token ids, never from a caller-provided hit label.
        """

        _validate_scope(scope)
        if any(token < 0 for token in tokens):
            raise ValueError("token ids may not be negative")
        complete_length = len(tokens) - len(tokens) % self.block_size
        if complete_length == 0:
            return None
        complete = tokens[:complete_length]
        key = _digest(scope, complete)
        self._entries[(scope, key)] = complete
        return key

    def lookup(self, scope: CacheScope, tokens: tuple[int, ...]) -> PrefixDecision:
        """Decide the longest exact, in-scope prefix that may bypass prefill.

        Candidates from another model, tokenizer, adapter, or tenant are invisible.
        Within the scope, the actual requested token prefix must equal the cached
        sequence; a common label or hash supplied by the request cannot force a hit.
        Longest-prefix selection occurs only after those checks. The remainder is
        reported as recompute work, including a partial trailing block.
        """

        _validate_scope(scope)
        if any(token < 0 for token in tokens):
            raise ValueError("token ids may not be negative")
        candidates = [
            (key, cached)
            for (entry_scope, key), cached in self._entries.items()
            if entry_scope == scope
            and len(cached) <= len(tokens)
            and tokens[: len(cached)] == cached
        ]
        if not candidates:
            return PrefixDecision(0, len(tokens), None, "no exact in-scope prefix")
        key, cached = max(candidates, key=lambda candidate: len(candidate[1]))
        return PrefixDecision(
            len(cached),
            len(tokens) - len(cached),
            key,
            f"reused {len(cached) // self.block_size} complete blocks",
        )


def _digest(scope: CacheScope, tokens: tuple[int, ...]) -> str:
    material = "\x1f".join(
        (
            scope.model_revision,
            scope.tokenizer_revision,
            scope.adapter_revision,
            scope.tenant,
            ",".join(map(str, tokens)),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _validate_scope(scope: CacheScope) -> None:
    if any(
        not field.strip()
        for field in (
            scope.model_revision,
            scope.tokenizer_revision,
            scope.adapter_revision,
            scope.tenant,
        )
    ):
        raise ValueError("every cache scope field is required")
