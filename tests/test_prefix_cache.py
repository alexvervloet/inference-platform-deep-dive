from __future__ import annotations

import unittest

from inference_platform.prefix_cache import CacheScope, PrefixCache


ACME = CacheScope("model@abc", "tokenizer@1", "adapter:none", "acme")


class PrefixCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = PrefixCache(block_size=2)
        self.cache.store(ACME, (10, 20, 30, 40, 50))

    def test_reuses_only_complete_exact_prefix_blocks(self) -> None:
        decision = self.cache.lookup(ACME, (10, 20, 30, 40, 99, 100))
        self.assertEqual(decision.reused_tokens, 4)
        self.assertEqual(decision.recompute_tokens, 2)
        self.assertIn("2 complete blocks", decision.reason)

    def test_changed_token_forces_recompute_despite_matching_length(self) -> None:
        decision = self.cache.lookup(ACME, (10, 20, 31, 40, 50))
        self.assertEqual(decision.reused_tokens, 0)
        self.assertEqual(decision.reason, "no exact in-scope prefix")

    def test_tenant_model_tokenizer_and_adapter_are_cache_boundaries(self) -> None:
        alternatives = (
            CacheScope("model@def", "tokenizer@1", "adapter:none", "acme"),
            CacheScope("model@abc", "tokenizer@2", "adapter:none", "acme"),
            CacheScope("model@abc", "tokenizer@1", "adapter:legal", "acme"),
            CacheScope("model@abc", "tokenizer@1", "adapter:none", "beta"),
        )
        for scope in alternatives:
            with self.subTest(scope=scope):
                self.assertEqual(self.cache.lookup(scope, (10, 20, 30, 40)).reused_tokens, 0)

    def test_short_partial_block_is_not_stored(self) -> None:
        self.assertIsNone(self.cache.store(ACME, (1,)))


if __name__ == "__main__":
    unittest.main()
