"""Lesson 4: only exact token prefixes inside the same execution scope are reusable."""

from inference_platform.prefix_cache import CacheScope, PrefixCache


cache = PrefixCache(block_size=4)
acme = CacheScope("model@abc", "tokenizer@7", "adapter:none", "acme")
beta = CacheScope("model@abc", "tokenizer@7", "adapter:none", "beta")

cache.store(acme, (101, 102, 103, 104, 201, 202, 203, 204, 999))
same_scope = cache.lookup(acme, (101, 102, 103, 104, 201, 202, 203, 204, 301))
other_tenant = cache.lookup(beta, (101, 102, 103, 104, 201, 202, 203, 204, 301))

print(f"same scope: reuse={same_scope.reused_tokens}, {same_scope.reason}")
print(f"other tenant: reuse={other_tenant.reused_tokens}, {other_tenant.reason}")
