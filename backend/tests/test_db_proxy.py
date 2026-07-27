"""LazyDbProxy lets api.py construct `db` at module import time without
blocking on the real LightRAGManager. Attribute access on the proxy blocks
until bind() is called with the real instance."""
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestLazyDbProxy(unittest.TestCase):
    def test_attribute_access_blocks_until_bind(self):
        from db_proxy import LazyDbProxy
        proxy = LazyDbProxy()

        result = []

        def reader():
            result.append(proxy.collection)

        t = threading.Thread(target=reader)
        t.start()

        # Reader should still be blocked.
        t.join(timeout=0.2)
        self.assertTrue(t.is_alive(), "reader returned before bind()")

        real = MagicMock()
        real.collection = "hello"
        proxy.bind(real)
        t.join(timeout=1.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, ["hello"])

    def test_method_calls_route_to_real_after_bind(self):
        from db_proxy import LazyDbProxy
        proxy = LazyDbProxy()
        real = MagicMock()
        real.add_rule.return_value = "rule-id-x"
        proxy.bind(real)
        self.assertEqual(proxy.add_rule("a", "b", "c", "d"), "rule-id-x")
        real.add_rule.assert_called_once_with("a", "b", "c", "d")

    def test_bind_twice_raises(self):
        from db_proxy import LazyDbProxy
        proxy = LazyDbProxy()
        proxy.bind(MagicMock())
        with self.assertRaises(RuntimeError):
            proxy.bind(MagicMock())


class TestLazyDbProxyTimeout(unittest.TestCase):
    """An unbound proxy must fail loudly rather than block forever.

    The wait is intentional — in production the embedding model may still be
    loading. But an unbounded wait turns "nobody will ever call bind()" into a
    permanent hang with no diagnostic: the request thread parks on
    Event.wait() and the caller sees a request that never returns.
    """

    def test_unbound_access_raises_after_timeout(self):
        from db_proxy import LazyDbProxy
        proxy = LazyDbProxy(ready_timeout=0.1)

        with self.assertRaises(TimeoutError):
            proxy.collection

    def test_timeout_error_names_the_attribute_and_the_cause(self):
        """The message must point at the unbound proxy, not look like a network error."""
        from db_proxy import LazyDbProxy
        proxy = LazyDbProxy(ready_timeout=0.1)

        with self.assertRaises(TimeoutError) as ctx:
            proxy.find_similar_rule

        message = str(ctx.exception)
        self.assertIn("find_similar_rule", message)
        self.assertIn("bind()", message)

    def test_bind_before_timeout_still_succeeds(self):
        """A slow-but-successful bind must not be turned into an error."""
        from db_proxy import LazyDbProxy
        proxy = LazyDbProxy(ready_timeout=5.0)
        real = MagicMock()
        real.collection = "bound-in-time"

        threading.Timer(0.05, lambda: proxy.bind(real)).start()

        self.assertEqual(proxy.collection, "bound-in-time")

    def test_default_timeout_is_bounded(self):
        """The default must be finite, or the deadlock returns by omission."""
        from db_proxy import LazyDbProxy
        proxy = LazyDbProxy()

        self.assertIsNotNone(proxy._ready_timeout)
        self.assertGreater(proxy._ready_timeout, 0)


if __name__ == "__main__":
    unittest.main()
