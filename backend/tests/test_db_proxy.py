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


if __name__ == "__main__":
    unittest.main()
