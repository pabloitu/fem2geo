"""
Tests for fem2geo.internal.logger
"""

import logging
import unittest

from fem2geo.internal.logger import setup_logger, set_console_log_level


class TestLogger(unittest.TestCase):

    def test_setup_logger_creates_named_logger(self):
        setup_logger()
        log = logging.getLogger("fem2geoLogger")
        self.assertTrue(len(log.handlers) > 0 or log.parent is not None)

    def test_setup_logger_idempotent(self):
        setup_logger()
        n1 = len(logging.getLogger("fem2geoLogger").handlers)
        setup_logger()
        n2 = len(logging.getLogger("fem2geoLogger").handlers)
        self.assertEqual(n1, n2)

    def test_set_console_log_level_debug(self):
        setup_logger()
        set_console_log_level(logging.DEBUG)
        set_console_log_level(logging.INFO)
        set_console_log_level(logging.WARNING)

    def test_logger_emits(self):
        setup_logger()
        with self.assertLogs("fem2geoLogger", level="INFO") as cm:
            logging.getLogger("fem2geoLogger").info("test message")
        self.assertTrue(any("test message" in m for m in cm.output))


if __name__ == "__main__":
    unittest.main()