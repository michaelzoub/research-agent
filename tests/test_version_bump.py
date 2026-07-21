from __future__ import annotations

import unittest

from scripts.bump_version import bump_version


class VersionBumpTests(unittest.TestCase):
    def test_patch_is_default_semantic_progression(self) -> None:
        self.assertEqual(bump_version("1.2.3", "patch"), "1.2.4")

    def test_minor_resets_patch(self) -> None:
        self.assertEqual(bump_version("1.2.3", "minor"), "1.3.0")

    def test_major_resets_minor_and_patch(self) -> None:
        self.assertEqual(bump_version("1.2.3", "major"), "2.0.0")

    def test_invalid_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bump_version("1.2", "patch")


if __name__ == "__main__":
    unittest.main()
