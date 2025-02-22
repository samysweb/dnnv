import unittest

from dnnv.verifiers.neurify import Neurify

from system_tests.test_verifiers.utils import VerifierTests


@unittest.skipIf(not Neurify.is_installed(), "Neurify is not installed")
class NeurifyVerifierTests(VerifierTests, unittest.TestCase):
    def initialize(self):
        self.verifier = Neurify
        self.is_complete = False


if __name__ == "__main__":
    unittest.main()
