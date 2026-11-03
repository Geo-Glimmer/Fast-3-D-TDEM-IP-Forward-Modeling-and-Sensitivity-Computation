"""Minimal structural checks for the provided primary 14-pole model."""

import unittest
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from tem_ip_ipnet.jax_model import JaxIPNet
from tem_ip_ipnet.physics import cole_cole_kernel, multi_debye_kernel


class TestIPNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = JaxIPNet(ROOT / "models/ipnet_14poles_100k_jax.npz")

    def test_architecture(self):
        self.assertEqual(self.model.config, {"n_poles": 14, "width": 512, "fourier_size": 64, "n_blocks": 4})

    def test_physical_output(self):
        tau = 0.01
        c = 0.5
        poles, weights = self.model(np.array([[np.log10(tau), c]], dtype=np.float32))
        p, w = np.asarray(poles[0]), np.asarray(weights[0])
        self.assertEqual(p.shape, (14,))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all(p > 0))
        self.assertTrue(np.all(w >= 0))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        f = np.logspace(-2, 6, 100)
        a = cole_cole_kernel(tau, c, f)
        b = multi_debye_kernel(p, w, f)
        self.assertLess(float(np.mean(abs(a - b))), 0.1)


if __name__ == "__main__":
    unittest.main()
