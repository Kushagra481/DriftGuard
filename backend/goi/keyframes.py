"""Observable geometry chooses visual correction frames, never rejects by GOI."""
import numpy as np


class KeyframePolicy:
    def __init__(self):
        self.information = np.eye(6)*1e-3
        self.last_time = -10.
        self.count = 0

    def choose(self, time, geometry, quality):
        current = geometry["information"]
        # Relative log-determinant information utility; accumulated information
        # decays to represent a local keyframe window, not an EKF covariance.
        prior = self.information
        _, before = np.linalg.slogdet(prior+np.eye(6)*1e-6)
        _, after = np.linalg.slogdet(prior+current+np.eye(6)*1e-6)
        gain = float(max(0, after-before))
        weak_value, weak_vectors = np.linalg.eigh(prior)
        weak = weak_vectors[:, 0]
        improvement = float(weak@current@weak/max(weak_value[0], 1e-6))
        elapsed = time-self.last_time
        coverage = geometry["coverage"]
        parallax = geometry["parallax"]
        well_conditioned = geometry["conditionNumber"] < 1e6
        # Maximum 10 Hz visual correction, minimum healthy correction every
        # 0.3 s. Fixed-interval ablation uses the same 0.3 s maximum interval.
        useful = gain > 2.0 or improvement > .3 or parallax > .01
        chosen = bool(quality > .035 and elapsed >= .095 and
                      ((useful and coverage >= .25 and well_conditioned) or elapsed >= .3))
        if chosen:
            self.information = .65*prior + current
            self.last_time = time
            self.count += 1
        reason = (f"{'Selected' if chosen else 'Skipped'}: Δlogdet={gain:.3f}; "
                  f"weak-direction gain={improvement:.3f}; coverage={coverage:.2f}; "
                  f"parallax={parallax:.5f} rad; κ={geometry['conditionNumber']:.1f}")
        return chosen, gain, reason
