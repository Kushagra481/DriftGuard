"""15-state right-multiplicative error-state Kalman filter, no smoothing."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import cho_factor, cho_solve
from backend.simulation.geometry import GRAVITY, skew


class ESKF:
    # Error-state order: p, v, theta (right-local), bg, ba.
    def __init__(self, position=None, rotation=None, adaptive=False):
        self.p = np.zeros(3) if position is None else position.copy()
        self.v = np.zeros(3)
        self.r = Rotation.identity() if rotation is None else rotation
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.P = np.diag([.03**2]*3 + [.1**2]*3 + [.015**2]*3
                         + [.02**2]*3 + [.08**2]*3)
        self.adaptive = adaptive
        self.last_mahalanobis = 0.
        self.rejected = False
        self.accepted_updates = 0

    def propagate(self, gyro, accel, dt, quality=1.):
        omega, specific = gyro-self.bg, accel-self.ba
        R = self.r.as_matrix()
        # Midpoint orientation for acceleration; exact SO(3) exponential.
        mid = self.r * Rotation.from_rotvec(omega * dt*.5)
        aw = mid.apply(specific) + GRAVITY
        self.p += self.v*dt + .5*aw*dt*dt
        self.v += aw*dt
        self.r = self.r * Rotation.from_rotvec(omega*dt)
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -R @ skew(specific)
        F[3:6, 12:15] = -R
        F[6:9, 6:9] = -skew(omega)
        F[6:9, 9:12] = -np.eye(3)
        Phi = np.eye(15) + F*dt + .5*(F@F)*(dt**2)
        # Continuous process variance is a filter tuning model, not privileged
        # access to the simulator noise/bias values.
        inflation = min(100., 1/max(quality, .01)) if self.adaptive else 1.
        q = np.array([1e-9]*3 + [.04**2]*3 + [.008**2]*3
                     + [.002**2]*3 + [.012**2]*3) * inflation
        self.P = Phi@self.P@Phi.T + np.diag(q)*dt
        self.P = (self.P+self.P.T)*.5
        if not np.all(np.isfinite(self.P)) or not np.all(np.isfinite(self.p)):
            raise FloatingPointError("Non-finite ESKF state or covariance")

    def update(self, position, rotation, covariance, gate=True):
        H = np.zeros((6, 15))
        H[:3, :3] = np.eye(3)
        H[3:, 6:9] = np.eye(3)
        innovation = np.r_[position-self.p, (self.r.inv()*rotation).as_rotvec()]
        S = H@self.P@H.T + covariance
        factor = cho_factor(S, lower=True, check_finite=False)
        self.last_mahalanobis = float(innovation@cho_solve(factor, innovation, check_finite=False))
        self.rejected = bool(gate and self.last_mahalanobis > 22.458)  # chi²(6), 99.9%
        if self.rejected:
            return False
        gain = cho_solve(factor, H@self.P, check_finite=False).T
        delta = gain@innovation
        self.p += delta[:3]
        self.v += delta[3:6]
        self.r = self.r * Rotation.from_rotvec(delta[6:9])
        self.bg += delta[9:12]
        self.ba += delta[12:15]
        # Joseph update and right-error reset, not a covariance shortcut.
        A = np.eye(15)-gain@H
        self.P = A@self.P@A.T + gain@covariance@gain.T
        reset = np.eye(15)
        reset[6:9, 6:9] -= .5*skew(delta[6:9])
        self.P = reset@self.P@reset.T
        self.P = (self.P+self.P.T)*.5
        self.accepted_updates += 1
        return True

    def uncertainty(self):
        return (float(np.sqrt(np.trace(self.P[:3, :3]))),
                float(np.rad2deg(np.sqrt(np.trace(self.P[6:9, 6:9])))))
