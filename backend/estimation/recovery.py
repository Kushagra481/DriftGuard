"""Explicit tracking states and uncertainty bounds, with recovery accounting."""
class Recovery:
    def __init__(self):
        self.state = "NORMAL"
        self.lost_at = None
        self.recovery_started = None
        self.durations = []
        self.recovered_at = None

    def step(self, time, visual_quality, imu_quality, sigma, accepted):
        previous = self.state
        if sigma > .75:
            state, cause = "TRACKING_UNCERTAIN", "Position uncertainty exceeded 0.75 m; do not trust spatial anchors."
        elif visual_quality < .03:
            if imu_quality < .55:
                state, cause = "MULTI_SENSOR_DEGRADED", "Both sensor quality estimates are below their usable bounds."
            else:
                state, cause = "INERTIAL_HOLDOVER", "No reliable visual pose; propagating IMU with growing uncertainty."
        elif visual_quality < .22 and imu_quality < .55:
            state, cause = "MULTI_SENSOR_DEGRADED", "Weak visual constraints and degraded inertial measurements."
        elif visual_quality < .22:
            state, cause = "VISUAL_DEGRADED", "Visual feature support, residuals or geometric coverage are weak."
        elif imu_quality < .55:
            state, cause = "IMU_DEGRADED", "Inertial noise, missing packets, saturation or estimated bias is high."
        elif previous not in ("NORMAL", "VISUAL_RECOVERY") or self.recovery_started is not None:
            if accepted is True or (self.recovery_started is not None and accepted is None):
                if self.recovery_started is None:
                    self.recovery_started = time
                state, cause = "VISUAL_RECOVERY", "Consistent visual corrections returned; validating for 0.2 seconds."
                if time-self.recovery_started >= .2:
                    state, cause = "NORMAL", "Visual innovations are consistent and uncertainty is bounded."
                    if self.lost_at is not None:
                        self.durations.append(time-self.lost_at)
                    self.lost_at = self.recovery_started = None
                    self.recovered_at = time
            else:
                state, cause = "VISUAL_DEGRADED", "Visual pose failed the predicted-state consistency gate."
        elif accepted is False:
            state, cause = "VISUAL_DEGRADED", "Visual pose failed the predicted-state consistency gate."
        else:
            state, cause = "NORMAL", "Sensor measurements are consistent and uncertainty is bounded."
        if state not in ("NORMAL", "VISUAL_RECOVERY") and self.lost_at is None:
            self.lost_at = time
        if state not in ("NORMAL", "VISUAL_RECOVERY"):
            self.recovery_started = None
        self.state = state
        return state, cause
