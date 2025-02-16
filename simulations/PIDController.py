import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=100, output_limit = 10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        self.integral_limit = integral_limit
        self.output_limit = output_limit

    def compute(self, error, dt):
        p = self.kp*error #proportional: k_prop * pitch_err
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self.integral
        d = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        output = p + i + d
        return np.clip(output, -self.output_limit, self.output_limit)