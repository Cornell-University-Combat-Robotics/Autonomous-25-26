class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0), max_windup=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Anti-windup and output constraints
        self.min_out, self.max_out = output_limits
        self.max_windup = max_windup
        
        # State variables
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        if dt <= 0.0:
            return 0.0

        # Proportional
        p = self.kp * error

        # Integral (with anti-windup clamping)
        self.integral += error * dt
        # Prevent the integral from building up infinitely if the bot is physically stuck
        self.integral = max(min(self.integral, self.max_windup), -self.max_windup)
        i = self.ki * self.integral

        # Derivative
        d = self.kd * (error - self.prev_error) / dt

        # Save current error for next derivative calculation
        self.prev_error = error

        # Calculate total output and clamp to motor limits (-1 to 1)
        output = p + i + d
        return max(min(output, self.max_out), self.min_out)
        
    def reset(self):
        """Call this if Huey is recovering or teleported, to clear the integral memory."""
        self.prev_error = 0.0
        self.integral = 0.0