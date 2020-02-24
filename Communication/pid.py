class PID:

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, reference=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.reference = reference
        self.previous_error = 0.0
        self.accumulated_error = 0.0

    def control(self, input):

        # Calculate new error and accumulate
        error = self.reference - input
        self.accumulated_error += error
        error_diff = error - self.previous_error

        # Calculate control output
        P_term = self.Kp * error
        D_term = self.Kd * error_diff
        I_term = self.Ki * self.accumulated_error
        control = P_term + I_term + D_term
        # Store current error
        self.previous_error = error
        # Return control value
        return control

