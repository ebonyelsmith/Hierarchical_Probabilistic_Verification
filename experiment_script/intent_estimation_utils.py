import numpy as np
from collections import deque


class ControlGainEstimator:
    def __init__(self, window_size=10, dt=0.1):
        """
        Online MLE for estimating the other drone's control gain.

        Args:
            window_size (int): Number of recent observations to consider for estimation.
        """
        self.window_size = window_size
        self.dt = dt

        # Sliding window for storing recent observations
        self.y_history = deque(maxlen=window_size)
        self.u_history = deque(maxlen=window_size)

        self.prev_state = None

    def update_window(self, state, act_other):
        """
        Update the sliding window with new observations.

        Args:
            state (np.ndarray): Current state of both drones.
            act_other (np.ndarray): Action taken by the other drone.
        """
        if self.prev_state is None:
            self.prev_state = state.copy()
            return
        
        # Compute velocity increments for the other drone: (v_next - v_prev)/dt
        delta_v1_other = (state[7] - self.prev_state[7]) / self.dt
        delta_v2_other = (state[9] - self.prev_state[9]) / self.dt
        delta_v3_other = (state[11] - self.prev_state[11]) / self.dt

        y_t = np.array([delta_v1_other, delta_v2_other, delta_v3_other])
        u_t = act_other
        self.y_history.append(y_t)
        self.u_history.append(u_t)

        self.prev_state = state.copy()

    def estimate_control_gain(self):
        """
        Estimate control gain of other gain using MLE (least squares over sliding window).
        Returns:
            np.ndarray: Estimated control gain scalar.
        """
        if len(self.y_history) < self.window_size:
            # Not enough data to estimate
            return None
        
        Y = np.concatenate(list(self.y_history))
        U = np.concatenate(list(self.u_history))

        # least squares MLE: cg2_hat = (U^T Y) / (U^T U)
        numerator = np.dot(U, Y)
        denominator = np.dot(U, U)
        if denominator == 0:
            return None
        
        cg2_hat = numerator / denominator
        return cg2_hat









