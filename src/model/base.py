import numpy as np


class Circle:

    def __init__(self, cx, cy, r, cost=None):
        self.cx = cx
        self.cy = cy
        self.r = r
        self.cost = cost

    def evaluate_cost(self, pts):
        """
        Sum of absolute distances from every point to the circle

        Parameters:
        pts: array of shape (N, 2)

        Returns:
        float: the cost, also stored in self.cost
        """
        # distance of each point from the center
        d = np.hypot(pts[:, 0] - self.cx, pts[:, 1] - self.cy)

        # distance of each point from the circle itself
        self.cost = np.sum(np.abs(d - self.r))
        return self.cost

    def params(self):
        return np.array([self.cx, self.cy, self.r])
