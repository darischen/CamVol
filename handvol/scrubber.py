class VolumeScrubber:
    def __init__(self, sensitivity=80, smoothing=0.3):
        self.anchor_y = None
        self.anchor_vol = None
        self.sensitivity = sensitivity
        self.smoothing = smoothing
        self.smoothed_y = None

    def enter(self, tip_y, current_vol):
        self.anchor_y = tip_y
        self.anchor_vol = current_vol
        self.smoothed_y = tip_y

    def update(self, tip_y):
        self.smoothed_y = (self.smoothing * tip_y +
                          (1 - self.smoothing) * self.smoothed_y)
        delta = self.anchor_y - self.smoothed_y
        new_vol = self.anchor_vol + self.sensitivity * delta
        return max(0, min(100, new_vol))

    def exit(self):
        self.anchor_y = None

    @property
    def active(self):
        return self.anchor_y is not None
