import threading

class TrainingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stats = {
            "generation": 0,
            "level": 1,
            "games_played": 0,
            "win_rate": 0.0,
            "epsilon": 0.0,
            "loss": 0.0,
            "mean_score": 0.0
        }
        self.controls = {
            "paused": False,
            "save_requested": False,
            "learning_rate": 0.0005
        }

    def update_stats(self, **kwargs):
        with self.lock:
            self.stats.update(kwargs)

    def get_stats(self):
        with self.lock:
            return self.stats.copy()
            
    def get_controls(self):
        with self.lock:
            return self.controls.copy()
            
    def set_control(self, key, value):
        with self.lock:
            if key in self.controls:
                self.controls[key] = value

# Global instance
state = TrainingState()
