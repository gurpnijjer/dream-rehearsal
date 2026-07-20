import gym
import numpy as np


def _register_custom_envs():
    """Register MiniGrid sizes the Continual-Dreamer pair uses that stock MiniGrid omits.

    Stock ships DoorKey 5x5/6x6/8x8/16x16 only; the paper's interfering pair uses
    DoorKey-9x9. entry_point resolves lazily at gymnasium.make() time, by which point
    the minigrid package is imported (via the wrappers import in MiniGrid.__init__).
    """
    import gymnasium

    if "MiniGrid-DoorKey-9x9-v0" not in gymnasium.registry:
        gymnasium.register(
            id="MiniGrid-DoorKey-9x9-v0",
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 9},
        )


_register_custom_envs()


class MiniGrid:
    """DreamerV3-torch wrapper for MiniGrid (gymnasium) -> old-gym dict-obs interface.

    Emits 64x64x3 RGB partial-view images (paper-standard RGBImgPartialObsWrapper),
    routed to the CNN encoder via cnn_keys='image' (mirrors the crafter image path).
    Adapts gymnasium's 5-tuple step / (obs, info) reset to the 4-tuple step / obs-only
    reset that NM512's wrappers (OneHotAction, TimeLimit, ...) expect.
    """

    metadata = {}

    def __init__(self, task, size=(64, 64), seed=0):
        import gymnasium
        from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

        env = gymnasium.make(f"MiniGrid-{task}-v0")
        env = RGBImgPartialObsWrapper(env)  # obs dict gets an RGB 'image' of the agent view
        env = ImgObsWrapper(env)            # obs -> just the image array
        self._env = env
        self._size = tuple(size)
        self._seed = int(seed)
        self._first = True
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (*self._size, 3), dtype=np.uint8),
                "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            }
        )

    @property
    def action_space(self):
        # MiniGrid is Discrete(7). NM512's OneHotAction reads .n; .discrete flag mirrors crafter.
        space = gym.spaces.Discrete(int(self._env.action_space.n))
        space.discrete = True
        return space

    def _resize(self, image):
        if image.shape[:2] != self._size:
            import cv2

            image = cv2.resize(image, self._size, interpolation=cv2.INTER_AREA)
        return image.astype(np.uint8)

    def step(self, action):
        # OneHotAction normally converts to int upstream; stay defensive against a leaked one-hot.
        action = int(np.asarray(action).argmax()) if np.ndim(action) > 0 else int(action)
        image, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        obs = {
            "image": self._resize(image),
            "is_first": False,
            "is_last": done,
            # truncation is the time-limit, not a true terminal -> only `terminated` counts.
            "is_terminal": bool(terminated),
        }
        return obs, np.float32(reward), done, info

    def render(self):
        return self._env.render()

    def reset(self):
        # Seed once for reproducibility, then let layouts vary across episodes.
        if self._first:
            image, info = self._env.reset(seed=self._seed)
            self._first = False
        else:
            image, info = self._env.reset()
        obs = {
            "image": self._resize(image),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs
