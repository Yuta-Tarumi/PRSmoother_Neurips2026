
import numpy as np
rng = np.random.default_rng(seed=12345)
X = rng.standard_normal((40))  
np.save("/data/RB250005/Lorenz96/offset", X)
