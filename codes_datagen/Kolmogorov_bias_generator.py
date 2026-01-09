
import numpy as np
for seed in range(42, 47):
    rng = np.random.default_rng(seed=seed)
    X = rng.standard_normal((128, 128))  
    np.save(f"/data/RB250005/Kolmogorov/offset_seed{seed}", X)

