"""Utility to generate a dataset of *N* random shapes + loads."""
import numpy as np
from pathlib import Path
from tqdm import tqdm  # progress bar
from cosserat_rod_model import CosseratRodModel

def random_dataset(N: int = 100,
                   save_path: str | Path = "dataset.npz",
                   tendon_count: int = 4):
    model = CosseratRodModel(tendon_count=tendon_count)
    shapes = []      # SE(3) 4×4 tip frames
    tensions = []    # τ vectors
    wrenches = []    # external tip wrench (f_ext||l_ext)
    rng = np.random.default_rng(42)
    for _ in tqdm(range(N)):
        tau = rng.uniform(1.0, 5.0, tendon_count)  # N
        f_ext = rng.normal(0, 0.1, 3)              # small external
        l_ext = rng.normal(0, 0.05, 3)
        T, _ = model.forward_kinematics(tau, f_ext, l_ext, return_states=True)
        shapes.append(T)
        tensions.append(tau)
        wrenches.append(np.hstack([f_ext, l_ext]))
    np.savez_compressed(save_path,
                       shapes=np.array(shapes),
                       tensions=np.array(tensions),
                       wrenches=np.array(wrenches))
    print(f"✔ Dataset saved to {save_path}")

if __name__ == "__main__":
    random_dataset()