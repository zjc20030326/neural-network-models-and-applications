import glob
from pathlib import Path
from collections import deque

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import wavfile
from scipy.ndimage import binary_dilation
from scipy.stats import kurtosis
from PIL import Image


def problem3_hopf_simulation():
    tauE = 1.0
    MEE, MEI, MIE, MII = 2.0, 3.0, 3.0, 1.0
    hE, hI = 2.0, -1.0

    tauI_H = tauE * (1 + MII) / (MEE - 1)
    omega_H = np.sqrt((MEI * MIE - (MEE - 1) * (1 + MII)) / (tauE * tauI_H))
    print("tauI_H =", tauI_H, "omega_H =", omega_H)

    def rhs(t, y, tauI):
        vE, vI = y
        zE = MEE * vE - MEI * vI + hE
        zI = MIE * vE - MII * vI + hI
        return [(-vE + max(zE, 0.0)) / tauE,
                (-vI + max(zI, 0.0)) / tauI]

    for tauI in [1.0, 1.8, 2.0, 2.2, 3.0]:
        sol = solve_ivp(lambda t, y: rhs(t, y, tauI), [0, 200], [1.2, 0.8],
                        max_step=0.05, rtol=1e-7, atol=1e-9)
        tail = sol.y[:, sol.t > 100]
        print("tauI=", tauI,
              "min=", tail.min(axis=1),
              "max=", tail.max(axis=1),
              "std=", tail.std(axis=1))


def problem4_direction_mle(seed=0, runs=2000):
    rng = np.random.default_rng(seed)
    N = 36
    pref = np.linspace(0.0, 180.0, N, endpoint=False)
    sigma = 20.0 * np.ones(N)
    Tobs = 2.0
    s_true = 75.0
    grid = np.linspace(0.0, 180.0, 1801)

    def rates(s):
        return np.exp(-0.5 * ((s - pref) / sigma) ** 2)

    F_grid = np.exp(-0.5 * ((grid[:, None] - pref[None, :]) / sigma[None, :]) ** 2)
    logF_grid = np.log(F_grid + 1e-300)

    def mle(counts):
        ll = (counts[None, :] * logF_grid).sum(axis=1) - Tobs * F_grid.sum(axis=1)
        return grid[np.argmax(ll)]

    fisher = Tobs * np.sum(rates(s_true) * (s_true - pref) ** 2 / sigma ** 4)
    estimates = []
    for _ in range(runs):
        counts = rng.poisson(Tobs * rates(s_true))
        estimates.append(mle(counts))
    estimates = np.array(estimates)
    print("Fisher information =", fisher)
    print("CRLB =", 1.0 / fisher)
    print("mean =", estimates.mean())
    print("bias =", estimates.mean() - s_true)
    print("variance =", estimates.var(ddof=1))
    print("MSE =", np.mean((estimates - s_true) ** 2))


def problem5_bss_audio():
    from sklearn.decomposition import FastICA, PCA

    wavs = sorted(Path('.').rglob('110000001mix*.wav'))
    if len(wavs) < 3:
        wavs = sorted(Path('/mnt/data').rglob('110000001mix*.wav'))
    wavs = wavs[:3]
    if len(wavs) != 3:
        raise FileNotFoundError('Need three mixed wav files named 110000001mix*.wav')

    signals = []
    fs0 = None
    for p in wavs:
        fs, x = wavfile.read(str(p))
        if fs0 is None:
            fs0 = fs
        if x.dtype == np.uint8:
            x = (x.astype(np.float64) - 128.0) / 128.0
        else:
            x = x.astype(np.float64)
            x = x / (np.max(np.abs(x)) + 1e-12)
        signals.append(x)
    X = np.vstack(signals).T
    X = X - X.mean(axis=0)

    pca = PCA(n_components=3, whiten=True, random_state=0)
    S_pca = pca.fit_transform(X)
    ica = FastICA(n_components=3, whiten='unit-variance', random_state=0,
                  max_iter=2000, tol=1e-5)
    S_ica = ica.fit_transform(X)

    def standardize(S):
        return (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-12)

    def metrics(name, S, Xrec):
        Z = standardize(S)
        C = np.corrcoef(Z.T)
        C2 = np.corrcoef((Z ** 2).T)
        off = np.mean(np.abs(C[np.triu_indices(3, 1)]))
        off2 = np.mean(np.abs(C2[np.triu_indices(3, 1)]))
        kur = np.mean(np.abs(kurtosis(Z, fisher=True, axis=0)))
        rec = np.linalg.norm(X - Xrec) / np.linalg.norm(X)
        print(name, 'rec=', rec, 'mean_abs_corr=', off,
              'mean_abs_sqcorr=', off2, 'mean_abs_kurtosis=', kur)
        return rec, off, off2, kur

    metrics('PCA', S_pca, pca.inverse_transform(S_pca))
    metrics('FastICA', S_ica, ica.inverse_transform(S_ica))

    outdir = Path('bss_outputs')
    outdir.mkdir(exist_ok=True)
    for name, S in [('pca', S_pca), ('fastica', S_ica)]:
        for j in range(3):
            y = S[:, j]
            y = y / (np.max(np.abs(y)) + 1e-12)
            wavfile.write(str(outdir / f'{name}_source{j+1}.wav'), fs0,
                          (32767 * y).astype(np.int16))


def problem7_maze_solver(target_xy=(320, 249)):
    candidates = list(Path('.').rglob('maze.jpg')) + list(Path('/mnt/data').rglob('maze.jpg'))
    if not candidates:
        raise FileNotFoundError('maze.jpg not found')
    maze_path = candidates[0]
    img = Image.open(str(maze_path)).convert('RGB')
    arr = np.array(img)

    bright = (arr[:, :, 0] > 100) & (arr[:, :, 1] > 100) & (arr[:, :, 2] > 100)
    ys, xs = np.where(bright)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    inside = np.zeros(bright.shape, dtype=bool)
    inside[y0:y1 + 1, x0:x1 + 1] = True

    score = arr[:, :, 0].astype(int) + arr[:, :, 1].astype(int) - 2 * arr[:, :, 2].astype(int)
    yellow = (score > 120) & (arr[:, :, 0] > 120) & (arr[:, :, 1] > 100)
    ys0, xs0 = np.where(yellow)
    start_yx = (int(round(ys0.mean())), int(round(xs0.mean())))

    wall = bright.copy()
    wall[yellow] = False
    wall = binary_dilation(wall, iterations=2)
    free = inside & (~wall)
    sy, sx = start_yx
    free[sy - 3:sy + 4, sx - 3:sx + 4] = True

    tx, ty = target_xy
    target_yx = (ty, tx)
    if not free[target_yx]:
        print('The selected target is on a wall after preprocessing.')
        return None

    H, W = free.shape
    dist = -np.ones((H, W), dtype=np.int32)
    prev = np.full((H, W, 2), -1, dtype=np.int16)
    q = deque([start_yx])
    dist[start_yx] = 0
    while q:
        y, x = q.popleft()
        if (y, x) == target_yx:
            break
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and free[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                prev[ny, nx] = [y, x]
                q.append((ny, nx))

    if dist[target_yx] < 0:
        print('Target is unreachable.')
        return None

    path = []
    y, x = target_yx
    while y >= 0 and x >= 0:
        path.append((y, x))
        py, px = prev[y, x]
        if py < 0:
            break
        y, x = int(py), int(px)
    path = path[::-1]
    print('start (x,y)=', (sx, sy), 'target (x,y)=', target_xy,
          'shortest length=', dist[target_yx])

    overlay = arr.copy()
    pm = np.zeros((H, W), dtype=bool)
    for y, x in path:
        pm[y, x] = True
    pm = binary_dilation(pm, iterations=1)
    overlay[pm] = [255, 0, 0]
    overlay[sy - 4:sy + 5, sx - 4:sx + 5] = [255, 255, 0]
    overlay[ty - 4:ty + 5, tx - 4:tx + 5] = [0, 255, 0]
    Image.fromarray(overlay).save('maze_solution.png')
    return path


if __name__ == '__main__':
    problem3_hopf_simulation()
    problem4_direction_mle()
    problem5_bss_audio()
    problem7_maze_solver(target_xy=(320, 249))
