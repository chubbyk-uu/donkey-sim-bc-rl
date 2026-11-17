# Setup and connection

Run commands from the repository root. Use WSL2 for Python and Windows Donkey
Simulator on TCP 9091. Existing environment: `conda activate rlpy312`.

## New environment

```bash
conda create -n rlpy312 python=3.12
conda activate rlpy312
# Install CUDA-compatible PyTorch and torchvision before the remaining packages.
pip install -r requirements.txt
```

[requirements.txt](../../requirements.txt) records the working dependencies.
The local setup used torch 2.12.0 / torchvision 0.27.0 cu130 builds; choose builds
compatible with the target machine. DINOv2 is fetched with `torch.hub` and cached
locally; upstream backbone/source revisions are not pinned by the model contract.
Actual encoder tensor hashes are recorded in evaluation metadata.

The requirements pin upstream `gym-donkeycar` commit
`a1f4ca6961e17a6929c1f2e883358a1edafb479d`. The PyPI SDK uses the old Gym API.
An editable SDK checkout is possible for development; record its revision and
do not silently substitute upstream HEAD for the pinned dependency.

## Retrieve project models

```bash
git lfs install
git lfs pull
```

The accepted student requires both:

```text
models/visual_student_no_motion_v1/student.pt
models/dinov2_geometry_15d/proj15_lags_ridge.pt
```

The DINOv2 pretrained backbone is downloaded separately. Teacher models and
training datasets are not required for student inference. Raw datasets, feature
caches and replay buffers are excluded from Git and must be restored to rerun fits.

## Connect to Windows

The WSL2 host IP can change after reboot. Derive it again:

```bash
export DONKEY_SIM_HOST="$(ip route | awk '/default/ {print $3}')"
export DONKEY_SIM_PORT=9091
nc -vz "$DONKEY_SIM_HOST" "$DONKEY_SIM_PORT"
```

For the student, open the generated loop with timeScale 2, FPS 120, random light
and trees enabled, cones disabled, useSeed disabled. Client metadata records
saved menu values, not independent live engine state.

Continue with the [quick start](../../README.md#quick-start).
Fixed-light VAE and Mountain need different settings; see [models](models.md).
