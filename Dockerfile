# S3PO-GS Podman image
# Built and run by SAL's runtime-stress framework via S3POGSAlgorithm
# wrapper at src/algorithms/s3pogs.py. See docs/S3POGS_PODMAN.md for
# the operator-side build / run / troubleshooting guide.
#
# Notes for future maintainers:
# - S3PO-GS compiles native CUDA extensions in four places:
#     * submodules/simple-knn (CUDAExtension over a kd-tree kernel,
#       same package Photo-SLAM / GigaSLAM bundle)
#     * submodules/diff-gaussian-rasterization (CUDAExtension over the
#       3D-GS Gaussian-splat rasterizer, same package Photo-SLAM /
#       GigaSLAM bundle)
#     * croco/models/curope (rotary-position-embedding CUDA kernels;
#       MASt3R encoder uses these)
#     * dust3r/croco/models/curope (a SECOND copy of curope shipped
#       under the in-tree dust3r folder. The upstream SETUP_NOTES is
#       explicit that BOTH must be built or RoPE2D import fails.)
#   The base image must be -devel (provides nvcc) and the strip step
#   below wipes prebuilt artifacts so the in-container PyTorch ABI is
#   the one used. TORCH_CUDA_ARCH_LIST is the explicit gencode list
#   curope/setup.py reads via torch.cuda.get_gencode_flags() (no GPU
#   present at build time).
# - CUDA 11.8 matches DROID/Photo/MASt3R/Giga (all HAMi-bindable on
#   this host) and the upstream SETUP_NOTES.md recipe.
# - The MASt3R foundation-model checkpoint (~2.5 GB) is NOT copied into
#   the image. The bundled mast3r/model.py's from_pretrained checks
#   if its argument is a file path first, then falls back to
#   huggingface_hub.PyTorchModelHubMixin.from_pretrained which fetches
#   "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric" from HF.
#   The wrapper bind-mounts ~/.cache/huggingface so the weights cache
#   across container invocations instead of re-downloading.
# - Build time: ~25-35 minutes on a typical host; one-time per host.
#   Dominated by the simple-knn + diff-gaussian-rasterization + 2×
#   curope CUDA compiles (one per gencode arch each).
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Python 3.10 (matches the upstream SETUP_NOTES.md recipe; the
# environment.yml says 3.11 but it has dependency conflicts and is
# explicitly superseded by the SETUP_NOTES instructions).
#
# OpenGL libs (libgl1 / libegl1 / libosmesa / libglfw3) are needed
# because the upstream pip deps include pyrender / pyOpenGL / glfw,
# which import OpenGL at module load even when GUI is disabled
# (slam.py's gui/slam_gui chain pulls them in transitively).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git build-essential cmake ninja-build pkg-config \
        libeigen3-dev libsuitesparse-dev libopencv-dev \
        libgl1-mesa-glx libegl1 libgles2-mesa libxrandr2 libxinerama1 \
        libxcursor1 libxi6 libxxf86vm1 libosmesa6 \
        libglfw3 libglfw3-dev \
        wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# pip for python3.10. Ubuntu's debian-patched ensurepip refuses to run
# against the system python, so install via get-pip.py instead. This
# is the canonical bootstrap path for deadsnakes-installed Pythons
# (matches GigaSLAM's image).
RUN wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python /tmp/get-pip.py --no-cache-dir 'pip==24.0' 'setuptools==69.5.1' 'wheel' && \
    rm -f /tmp/get-pip.py

# PyTorch 2.1.0 + torchvision 0.16.0 + torchaudio 2.1.0 from the CUDA
# 11.8 wheel index. Matches the upstream SETUP_NOTES.md install
# (conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0
# pytorch-cuda=11.8).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

WORKDIR /s3pogs
COPY . /s3pogs

# Strip prebuilt CUDA artifacts and host-side state (build/ dirs,
# results/, checkpoints/, .so files). These are tied to the host's
# CUDA / glibc / PyTorch ABI and would conflict with the in-container
# compile. The .dockerignore catches most of this but be explicit to
# avoid stale layer reuse on rebuild.
RUN rm -rf build *.so *.egg-info \
        results/ dist/ \
        checkpoints/ \
        submodules/simple-knn/build \
        submodules/simple-knn/*.egg-info \
        submodules/simple-knn/*.so \
        submodules/diff-gaussian-rasterization/build \
        submodules/diff-gaussian-rasterization/*.egg-info \
        submodules/diff-gaussian-rasterization/*.so \
        croco/models/curope/build \
        croco/models/curope/*.so \
        dust3r/croco/models/curope/build \
        dust3r/croco/models/curope/*.so

# Pin numpy<2 to keep the binary ABI stable across the rest of the
# requirements (torch 2.1.0 was built against numpy 1.x; pip would
# otherwise resolve numpy 2.x and trigger ABI breakage at import time
# in opencv / pyrender / open3d / scipy / matplotlib).
RUN pip install --no-cache-dir "numpy<2.0.0"

# Pin numpy + the cu118 torch trio in a constraints file so pip's
# transitive resolver for the upstream deps doesn't silently override
# either. Several upstream deps (wandb, torchmetrics, gradio,
# huggingface-hub) will otherwise re-resolve torch from PyPI and break
# the CUDA toolchain (the GigaSLAM port hit this; see
# docs/GIGASLAM_PODMAN.md "pip-resolver fights").
#
# --ignore-installed blinker: wandb / flask (transitive dep of gradio)
# pull a newer ``blinker`` that pip can't uninstall cleanly over the
# system distutils-installed ``blinker 1.4`` shipped by Ubuntu 22.04.
# Same workaround GigaSLAM's Dockerfile uses.
RUN printf '%s\n' \
        'torch==2.1.0+cu118' \
        'torchvision==0.16.0+cu118' \
        'torchaudio==2.1.0+cu118' \
        'numpy<2.0.0' \
        > /tmp/s3pogs-constraints.txt && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu118 \
        --extra-index-url https://pypi.org/simple \
        --ignore-installed blinker \
        --constraint /tmp/s3pogs-constraints.txt \
        tqdm einops 'evo==1.11.0' open3d lpips plyfile wandb tensorboard \
        trimesh pyrender glfw torchmetrics roma pyquaternion \
        gradio huggingface-hub configargparse addict munch \
        imgviz PyGLM safetensors pyyaml \
        opencv-python scipy matplotlib==3.7.0 pandas \
        scikit-learn pillow imageio && \
    rm -f /tmp/s3pogs-constraints.txt
# evo is pinned to 1.11.0 because S3PO-GS's utils/eval_utils.py calls
# evo.core.trajectory.align_trajectory(), which was REMOVED upstream in
# evo 1.30+. The host conda env (per environment_fixed.yml) ships
# evo 1.11.0 from pip's default resolver at install time, but a fresh
# install pulls 1.36+ which breaks at the first eval_ate() call:
#     AttributeError: module 'evo.core.trajectory' has no attribute
#     'align_trajectory'
# (Stack: slam.py -> SLAM.__init__ -> frontend.run -> eval_ate ->
# evaluate_evo -> trajectory.align_trajectory).

# Build the Gaussian-splat submodules (simple-knn +
# diff-gaussian-rasterization) without build isolation so the
# setup.py scripts can import torch during ext compile.
RUN pip install --no-cache-dir --no-build-isolation submodules/simple-knn
RUN pip install --no-cache-dir --no-build-isolation submodules/diff-gaussian-rasterization

# Build BOTH curope copies (RoPE2D CUDA kernels). The upstream
# SETUP_NOTES.md is explicit that both locations must be built or
# the import chain (slam.py -> mast3r.model -> dust3r.model ->
# RoPE2D) fails. setup.py here uses build_ext --inplace so the .so
# lands next to the .py.
#
# torch.cuda.get_gencode_flags() reads TORCH_CUDA_ARCH_LIST (set in
# ENV above) when no GPU is available at build time, so the
# in-container compile gets the gencode list we want.
RUN cd /s3pogs/croco/models/curope && \
    python setup.py build_ext --inplace
RUN cd /s3pogs/dust3r/croco/models/curope && \
    python setup.py build_ext --inplace

# The simple-knn / diff-gaussian-rasterization / curope .so files
# link against PyTorch's libc10/libtorch. Python's extension loader
# doesn't auto-add torch's lib dir to LD_LIBRARY_PATH at import time,
# so set it here so the import chain resolves cleanly.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

# Provide writable cache dirs that bind-mount targets land on.
RUN mkdir -p /root/.cache/torch/hub /root/.cache/huggingface /output

CMD ["python", "slam.py", "--help"]
