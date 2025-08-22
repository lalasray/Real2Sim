# Paths you control
PROJECT_ROOT=/mimer/NOBACKUP/groups/focs/virtualenv/real2sim
ENV_DIR=$PROJECT_ROOT/conda_envs/myenv
PKGS_DIR=$PROJECT_ROOT/conda_pkgs

mkdir -p "$(dirname "$ENV_DIR")" "$PKGS_DIR"

# Keep caches off $HOME
conda config --add envs_dirs "$(dirname "$ENV_DIR")" || true
conda config --add pkgs_dirs "$PKGS_DIR" || true
conda config --set channel_priority strict || true
conda config --set solver libmamba || true  # if supported

# Build/update from your environment.yml (assumed in $SLURM_SUBMIT_DIR)
conda env create -p "$ENV_DIR" -f environment.yml || \
conda env update -p "$ENV_DIR" -f environment.yml --prune

echo "Activating:"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
python -V
which python
