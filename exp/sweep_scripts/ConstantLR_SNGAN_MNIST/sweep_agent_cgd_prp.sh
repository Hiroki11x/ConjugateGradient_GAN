# Environemental Scripts Setup (Anonymize)

# ======== Module, Virtualenv and Other Dependencies======
source ../../env_common.sh
echo "PYTHON Environment: $PYTHON_PATH"
export PYTHONPATH=.
export PATH=$PYTHON_PATH:$PATH
which python

# ======== Copy Data ======

#======== Configurations ========
WANDB_ENTITY="XXXXXX"
export WANDB_ENTITY

LR_SCHEDULE="ConstantLR"
DATASET="mnist"
MODEL_NAME="SNGAN"
OPTIMIZER="cgd_prp"
WANDB_PROJECT_NAME="${LR_SCHEDULE}_${DATASET}_${MODEL_NAME}_${OPTIMIZER}"
export WANDB_PROJECT_NAME

# ======== Configurations ========
cd ../../../

SWEEP_PROJ="${WANDB_ENTITY}/${WANDB_PROJECT_NAME}"
SWEEP_ID_FILENAME="sweep_id_${WANDB_PROJECT_NAME}.txt"
if [ ! -e $SWEEP_ID_FILENAME ]; then
    wandb sweep sweep_config/${WANDB_PROJECT_NAME}.yaml 2> $SWEEP_ID_FILENAME
    SWEEP_ID=$(cat $SWEEP_ID_FILENAME | grep "Created sweep with ID" | cut -d ':' -f 3 | cut -c 2-)
    echo $SWEEP_ID > $SWEEP_ID_FILENAME
fi
if [ -e $SWEEP_ID_FILENAME ]; then
    SWEEP_ID=$(cat $SWEEP_ID_FILENAME)
fi
CMD_EXECUTE="wandb agent $SWEEP_PROJ/$SWEEP_ID"

echo "Job started on $(date)"
echo "................................"
echo "[CMD_EXECUTE] :  $CMD_EXECUTE"
echo ""
eval $CMD_EXECUTE
echo "Job done on $(date)"