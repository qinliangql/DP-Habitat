
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
python -u -m habitat_baselines.run \
  --config-name=pointnav/ppo_pointnav.yaml \
  habitat_baselines.evaluate=True