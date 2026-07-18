# Source inside the PyTorch container for the CREATION half (XTTS).
#
# Deliberately a SEPARATE user-site from lab_env_pt.sh: the detection half runs
# transformers 5.x for wav2vec2-XLS-R, but coqui-tts calls isin_mps_friendly,
# which transformers 5 removed. coqui-tts declares "transformers>=4.57" with no
# upper bound, so its metadata does not catch this - hence the hard pin here.
# Keeping the two user-sites apart stops the XTTS pin from silently changing the
# environment behind the published detection results.
export PYTHONUSERBASE="$HOME/adf/.pyuser_xtts"
export PYTHONPATH="$HOME/adf/.pyuser_xtts/lib/python3.12/site-packages:$HOME/adf"
# Overrides the container-s /etc/pip/constraint.txt, which pins tensorboard==2.16.2
# and blocks coqui-tts-trainer (needs >=2.17). Ours pins numpy only.
export PIP_CONSTRAINT="$HOME/adf/constraints.txt"
export COQUI_TOS_AGREED=1
export HF_HOME="$HOME/.cache/huggingface"
