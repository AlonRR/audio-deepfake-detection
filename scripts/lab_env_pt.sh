# Source inside the PyTorch container. librosa/soundfile already present; add transformers.
export PYTHONUSERBASE="$HOME/adf/.pyuser_pt"
export PYTHONPATH="$HOME/adf/.pyuser_pt/lib/python3.12/site-packages:$HOME/adf"
python3 -c "import transformers" 2>/dev/null \
  || python3 -m pip install --user -q -c "$HOME/adf/constraints.txt" transformers
