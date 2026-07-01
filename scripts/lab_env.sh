# Source inside the TF container. Ensures audio deps without shadowing container numpy.
export PYTHONUSERBASE="$HOME/adf/.pyuser"
export PYTHONPATH="$HOME/adf/.pyuser/lib/python3.12/site-packages:$HOME/adf"
python3 -c "import librosa, soundfile, matplotlib" 2>/dev/null \
  || python3 -m pip install --user -q -c "$HOME/adf/constraints.txt" librosa soundfile matplotlib audioread soxr
