# F1Tenth Gym Environment Setup

## Install F1Tenth Gym

```bash
git clone https://github.com/f1tenth/f1tenth_gym
cd f1tenth_gym
conda create -n f1tenth_gym python=3.8
conda activate f1tenth_gym
pip install -e .
pip install -r requirements.txt
```

If there are any error during installation([source](https://github.com/freqtrade/freqtrade/issues/8376#issuecomment-1519257211)):

```bash
pip install wheel==0.38.4
```

### Test the installation

```bash
cd examples
python3 waypoint_follow.py
```
