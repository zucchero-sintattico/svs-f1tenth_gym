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

### Test the installation

```bash
cd examples
python3 waypoint_follow.py
```
