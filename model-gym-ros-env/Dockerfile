# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ros:foxy

SHELL ["/bin/bash", "-c"]

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
    nano \
    vim \
    python3-pip \
    libeigen3-dev \
    tmux \
    ros-foxy-rviz2
RUN apt-get -y dist-upgrade
RUN pip3 install transforms3d

# f1tenth gym
RUN git clone https://github.com/f1tenth/f1tenth_gym
RUN cd f1tenth_gym && \
    pip3 install -e .

# ros2 gym bridge
RUN mkdir -p sim_ws/src/f1tenth_gym_ros
COPY ./f1tenth_gym_ros /sim_ws/src/f1tenth_gym_ros
RUN source /opt/ros/foxy/setup.bash && \
    cd sim_ws/ && \
    apt-get update --fix-missing && \
    rosdep install -i --from-path src --rosdistro foxy -y && \
    colcon build

# our dependencies
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc && \
    echo "source /sim_ws/install/setup.bash" >> ~/.bashrc && \
    pip install --upgrade pip && \
    source /opt/ros/foxy/setup.bash && \
    cd /sim_ws/src && \
    git clone https://github.com/f1tenth/f1tenth_gym.git && \
    cd f1tenth_gym && \
    git checkout v1.0.0 && \
    cd .. && \
    ros2 pkg create car_node --build-type ament_python --dependencies rclpy && \
    mv f1tenth_gym/* car_node/car_node

# COPY ./car_node/car_node /sim_ws/src/car_node/car_node
COPY ./car_node/setup.py /sim_ws/src/car_node/setup.py

COPY ./car_node/car_node/main.py /sim_ws/src/car_node/car_node/main.py
COPY ./car_node/car_node/wrapper.py /sim_ws/src/car_node/car_node/wrapper.py
COPY ./car_node/car_node/map_utility.py /sim_ws/src/car_node/car_node/map_utility.py

RUN pip install setuptools==65.7.0 && \
    pip install -e /sim_ws/src/car_node/car_node && \
    pip install stable_baselines3 && \
    pip install 'shimmy>=0.2.1' && \
    cd sim_ws/ && \
    colcon build

#     pip install gym[atari,accept-rom-license]==0.19.0 && \
#     pip install 'stable-baselines3[extra]' && \
#     pip install -e /sim_ws/src/car_node/car_node && \
#     source /opt/ros/foxy/setup.bash && \
#     cd sim_ws/ && \
#     colcon build

WORKDIR '/sim_ws'
ENTRYPOINT ["/bin/bash"]
