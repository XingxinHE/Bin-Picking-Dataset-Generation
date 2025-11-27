import pybullet as p
import pybullet_data
import time

# 1. Start the simulation
physicsClient = p.connect(p.GUI)
print(pybullet_data.getDataPath())