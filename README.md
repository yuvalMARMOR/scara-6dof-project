# SCARA 6-DOF Project

Full project of a 6-DOF SCARA robotic arm.

Core development is done in **Python** for kinematics, workspace analysis, plotting, simulations, and animations.

**ROS 2** is used for visualization in **RViz** and **Gazebo** with URDF/Xacro models.

## 🚀 Features

- Forward kinematics and inverse kinematics calculations
- Workspace plots and 3D visualizations
- Python-based simulations and animations
- ROS 2 URDF/Xacro robot model
- RViz and Gazebo visualization
- Academic report and setup instructions

## 📂 Repository Structure

```text
.
├── simulation.py
├── imges.py
├── final_ws/
│   └── src/
├── docs/
│   ├── SCARA-6DOF-Final-Report.pdf
│   └── Setup-Instructions.pdf
├── LICENSE
└── .gitignore
```

## 🐍 Python

Install the required Python packages:

```bash
pip install numpy matplotlib
```

Run the main simulation:

```bash
python simulation.py
```

Run the plotting and workspace visualization script:

```bash
python imges.py
```

## 🤖 ROS 2 Visualization

The ROS 2 workspace is located in:

```text
final_ws/
```

Build the workspace:

```bash
cd final_ws
colcon build
```

Source the workspace on Linux:

```bash
source install/setup.bash
```

Source the workspace on Windows:

```bash
call install\setup.bat
```

Launch the robot visualization:

```bash
ros2 launch robot_arm_description display.launch.py
```

## 📄 Documentation

- [SCARA-6DOF-Final-Report.pdf](docs/SCARA-6DOF-Final-Report.pdf)
- [Setup-Instructions.pdf](docs/Setup-Instructions.pdf)

## 📜 License

This project is licensed under the [MIT License](LICENSE).

