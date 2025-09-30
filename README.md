# SCARA 6-DOF Project

Full project of a 6-DOF SCARA robotic arm.  
Core development is in **Python** (kinematics, workspace, plots).  
**ROS 2** is used only for visualization in **RViz** and **Gazebo** with URDF/Xacro.

## 🚀 Features
- Forward & Inverse Kinematics (FK & IK) calculations
- Workspace plots and 3D visualizations
- Python-based simulations and animations
- ROS 2 URDF/Xacro models with RViz & Gazebo visualization

## 📂 Repository structure
.
├─ simulation.py          # Main Python simulation (FK/IK, analysis, animations)
├─ imges.py               # Plots & workspace figures
├─ final_ws/              # ROS 2 workspace (for visualization only)
│  └─ src/
├─ docs/                  # Reports & instructions
│   ├─ [SCARA-6DOF-Final-Report.pdf](docs/SCARA-6DOF-Final-Report.pdf)
│   └─ [Setup-Instructions.pdf](docs/Setup-Instructions.pdf)
└─ .gitignore

## 🐍 Python (main part)
**Installation:**
pip install numpy matplotlib

**Run:**
python simulation.py
python imges.py

## 🤖 ROS 2 Visualization
Workspace: final_ws/

**Build & Launch RViz:**
cd final_ws
colcon build
source install/setup.bash   # Linux
call install\setup.bat      # Windows
ros2 launch robot_arm_description display.launch.py

## 📄 Documentation
- [SCARA-6DOF-Final-Report.pdf](docs/SCARA-6DOF-Final-Report.pdf) – full final project report  
- [Setup-Instructions.pdf](docs/Setup-Instructions.pdf) – setup and usage instructions  

## 📜 License
This project is licensed under the [MIT License](LICENSE).


## 📄 Documentation
Reports and explanations are inside the `docs/` folder.
