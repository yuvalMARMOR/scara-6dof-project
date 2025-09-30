import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SCARAGraphGenerator:
    """
    SCARA robot graph generator - Creates all graphs from the Hebrew research paper
    """
    
    def __init__(self):
        # Robot physical parameters from the research paper
        self.base_height = 800.0  # mm
        self.link_offset = 150.0  # mm
        
        # Visualization settings
        self.workspace_data = []
        
        print("SCARA Graph Generator initialized")
    
    def forward_kinematics(self, theta1, l1, l2, theta2, theta3, theta4):
        """Calculate end effector position using forward kinematics"""
        # Convert angles to radians
        th1 = np.deg2rad(theta1)
        th2 = np.deg2rad(theta2)
        th3 = np.deg2rad(theta3)
        
        # Trigonometric calculations
        cos1, sin1 = np.cos(th1), np.sin(th1)
        cos2, sin2 = np.cos(th2), np.sin(th2)
        cos3, sin3 = np.cos(th3), np.sin(th3)
        
        # Position equations from the research paper
        x = -150*sin1 - 150*cos3*sin1 - l2*sin1 - 150*cos1*cos2*sin3
        y = 150*cos1 + 150*cos1*cos3 + l2*cos1 - 150*cos2*sin1*sin3
        z = l1 + 150*sin2*sin3 + 800
        
        return np.array([x, y, z])
    
    def inverse_kinematics(self, target_position, target_orientation=None, previous_config=None):
        """Calculate joint angles for desired end effector position"""
        x, y, z = target_position
        
        # Calculate vertical extension
        l1 = z - self.base_height
        l1 = max(0, min(500, l1))
        
        # Calculate radial distance and base angle
        radial_distance = np.sqrt(x**2 + y**2)
        
        if radial_distance < 1e-6:
            theta1 = 0 if previous_config is None else previous_config[0]
            l2 = 0
        else:
            theta1_raw = np.rad2deg(np.arctan2(y, x))
            
            if previous_config is not None:
                angle_difference = theta1_raw - previous_config[0]
                while angle_difference > 180:
                    angle_difference -= 360
                while angle_difference < -180:
                    angle_difference += 360
                theta1 = previous_config[0] + 0.7 * angle_difference
            else:
                theta1 = theta1_raw
            
            l2 = radial_distance - self.link_offset
            l2 = max(0, min(500, l2))
        
        # Calculate wrist orientations
        if target_orientation is None:
            if previous_config is not None:
                radial_norm = min(radial_distance / 800.0, 1.0)
                height_norm = l1 / 500.0
                
                theta2_target = 20 * np.sin(radial_norm * 2 * np.pi)
                theta3_target = 15 * np.cos(height_norm * np.pi) if height_norm > 0 else 0
                theta4_target = previous_config[5] + 3.0
                
                smoothing_factor = 0.7
                theta2 = previous_config[3] + smoothing_factor * (theta2_target - previous_config[3])
                theta3 = previous_config[4] + smoothing_factor * (theta3_target - previous_config[4])
                theta4 = theta4_target
            else:
                theta2 = theta3 = theta4 = 0
        else:
            theta2, theta3, theta4 = target_orientation
        
        # Apply joint limits
        theta1 = max(-180, min(180, theta1))
        theta2 = max(-180, min(180, theta2))
        theta3 = max(-135, min(135, theta3))
        theta4 = max(-180, min(180, theta4))
        
        return [theta1, l1, l2, theta2, theta3, theta4]
    
    def compute_robot_frames(self, theta1, l1, l2, theta2, theta3, theta4):
        """Calculate positions of all robot joints and links"""
        th1 = np.deg2rad(theta1)
        frame_positions = []
        
        # World frame origin
        frame_positions.append(np.array([0, 0, 0]))
        # Base platform
        frame_positions.append(np.array([0, 0, 50]))
        # Top of vertical column
        frame_positions.append(np.array([0, 0, 800]))
        # After vertical extension
        frame_positions.append(np.array([0, 0, 800 + l1]))
        # After horizontal extension
        horizontal_pos = np.array([l2 * np.cos(th1), l2 * np.sin(th1), 800 + l1])
        frame_positions.append(horizontal_pos)
        # First wrist joint
        wrist_offset1 = np.array([150 * np.cos(th1), 150 * np.sin(th1), 0])
        frame_positions.append(horizontal_pos + wrist_offset1)
        # End effector position
        end_effector_pos = self.forward_kinematics(theta1, l1, l2, theta2, theta3, theta4)
        frame_positions.append(end_effector_pos)
        
        return np.array(frame_positions)
    
    def plan_trajectory(self, start_pos, end_pos, motion_time=30.0):
        """Generate trajectory between two points using trapezoidal velocity profile"""
        acceleration_time = 10.0
        deceleration_start_time = 20.0
        time_step = 0.04
        
        time_points = np.arange(0, motion_time + time_step, time_step)
        num_points = len(time_points)
        
        positions = np.zeros((3, num_points))
        orientations = np.zeros((3, num_points))
        joint_configurations = []
        velocity_profile = np.zeros(num_points)
        
        previous_joints = None
        total_distance = np.linalg.norm(end_pos - start_pos)
        area_under_curve = motion_time/2 + deceleration_start_time/2 - acceleration_time/2
        max_velocity = total_distance / area_under_curve
        
        for i, t in enumerate(time_points):
            if t <= acceleration_time:
                acceleration = max_velocity / acceleration_time
                distance_factor = 0.5 * acceleration * t**2 / total_distance
                current_velocity = acceleration * t
            elif t <= deceleration_start_time:
                accel_distance = 0.5 * max_velocity * acceleration_time
                const_distance = max_velocity * (t - acceleration_time)
                distance_factor = (accel_distance + const_distance) / total_distance
                current_velocity = max_velocity
            else:
                decel_time = motion_time - deceleration_start_time
                time_into_decel = t - deceleration_start_time
                accel_distance = 0.5 * max_velocity * acceleration_time
                const_distance = max_velocity * (deceleration_start_time - acceleration_time)
                decel_acceleration = max_velocity / decel_time
                decel_distance = max_velocity * time_into_decel - 0.5 * decel_acceleration * time_into_decel**2
                distance_factor = (accel_distance + const_distance + decel_distance) / total_distance
                current_velocity = max_velocity - decel_acceleration * time_into_decel
            
            distance_factor = max(0, min(1, distance_factor))
            current_velocity = max(0, current_velocity)
            velocity_profile[i] = current_velocity
            
            positions[:, i] = start_pos + distance_factor * (end_pos - start_pos)
            
            roll_angle = 10 * np.sin(distance_factor * 2 * np.pi)
            pitch_angle = 15 * np.cos(distance_factor * np.pi)
            yaw_angle = distance_factor * 180
            orientations[:, i] = [roll_angle, pitch_angle, yaw_angle]
            
            joint_config = self.inverse_kinematics(positions[:, i], orientations[:, i], previous_joints)
            joint_configurations.append(joint_config)
            previous_joints = joint_config
        
        return {
            'time': time_points,
            'positions': positions,
            'orientations': orientations,
            'joint_configs': joint_configurations,
            'velocities': velocity_profile,
            'start_pos': start_pos,
            'end_pos': end_pos
        }
    
    def draw_robot_base(self, ax):
        """Draw the robot base structure"""
        angles = np.linspace(0, 2*np.pi, 32)
        outer_radius = 120
        
        x_outer = outer_radius * np.cos(angles)
        y_outer = outer_radius * np.sin(angles)
        
        ax.plot(x_outer, y_outer, 0, 'k-', linewidth=4)
        ax.plot(x_outer, y_outer, 50, 'k-', linewidth=4)
        ax.plot(x_outer, y_outer, self.base_height, 'k-', linewidth=3)
        ax.plot([0, 0], [0, 0], [0, 50], 'black', linewidth=12)
        ax.plot([0, 0], [0, 0], [50, self.base_height], 'darkblue', linewidth=10, alpha=0.9)
    
    def draw_robot_arm(self, ax, frame_positions, joint_angles):
        """Draw the robot arm links and joints"""
        x_coords = frame_positions[:, 0]
        y_coords = frame_positions[:, 1]
        z_coords = frame_positions[:, 2]
        
        link_colors = ['#1B4F72', '#C0392B', '#229954', '#2874A6', '#D68910', '#8E44AD', '#E74C3C']
        link_widths = [10, 8, 7, 6, 5, 4, 6]
        
        for i in range(len(frame_positions) - 1):
            color = link_colors[i % len(link_colors)]
            width = link_widths[i % len(link_widths)]
            
            ax.plot([x_coords[i], x_coords[i+1]], 
                   [y_coords[i], y_coords[i+1]], 
                   [z_coords[i], z_coords[i+1]], 
                   color=color, linewidth=width, solid_capstyle='round', alpha=0.9)
        
        joint_sizes = [120, 100, 90, 80, 70, 60, 50]
        joint_colors = ['navy', 'darkred', 'darkgreen', 'darkblue', 'darkorange', 'purple', 'crimson']
        
        for i, position in enumerate(frame_positions):
            color = joint_colors[i % len(joint_colors)]
            size = joint_sizes[i % len(joint_sizes)]
            
            if i == 2 or i == 3:
                marker = 's'
            elif i == len(frame_positions) - 1:
                marker = '*'
            else:
                marker = 'o'
                
            ax.scatter(position[0], position[1], position[2], 
                     c=color, s=size, alpha=0.9, 
                     marker=marker, edgecolors='black', linewidth=2)
    
    def calculate_workspace(self, resolution=18):
        """Calculate reachable workspace points"""
        print("Calculating robot workspace...")
        
        workspace_points = []
        theta1_values = np.linspace(-180, 180, resolution)
        l1_values = np.linspace(0, 500, resolution//2)
        l2_values = np.linspace(0, 500, resolution//2)
        theta2_values = np.linspace(-90, 90, resolution//3)
        theta3_values = np.linspace(-90, 90, resolution//3)
        
        for theta1 in theta1_values:
            for l1 in l1_values:
                for l2 in l2_values:
                    for theta2 in theta2_values[::2]:
                        for theta3 in theta3_values[::2]:
                            end_position = self.forward_kinematics(theta1, l1, l2, theta2, theta3, 0)
                            workspace_points.append(end_position)
        
        self.workspace_data = np.array(workspace_points)
        print(f"Workspace calculated: {len(workspace_points):,} reachable points")
        return self.workspace_data
    
    # GRAPH GENERATION METHODS
    
    def create_kinematics_verification_plots(self):
        """Create plots showing kinematics verification (Figures 6-8)"""
        print("Creating kinematics verification plots...")
        
        test_cases = [
            ([0, 250, 0, 90, 0, 0], [-150, 150, 1050]),
            ([0, 250, 250, 90, 90, 0], [0, 400, 1200]),
            ([180, 500, 500, 0, 0, 0], [-800, 0, 1300]),
            ([180, 500, 500, 90, 90, 180], [-650, 0, 1450]),
            ([180, 0, 0, 0, 270, 180], [-300, 0, 800])
        ]
        
        # Individual test case plots
        for i, (joints, expected_pos) in enumerate(test_cases, 1):
            fig = plt.figure(figsize=(12, 8))
            calculated_pos = self.forward_kinematics(*joints)
            error = np.linalg.norm(np.array(expected_pos) - calculated_pos)
            
            ax = fig.add_subplot(111, projection='3d')
            
            # Use the actual joint configuration to position the robot correctly
            robot_frames = self.compute_robot_frames(*joints)
            
            self.draw_robot_base(ax)
            self.draw_robot_arm(ax, robot_frames, joints)
            
            # Mark the actual end effector position
            ax.scatter(*calculated_pos, c='green', s=200, marker='o', 
                      label=f'End Effector Position\n[{calculated_pos[0]:.0f}, {calculated_pos[1]:.0f}, {calculated_pos[2]:.0f}] mm', alpha=0.9)
            
            # Set proper limits to show the robot in correct position
            ax.set_xlim([-1000, 1000])
            ax.set_ylim([-1000, 1000])
            ax.set_zlim([0, 1600])
            
            # Remove title as requested
            ax.set_xlabel('X (mm)', fontweight='bold')
            ax.set_ylabel('Y (mm)', fontweight='bold')
            ax.set_zlabel('Z (mm)', fontweight='bold')
            ax.legend(fontsize=10)
            ax.view_init(elev=20, azim=45)
            
            plt.tight_layout()
            plt.savefig(f'figure_6_kinematics_test_{i}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        errors = [np.linalg.norm(np.array(expected) - self.forward_kinematics(*joints)) 
                 for joints, expected in test_cases]
        test_numbers = range(1, 6)
        
        bars = ax.bar(test_numbers, errors, color=['green' if e < 1.0 else 'orange' for e in errors])
        ax.axhline(y=1.0, color='red', linestyle='--', label='Acceptable Error (1mm)')
        ax.set_xlabel('Test Case', fontweight='bold')
        ax.set_ylabel('Position Error (mm)', fontweight='bold')
        ax.set_title('Kinematics Verification Summary\nForward Kinematics Accuracy', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{error:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('figure_8_kinematics_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved figures 6.1-6.5 and figure 8")

    def create_workspace_plots(self):
        """Generate workspace analysis visualizations (Figures 9-12)"""
        print("Creating workspace analysis plots...")
        
        if len(self.workspace_data) == 0:
            self.calculate_workspace(resolution=18)
        
        # Use consistent color scheme for all plots
        colormap = 'viridis'
        
        # Figure 9 - 3D isometric view
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        height_colors = ax.scatter(self.workspace_data[:, 0], self.workspace_data[:, 1], 
                   self.workspace_data[:, 2], c=self.workspace_data[:, 2], 
                   s=2, alpha=0.6, cmap=colormap)
        ax.set_xlabel('X (mm)', fontweight='bold')
        ax.set_ylabel('Y (mm)', fontweight='bold')
        ax.set_zlabel('Z (mm)', fontweight='bold')
        ax.set_title('3D Workspace View - Isometric\n(Height Colored)', fontweight='bold')
        ax.view_init(elev=20, azim=45)
        plt.colorbar(height_colors, ax=ax, shrink=0.5, aspect=20, label='Height (mm)')
        
        plt.tight_layout()
        plt.savefig('figure_9_workspace_3d.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 10 - X-Y plane view
        fig, ax = plt.subplots(figsize=(10, 8))
        xy_colors = ax.scatter(self.workspace_data[:, 0], self.workspace_data[:, 1], 
                   c=self.workspace_data[:, 2], s=3, alpha=0.6, cmap=colormap)
        ax.set_xlabel('X (mm)', fontweight='bold')
        ax.set_ylabel('Y (mm)', fontweight='bold')
        ax.set_title('X-Y Plane View\n(Height Colored)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.colorbar(xy_colors, ax=ax, shrink=0.8, label='Height (mm)')
        
        plt.tight_layout()
        plt.savefig('figure_10_workspace_xy.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 11 - X-Z plane view
        fig, ax = plt.subplots(figsize=(10, 8))
        xz_colors = ax.scatter(self.workspace_data[:, 0], self.workspace_data[:, 2], 
                   c=self.workspace_data[:, 2], s=3, alpha=0.6, cmap=colormap)
        ax.set_xlabel('X (mm)', fontweight='bold')
        ax.set_ylabel('Z (mm)', fontweight='bold')
        ax.set_title('X-Z Plane View\n(Height Colored)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(xz_colors, ax=ax, shrink=0.8, label='Height (mm)')
        
        plt.tight_layout()
        plt.savefig('figure_11_workspace_xz.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 12 - Y-Z plane view
        fig, ax = plt.subplots(figsize=(10, 8))
        yz_colors = ax.scatter(self.workspace_data[:, 1], self.workspace_data[:, 2], 
                   c=self.workspace_data[:, 2], s=3, alpha=0.6, cmap=colormap)
        ax.set_xlabel('Y (mm)', fontweight='bold')
        ax.set_ylabel('Z (mm)', fontweight='bold')
        ax.set_title('Y-Z Plane View\n(Height Colored)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(yz_colors, ax=ax, shrink=0.8, label='Height (mm)')
        
        plt.tight_layout()
        plt.savefig('figure_12_workspace_yz.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved figures 9-12")

    def create_trajectory_analysis_plots(self, trajectory_data):
        """Generate trajectory analysis plots (Figures 16-17, 22-23)"""
        print("Creating trajectory analysis plots...")
        
        time_data = trajectory_data['time']
        position_data = trajectory_data['positions']
        orientation_data = trajectory_data['orientations']
        velocity_data = trajectory_data['velocities']
        
        # Figure 16 - 3D trajectory path
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        vel_normalized = velocity_data / np.max(velocity_data) if np.max(velocity_data) > 0 else velocity_data
        trajectory_colors = plt.cm.viridis(vel_normalized)
        
        for i in range(len(position_data[0]) - 1):
            ax.plot([position_data[0, i], position_data[0, i+1]], 
                    [position_data[1, i], position_data[1, i+1]], 
                    [position_data[2, i], position_data[2, i+1]], 
                    color=trajectory_colors[i], linewidth=3, alpha=0.8)
        
        ax.scatter(position_data[0, 0], position_data[1, 0], position_data[2, 0], 
                  c='green', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
        ax.scatter(position_data[0, -1], position_data[1, -1], position_data[2, -1], 
                  c='red', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('X (mm)', fontweight='bold')
        ax.set_ylabel('Y (mm)', fontweight='bold')
        ax.set_zlabel('Z (mm)', fontweight='bold')
        ax.set_title('3D Trajectory Path\n(Velocity Colored)', fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('figure_16_3d_trajectory_path.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 17 - Velocity profile
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(time_data, velocity_data, 'purple', linewidth=4, label='End-effector velocity')
        ax.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Accel end')
        ax.axvline(x=20, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Decel start')
        
        max_vel = np.max(velocity_data)
        ax.text(5, max_vel*0.8, 'ACCEL', ha='center', fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax.text(15, max_vel*0.8, 'CONSTANT', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(25, max_vel*0.8, 'DECEL', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Velocity (mm/s)', fontweight='bold')
        ax.set_title('Trapezoidal Velocity Profile', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_17_velocity_profile.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 22 - Position vs time
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(time_data, position_data[0], 'r-', linewidth=3, label='X position')
        ax.plot(time_data, position_data[1], 'g-', linewidth=3, label='Y position')
        ax.plot(time_data, position_data[2], 'b-', linewidth=3, label='Z position')
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Position (mm)', fontweight='bold')
        ax.set_title('End-Effector Position vs Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_22_position_vs_time.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 23 - Orientation vs time
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(time_data, orientation_data[0], 'r-', linewidth=3, label='Roll')
        ax.plot(time_data, orientation_data[1], 'g-', linewidth=3, label='Pitch')
        ax.plot(time_data, orientation_data[2], 'b-', linewidth=3, label='Yaw')
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Orientation (degrees)', fontweight='bold')
        ax.set_title('End-Effector Orientation\n(6-DOF Motion)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_23_orientation_vs_time.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved figures 16-17, 22-23")

    def create_joint_analysis_plots(self, trajectory_data):
        """Generate joint analysis plots (Figures 25-26)"""
        print("Creating joint analysis plots...")
        
        time_data = trajectory_data['time']
        joint_data = np.array(trajectory_data['joint_configs']).T
        
        # Figure 25 - Rotational joints
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(time_data, joint_data[0], 'r-', linewidth=3, label='Base (θ₁)')
        ax.plot(time_data, joint_data[3], 'g-', linewidth=3, label='Pitch (θ₂)')
        ax.plot(time_data, joint_data[4], 'b-', linewidth=3, label='Roll (θ₃)')
        ax.plot(time_data, joint_data[5], 'm-', linewidth=3, label='End eff (θ₄)')
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Angle (degrees)', fontweight='bold')
        ax.set_title('Rotational Joint Angles vs Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_25_rotational_joints.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 26 - Prismatic joints
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(time_data, joint_data[1], 'c-', linewidth=4, label='Vertical (l₁)')
        ax.plot(time_data, joint_data[2], 'm-', linewidth=4, label='Horizontal (l₂)')
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Extension (mm)', fontweight='bold')
        ax.set_title('Prismatic Joint Extensions vs Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_26_prismatic_joints.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved figures 25-26")

    def create_velocity_analysis_plots(self, trajectory_data):
        """Generate velocity analysis plots (Figure 27)"""
        print("Creating velocity analysis plots...")
        
        time_data = trajectory_data['time']
        position_data = trajectory_data['positions']
        dt = time_data[1] - time_data[0]
        
        velocity_x = []
        velocity_y = []
        velocity_z = []
        
        for i in range(len(position_data[0]) - 1):
            vel_x = (position_data[0, i+1] - position_data[0, i]) / dt
            vel_y = (position_data[1, i+1] - position_data[1, i]) / dt
            vel_z = (position_data[2, i+1] - position_data[2, i]) / dt
            
            velocity_x.append(vel_x)
            velocity_y.append(vel_y)
            velocity_z.append(vel_z)
        
        velocity_x.append(0)
        velocity_y.append(0)
        velocity_z.append(0)
        
        # Figure 27 - Velocity components
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(time_data, velocity_x, 'r-', linewidth=3, label='X Velocity')
        ax.plot(time_data, velocity_y, 'g-', linewidth=3, label='Y Velocity')
        ax.plot(time_data, velocity_z, 'b-', linewidth=3, label='Z Velocity')
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Velocity (mm/s)', fontweight='bold')
        ax.set_title('End-Effector Velocity Components', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_27_velocity_components.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved figure 27")

    def create_static_force_analysis(self, trajectory_data):
        """Create static force and torque analysis plots (Figures 30-31)"""
        print("Creating static force analysis plots...")
        
        joint_configs = trajectory_data['joint_configs']
        time_data = trajectory_data['time']
        
        mass = 1.0  # kg
        g = 9.81    # m/s^2
        L_EF = 0.15 # m
        
        forces_torques = []
        
        for joints in joint_configs:
            theta1, l1, l2, theta2, theta3, theta4 = joints
            
            theta2_rad = np.deg2rad(theta2)
            theta3_rad = np.deg2rad(theta3)
            
            F_l1 = mass * g
            T_theta2 = mass * g * L_EF * np.cos(theta2_rad) * np.sin(theta3_rad)
            T_theta3 = mass * g * L_EF * np.sin(theta2_rad) * np.cos(theta3_rad)
            
            forces_torques.append([0, F_l1, 0, T_theta2, T_theta3, 0])
        
        forces_torques = np.array(forces_torques).T
        
        # Figure 30 - Torques
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(time_data, forces_torques[3], 'r-', linewidth=3, label='T_θ2 (Wrist Pitch)')
        ax.plot(time_data, forces_torques[4], 'g-', linewidth=3, label='T_θ3 (Wrist Roll)')
        ax.plot(time_data, forces_torques[0], 'b-', linewidth=3, label='T_θ1 (Base)')
        ax.plot(time_data, forces_torques[5], 'm-', linewidth=3, label='T_θ4 (End Effector)')
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Torque (N⋅m)', fontweight='bold')
        ax.set_title('Static Torques in Robot Joints vs Time\n(1 kg payload)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_30_static_torques.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 31 - Forces
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(time_data, forces_torques[1], 'c-', linewidth=4, label='F_l1 (Vertical Force)')
        ax.plot(time_data, forces_torques[2], 'orange', linewidth=4, label='F_l2 (Horizontal Force)')
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Force (N)', fontweight='bold')
        ax.set_title('Static Forces in Prismatic Joints vs Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_31_static_forces.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved figures 30-31")

    def create_velocity_comparison_analysis(self, trajectory_data):
        """Create velocity comparison between analytical and numerical methods (Figure 29)"""
        print("Creating velocity comparison analysis...")
        
        time_data = trajectory_data['time']
        position_data = trajectory_data['positions']
        analytical_velocity = trajectory_data['velocities']
        
        dt = time_data[1] - time_data[0]
        numerical_velocity = np.zeros_like(analytical_velocity)
        
        for i in range(len(position_data[0]) - 1):
            pos_diff = np.linalg.norm(position_data[:, i+1] - position_data[:, i])
            numerical_velocity[i] = pos_diff / dt
        
        numerical_velocity[-1] = 0
        
        # Figure 29
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        ax1.plot(time_data, analytical_velocity, 'b-', linewidth=3, label='Analytical')
        ax1.plot(time_data, numerical_velocity, 'r--', linewidth=2, label='Forward Difference')
        ax1.set_xlabel('Time (s)', fontweight='bold')
        ax1.set_ylabel('Velocity (mm/s)', fontweight='bold')
        ax1.set_title('Velocity Comparison: Analytical vs Numerical Methods', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        error = np.abs(analytical_velocity - numerical_velocity)
        ax2.plot(time_data, error, 'purple', linewidth=2, label='Absolute Error')
        ax2.set_xlabel('Time (s)', fontweight='bold')
        ax2.set_ylabel('Error (mm/s)', fontweight='bold')
        ax2.set_title('Numerical Differentiation Error', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_29_velocity_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved figure 29")

    def create_alternative_motion_planning(self):
        """Create alternative trapezoidal motion planning plots (Figures 32-37)"""
        print("Creating alternative motion planning analysis...")
        
        start_joints = [0, 250, 0, 90, 0, 0]
        end_joints = [90, 300, 400, 45, 45, 180]
        
        total_time = 5.0
        accel_time = 1.0
        decel_time = 1.0
        dt = 0.02
        
        time_points = np.arange(0, total_time + dt, dt)
        n_points = len(time_points)
        
        joint_trajectories = np.zeros((6, n_points))
        joint_velocities = np.zeros((6, n_points))
        
        for j in range(6):
            delta = end_joints[j] - start_joints[j]
            steady_time = total_time - accel_time - decel_time
            max_vel = delta / (0.5 * accel_time + steady_time + 0.5 * decel_time)
            accel = max_vel / accel_time
            
            for i, t in enumerate(time_points):
                if t <= accel_time:
                    pos = start_joints[j] + 0.5 * accel * t**2
                    vel = accel * t
                elif t <= (total_time - decel_time):
                    accel_dist = 0.5 * accel * accel_time**2
                    pos = start_joints[j] + accel_dist + max_vel * (t - accel_time)
                    vel = max_vel
                else:
                    accel_dist = 0.5 * accel * accel_time**2
                    const_dist = max_vel * steady_time
                    decel_t = t - (total_time - decel_time)
                    pos = start_joints[j] + accel_dist + const_dist + max_vel * decel_t - 0.5 * accel * decel_t**2
                    vel = max_vel - accel * decel_t
                
                joint_trajectories[j, i] = pos
                joint_velocities[j, i] = vel
        
        # Figure 32 - Rotational positions
        fig, ax = plt.subplots(figsize=(12, 8))
        joint_names_rot = ['θ₁ (Base)', 'θ₂ (Wrist Pitch)', 'θ₃ (Wrist Roll)', 'θ₄ (End Effector)']
        joint_indices_rot = [0, 3, 4, 5]
        colors_rot = ['red', 'green', 'blue', 'magenta']
        
        for i, (idx, name, color) in enumerate(zip(joint_indices_rot, joint_names_rot, colors_rot)):
            ax.plot(time_points, joint_trajectories[idx], color=color, linewidth=3, label=name)
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Angle (degrees)', fontweight='bold')
        ax.set_title('Rotational Joint Positions vs Time\n(Trapezoidal Profile)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_32_rotational_positions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 33 - Prismatic positions
        fig, ax = plt.subplots(figsize=(12, 8))
        joint_names_pris = ['l₁ (Vertical)', 'l₂ (Horizontal)']
        joint_indices_pris = [1, 2]
        colors_pris = ['cyan', 'orange']
        
        for i, (idx, name, color) in enumerate(zip(joint_indices_pris, joint_names_pris, colors_pris)):
            ax.plot(time_points, joint_trajectories[idx], color=color, linewidth=4, label=name)
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Extension (mm)', fontweight='bold')
        ax.set_title('Prismatic Joint Positions vs Time\n(Trapezoidal Profile)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_33_prismatic_positions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 34 - Rotational velocities
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, (idx, name, color) in enumerate(zip(joint_indices_rot, joint_names_rot, colors_rot)):
            ax.plot(time_points, joint_velocities[idx], color=color, linewidth=3, label=name)
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Angular Velocity (deg/s)', fontweight='bold')
        ax.set_title('Rotational Joint Velocities vs Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_34_rotational_velocities.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 35 - Prismatic velocities
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, (idx, name, color) in enumerate(zip(joint_indices_pris, joint_names_pris, colors_pris)):
            ax.plot(time_points, joint_velocities[idx], color=color, linewidth=4, label=name)
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Linear Velocity (mm/s)', fontweight='bold')
        ax.set_title('Prismatic Joint Velocities vs Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_35_prismatic_velocities.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 36 - 3D trajectory
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        end_effector_positions = []
        for i in range(n_points):
            pos = self.forward_kinematics(*joint_trajectories[:, i])
            end_effector_positions.append(pos)
        
        end_effector_positions = np.array(end_effector_positions).T
        
        ax.plot(end_effector_positions[0], end_effector_positions[1], end_effector_positions[2], 
                'purple', linewidth=3, label='Alternative Trajectory')
        ax.scatter(end_effector_positions[0, 0], end_effector_positions[1, 0], end_effector_positions[2, 0], 
                   c='green', s=100, marker='o', label='Start')
        ax.scatter(end_effector_positions[0, -1], end_effector_positions[1, -1], end_effector_positions[2, -1], 
                   c='red', s=100, marker='s', label='End')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Trajectory\n(Alternative Planning)', fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('figure_36_3d_trajectory.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 37 - End effector velocity
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ee_velocities = []
        for i in range(n_points - 1):
            vel_vec = (end_effector_positions[:, i+1] - end_effector_positions[:, i]) / dt
            ee_velocities.append(np.linalg.norm(vel_vec))
        ee_velocities.append(0)
        
        ax.plot(time_points, ee_velocities, 'navy', linewidth=4, label='End Effector Velocity')
        ax.axvline(x=accel_time, color='red', linestyle='--', alpha=0.7, label='Accel End')
        ax.axvline(x=total_time-decel_time, color='orange', linestyle='--', alpha=0.7, label='Decel Start')
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Velocity (mm/s)', fontweight='bold')
        ax.set_title('End Effector Velocity Profile\n(Alternative Planning)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_37_end_effector_velocity.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved figures 32-37")

    def generate_all_graphs(self):
        """Generate all graphs from the Hebrew research paper"""
        print("="*60)
        print("SCARA ROBOT GRAPH GENERATOR")
        print("Based on Ben-Gurion University Research")
        print("="*60)
        
        # Create kinematics verification plots (Figures 6-8)
        self.create_kinematics_verification_plots()
        
        # Generate workspace analysis (Figures 9-12)
        self.create_workspace_plots()
        
        # Plan trajectory for other plots
        start_position = np.array([-150.0, 150.0, 1050.0])
        end_position = np.array([0.0, 400.0, 1200.0])
        
        print("Planning trajectory for analysis plots...")
        trajectory = self.plan_trajectory(start_position, end_position)
        
        # Create trajectory analysis plots (Figures 16-17, 22-23)
        self.create_trajectory_analysis_plots(trajectory)
        
        # Create joint analysis plots (Figures 25-26)
        self.create_joint_analysis_plots(trajectory)
        
        # Create velocity analysis plots (Figure 27)
        self.create_velocity_analysis_plots(trajectory)
        
        # Create static force analysis (Figures 30-31)
        self.create_static_force_analysis(trajectory)
        
        # Create velocity comparison analysis (Figure 29)
        self.create_velocity_comparison_analysis(trajectory)
        
        # Create alternative motion planning analysis (Figures 32-37)
        self.create_alternative_motion_planning()
        
        print("\n" + "="*60)
        print("ALL GRAPHS GENERATED SUCCESSFULLY!")
        print("="*60)
        print("Generated files (26 total):")
        print("  Kinematics Verification (6 files):")
        print("    • figure_6_kinematics_test_1.png through figure_6_kinematics_test_5.png")
        print("    • figure_8_kinematics_summary.png")
        print("  Workspace Analysis (4 files):")
        print("    • figure_9_workspace_3d.png through figure_12_workspace_yz.png")
        print("  Trajectory Analysis (4 files):")
        print("    • figure_16_3d_trajectory_path.png, figure_17_velocity_profile.png")
        print("    • figure_22_position_vs_time.png, figure_23_orientation_vs_time.png")
        print("  Joint Analysis (2 files):")
        print("    • figure_25_rotational_joints.png, figure_26_prismatic_joints.png")
        print("  Velocity Analysis (2 files):")
        print("    • figure_27_velocity_components.png, figure_29_velocity_comparison.png")
        print("  Static Force Analysis (2 files):")
        print("    • figure_30_static_torques.png, figure_31_static_forces.png")
        print("  Alternative Motion Planning (6 files):")
        print("    • figure_32_rotational_positions.png through figure_37_end_effector_velocity.png")
        print("="*60)

def main():
    """Main function to generate all graphs"""
    try:
        print("Starting SCARA Robot Graph Generation...")
        print("Creating all graphs from Ben-Gurion University research paper")
        print()
        
        # Create the graph generator
        generator = SCARAGraphGenerator()
        
        # Generate all graphs
        generator.generate_all_graphs()
        
    except KeyboardInterrupt:
        print("\nGraph generation stopped by user")
    except Exception as e:
        print(f"Error during graph generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()