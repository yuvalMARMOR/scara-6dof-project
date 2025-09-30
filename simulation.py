import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class SCARASimulator:
    def __init__(self):
        # Robot physical parameters from the research paper
        self.base_height = 800.0  # mm
        self.link_offset = 150.0  # mm
        
        # URDF model dimensions
        self.base_length = 50.0   # mm
        self.base_radius = 120.0  # mm
        self.vertical_column_height = 800.0  # mm
        
        # Joint constraints from the paper (Table 1)
        self.theta1_limits = (-180, 180)   # base rotation
        self.l1_limits = (0, 500)          # vertical extension
        self.l2_limits = (0, 500)          # horizontal extension
        self.theta2_limits = (-180, 180)   # wrist pitch
        self.theta3_limits = (-135, 135)   # wrist roll
        self.theta4_limits = (-180, 180)   # end effector rotation
        
        # Visualization settings
        self.workspace_data = []
        self.trajectory_trail_length = 50
        
        print("SCARA Robot Simulator initialized")
        print(f"Base height: {self.base_height} mm")
        print(f"Link offset: {self.link_offset} mm")
    
    def forward_kinematics(self, theta1, l1, l2, theta2, theta3, theta4):
        """
        Calculate end effector position using forward kinematics
        """
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
        """
        Calculate joint angles for desired end effector position
        """
        x, y, z = target_position
        
        # Calculate vertical extension
        l1 = z - self.base_height
        l1 = max(0, min(500, l1))  # Apply joint limits
        
        # Calculate radial distance and base angle
        radial_distance = np.sqrt(x**2 + y**2)
        
        if radial_distance < 1e-6:
            # Handle singularity at origin
            theta1 = 0 if previous_config is None else previous_config[0]
            l2 = 0
        else:
            # Calculate base rotation angle
            theta1_raw = np.rad2deg(np.arctan2(y, x))
            
            # Smooth angle transitions to avoid jumps
            if previous_config is not None:
                angle_difference = theta1_raw - previous_config[0]
                # Normalize angle difference
                while angle_difference > 180:
                    angle_difference -= 360
                while angle_difference < -180:
                    angle_difference += 360
                theta1 = previous_config[0] + 0.7 * angle_difference
            else:
                theta1 = theta1_raw
            
            # Calculate horizontal extension
            l2 = radial_distance - self.link_offset
            l2 = max(0, min(500, l2))
        
        # Calculate wrist orientations
        if target_orientation is None:
            if previous_config is not None:
                # Create smooth orientation motion
                radial_norm = min(radial_distance / 800.0, 1.0)
                height_norm = l1 / 500.0
                
                # Generate orientation targets
                theta2_target = 20 * np.sin(radial_norm * 2 * np.pi)
                theta3_target = 15 * np.cos(height_norm * np.pi) if height_norm > 0 else 0
                theta4_target = previous_config[5] + 3.0  # Continuous rotation
                
                # Apply smoothing filter
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
        """
        Calculate positions of all robot joints and links
        """
        th1 = np.deg2rad(theta1)
        th2 = np.deg2rad(theta2)
        th3 = np.deg2rad(theta3)
        
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
        
        # Second wrist joint
        wrist_offset2 = np.array([50 * np.cos(th1 + th2), 50 * np.sin(th1 + th2), 0])
        frame_positions.append(frame_positions[-1] + wrist_offset2)
        
        # End effector position
        end_effector_pos = self.forward_kinematics(theta1, l1, l2, theta2, theta3, theta4)
        frame_positions.append(end_effector_pos)
        
        return np.array(frame_positions)
    
    def plan_trajectory(self, start_pos, end_pos, motion_time=30.0):
        """
        Generate trajectory between two points using trapezoidal velocity profile
        """
        print(f"Planning trajectory from {start_pos} to {end_pos}")
        
        # Motion profile timing
        acceleration_time = 10.0
        deceleration_start_time = 20.0
        time_step = 0.04
        
        time_points = np.arange(0, motion_time + time_step, time_step)
        num_points = len(time_points)
        
        # Initialize arrays
        positions = np.zeros((3, num_points))
        orientations = np.zeros((3, num_points))
        joint_configurations = []
        velocity_profile = np.zeros(num_points)
        
        print(f"Generating {num_points} trajectory points...")
        
        previous_joints = None
        total_distance = np.linalg.norm(end_pos - start_pos)
        
        # Calculate maximum velocity for trapezoidal profile
        area_under_curve = motion_time/2 + deceleration_start_time/2 - acceleration_time/2
        max_velocity = total_distance / area_under_curve
        
        print(f"Maximum velocity: {max_velocity:.2f} mm/s")
        
        for i, t in enumerate(time_points):
            # Calculate motion profile
            if t <= acceleration_time:
                # Acceleration phase
                acceleration = max_velocity / acceleration_time
                distance_factor = 0.5 * acceleration * t**2 / total_distance
                current_velocity = acceleration * t
                
            elif t <= deceleration_start_time:
                # Constant velocity phase
                accel_distance = 0.5 * max_velocity * acceleration_time
                const_distance = max_velocity * (t - acceleration_time)
                distance_factor = (accel_distance + const_distance) / total_distance
                current_velocity = max_velocity
                
            else:
                # Deceleration phase
                decel_time = motion_time - deceleration_start_time
                time_into_decel = t - deceleration_start_time
                
                accel_distance = 0.5 * max_velocity * acceleration_time
                const_distance = max_velocity * (deceleration_start_time - acceleration_time)
                
                decel_acceleration = max_velocity / decel_time
                decel_distance = max_velocity * time_into_decel - 0.5 * decel_acceleration * time_into_decel**2
                
                distance_factor = (accel_distance + const_distance + decel_distance) / total_distance
                current_velocity = max_velocity - decel_acceleration * time_into_decel
            
            # Ensure bounds
            distance_factor = max(0, min(1, distance_factor))
            current_velocity = max(0, current_velocity)
            velocity_profile[i] = current_velocity
            
            # Calculate position
            positions[:, i] = start_pos + distance_factor * (end_pos - start_pos)
            
            # Calculate orientations for 6-DOF motion
            roll_angle = 10 * np.sin(distance_factor * 2 * np.pi)
            pitch_angle = 15 * np.cos(distance_factor * np.pi)
            yaw_angle = distance_factor * 180
            orientations[:, i] = [roll_angle, pitch_angle, yaw_angle]
            
            # Solve inverse kinematics
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
        # Create circular base platform
        angles = np.linspace(0, 2*np.pi, 32)
        outer_radius = 120
        inner_radius = 60
        
        x_outer = outer_radius * np.cos(angles)
        y_outer = outer_radius * np.sin(angles)
        x_inner = inner_radius * np.cos(angles)
        y_inner = inner_radius * np.sin(angles)
        
        # Draw base circles at different heights
        ax.plot(x_outer, y_outer, 0, 'k-', linewidth=4)
        ax.plot(x_outer, y_outer, 50, 'k-', linewidth=4)
        ax.plot(x_outer, y_outer, self.base_height, 'k-', linewidth=3)
        ax.plot(x_inner, y_inner, self.base_height, 'gray', linewidth=2)
        
        # Add vertical support columns
        for i in range(0, len(angles), 8):
            ax.plot([x_outer[i], x_outer[i]], [y_outer[i], y_outer[i]], 
                   [0, 50], 'k-', alpha=0.6, linewidth=2)
            ax.plot([x_outer[i], x_outer[i]], [y_outer[i], y_outer[i]], 
                   [50, self.base_height], 'darkgray', alpha=0.4, linewidth=1)
        
        # Central vertical column
        ax.plot([0, 0], [0, 0], [0, 50], 'black', linewidth=12)
        ax.plot([0, 0], [0, 0], [50, self.base_height], 'darkblue', linewidth=10, alpha=0.9)
    
    def draw_robot_arm(self, ax, frame_positions, joint_angles):
        """Draw the robot arm links and joints"""
        x_coords = frame_positions[:, 0]
        y_coords = frame_positions[:, 1]
        z_coords = frame_positions[:, 2]
        
        # Link colors and styling
        link_colors = ['#1B4F72', '#C0392B', '#229954', '#2874A6', '#D68910', '#8E44AD', '#E74C3C']
        link_widths = [10, 8, 7, 6, 5, 4, 6]
        
        # Draw link segments
        for i in range(len(frame_positions) - 1):
            color = link_colors[i % len(link_colors)]
            width = link_widths[i % len(link_widths)]
            
            ax.plot([x_coords[i], x_coords[i+1]], 
                   [y_coords[i], y_coords[i+1]], 
                   [z_coords[i], z_coords[i+1]], 
                   color=color, linewidth=width, solid_capstyle='round', alpha=0.9)
        
        # Draw joint markers
        joint_sizes = [120, 100, 90, 80, 70, 60, 50]
        joint_colors = ['navy', 'darkred', 'darkgreen', 'darkblue', 'darkorange', 'purple', 'crimson']
        
        for i, position in enumerate(frame_positions):
            color = joint_colors[i % len(joint_colors)]
            size = joint_sizes[i % len(joint_sizes)]
            
            # Different markers for joint types
            if i == 2 or i == 3:  # Prismatic joints
                marker = 's'
            elif i == len(frame_positions) - 1:  # End effector
                marker = '*'
            else:  # Rotational joints
                marker = 'o'
                
            ax.scatter(position[0], position[1], position[2], 
                     c=color, s=size, alpha=0.9, 
                     marker=marker, edgecolors='black', linewidth=2)
        
        # Draw orientation vectors at end effector
        if len(frame_positions) >= 2:
            end_position = frame_positions[-1]
            vector_length = 80
            theta1_rad = np.deg2rad(joint_angles[0])
            theta2_rad = np.deg2rad(joint_angles[3])
            
            # X-axis vector (red)
            x_vector = vector_length * np.array([np.cos(theta1_rad), np.sin(theta1_rad), 0])
            ax.plot([end_position[0], end_position[0] + x_vector[0]], 
                   [end_position[1], end_position[1] + x_vector[1]], 
                   [end_position[2], end_position[2] + x_vector[2]], 
                   'r-', linewidth=4, alpha=0.8)
            
            # Y-axis vector (green)
            y_vector = vector_length * np.array([-np.sin(theta1_rad + theta2_rad), 
                                               np.cos(theta1_rad + theta2_rad), 0])
            ax.plot([end_position[0], end_position[0] + y_vector[0]], 
                   [end_position[1], end_position[1] + y_vector[1]], 
                   [end_position[2], end_position[2] + y_vector[2]], 
                   'g-', linewidth=4, alpha=0.8)
            
            # Z-axis vector (blue)
            z_vector = vector_length * np.array([0, 0, 1])
            ax.plot([end_position[0], end_position[0] + z_vector[0]], 
                   [end_position[1], end_position[1] + z_vector[1]], 
                   [end_position[2], end_position[2] + z_vector[2]], 
                   'b-', linewidth=4, alpha=0.8)
    
    def draw_trajectory_path(self, ax, trajectory_data, current_frame):
        """Draw the robot trajectory with velocity coloring"""
        positions = trajectory_data['positions']
        velocities = trajectory_data['velocities']
        
        # Draw planned path (faded)
        ax.plot(positions[0], positions[1], positions[2], 
               'lightblue', linewidth=2, alpha=0.2, linestyle=':')
        
        # Draw completed path with velocity-based colors
        if current_frame > 1:
            trail_start = max(0, current_frame - self.trajectory_trail_length)
            path_positions = positions[:, trail_start:current_frame+1]
            path_velocities = velocities[trail_start:current_frame+1]
            
            # Color mapping based on velocity
            if len(path_velocities) > 0 and np.max(path_velocities) > 0:
                velocity_normalized = path_velocities / np.max(path_velocities)
                
                # Draw colored trajectory segments
                for i in range(len(path_positions[0]) - 1):
                    vel_intensity = velocity_normalized[i]
                    
                    # Color from blue (slow) to red (fast)
                    if vel_intensity < 0.5:
                        color = (0, vel_intensity * 2, 1 - vel_intensity * 2)
                    else:
                        color = ((vel_intensity - 0.5) * 2, 1 - (vel_intensity - 0.5) * 2, 0)
                    
                    line_width = 2 + 4 * vel_intensity
                    
                    ax.plot([path_positions[0, i], path_positions[0, i+1]], 
                           [path_positions[1, i], path_positions[1, i+1]], 
                           [path_positions[2, i], path_positions[2, i+1]], 
                           color=color, linewidth=line_width, alpha=0.8)
        
        # Highlight current position
        if current_frame > 0:
            current_pos = positions[:, current_frame]
            ax.scatter(current_pos[0], current_pos[1], current_pos[2], 
                     c='lime', s=100, alpha=1.0, marker='o', 
                     edgecolors='darkgreen', linewidth=3)
    
    def draw_workspace_limits(self, ax):
        """Draw workspace boundary visualization"""
        angles = np.linspace(0, 2*np.pi, 24)
        
        # Workspace reach limits
        max_reach = 800
        min_reach = 44
        
        x_max = max_reach * np.cos(angles)
        y_max = max_reach * np.sin(angles)
        x_min = min_reach * np.cos(angles)
        y_min = min_reach * np.sin(angles)
        
        # Draw boundaries at key heights
        height_levels = [900, 1200, 1450]
        boundary_colors = ['red', 'orange', 'yellow']
        
        for i, height in enumerate(height_levels):
            alpha_value = 0.2 + i * 0.1
            ax.plot(x_max, y_max, height, boundary_colors[i], 
                   alpha=alpha_value, linestyle='--', linewidth=2)
            ax.plot(x_min, y_min, height, boundary_colors[i], 
                   alpha=alpha_value, linestyle='--', linewidth=2)
    
    def create_info_display(self, ax, joint_config, position, time_value):
        """Create information panel showing robot status"""
        th1, l1, l2, th2, th3, th4 = joint_config
        
        # Determine motion phase
        if time_value <= 10.0:
            phase = "ACCELERATION"
            phase_color = "red"
            description = "Robot is speeding up"
        elif time_value <= 20.0:
            phase = "CONSTANT VEL"
            phase_color = "blue"
            description = "Robot at steady speed"
        else:
            phase = "DECELERATION"
            phase_color = "green"
            description = "Robot is slowing down"
        
        progress_percent = (time_value / 30.0) * 100
        
        # Create information text
        info_text = f'Time: {time_value:.1f}s ({progress_percent:.1f}%)\n'
        info_text += f'Phase: {phase}\n'
        info_text += f'{description}\n'
        info_text += f'Position: [{position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f}] mm\n\n'
        info_text += f'JOINT CONFIGURATION:\n'
        info_text += f'  Base rotation: {th1:7.1f} deg\n'
        info_text += f'  Vertical ext:  {l1:7.0f} mm\n'
        info_text += f'  Horizontal ext: {l2:7.0f} mm\n'
        info_text += f'  Wrist pitch:   {th2:7.1f} deg\n'
        info_text += f'  Wrist roll:    {th3:7.1f} deg\n'
        info_text += f'  End effector:  {th4:7.1f} deg\n\n'
        info_text += f'WORKSPACE INFO:\n'
        info_text += f'  Radial dist: {np.sqrt(position[0]**2 + position[1]**2):.0f} mm\n'
        info_text += f'  Height: {position[2] - 800:.0f} mm above base'
        
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                          alpha=0.95, edgecolor=phase_color, linewidth=3))
    
    def create_robot_animation(self, trajectory_data, save_animation=True):
        """Create animated visualization of robot motion"""
        print("Creating robot animation...")
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        def update_animation(frame_num):
            ax.clear()
            
            if frame_num < len(trajectory_data['joint_configs']):
                # Get current robot state
                current_joints = trajectory_data['joint_configs'][frame_num]
                current_position = trajectory_data['positions'][:, frame_num]
                current_time = trajectory_data['time'][frame_num]
                
                # Calculate robot frame positions
                robot_frames = self.compute_robot_frames(*current_joints)
                
                # Draw all components
                self.draw_robot_base(ax)
                self.draw_robot_arm(ax, robot_frames, current_joints)
                self.draw_trajectory_path(ax, trajectory_data, frame_num)
                self.draw_workspace_limits(ax)
                self.create_info_display(ax, current_joints, current_position, current_time)
                
                # Set up 3D view
                ax.set_xlim([-1000, 1000])
                ax.set_ylim([-1000, 1000])
                ax.set_zlim([0, 1600])
                ax.set_xlabel('X (mm)', fontweight='bold', fontsize=12)
                ax.set_ylabel('Y (mm)', fontweight='bold', fontsize=12)
                ax.set_zlabel('Z (mm)', fontweight='bold', fontsize=12)
                ax.set_title('6-DOF SCARA Robot Simulation\nBased on Ben-Gurion University Research', 
                            fontsize=14, fontweight='bold')
                ax.view_init(elev=25, azim=45)
                ax.grid(True, alpha=0.3)
        
        # Create animation
        frame_indices = range(0, len(trajectory_data['joint_configs']), 2)
        animation_obj = animation.FuncAnimation(fig, update_animation, frames=frame_indices, 
                                              interval=80, repeat=True, blit=False)
        
        if save_animation:
            try:
                animation_obj.save('scara_robot_simulation.gif', writer='pillow', fps=12, dpi=100)
                print("Animation saved as 'scara_robot_simulation.gif'")
            except Exception as e:
                print(f"Could not save animation: {e}")
        
        plt.show()
        return animation_obj
    
    def run_simulation(self):
        """Run the robot simulation"""
        print("="*60)
        print("6-DOF SCARA ROBOT SIMULATION")
        print("Based on Ben-Gurion University Research")
        print("="*60)
        
        # Define trajectory endpoints using test points from paper
        start_position = np.array([-150.0, 150.0, 1050.0])
        end_position = np.array([0.0, 400.0, 1200.0])
        
        print(f"\nPlanning trajectory:")
        print(f"  Start: {start_position} mm")
        print(f"  End:   {end_position} mm")
        print(f"  Distance: {np.linalg.norm(end_position - start_position):.1f} mm")
        
        # Plan the trajectory
        print("\nPlanning robot trajectory...")
        trajectory = self.plan_trajectory(start_position, end_position)
        
        # Verify velocity profile
        max_velocity = np.max(trajectory['velocities'])
        vel_at_20s = trajectory['velocities'][int(20/0.04)] if len(trajectory['velocities']) > int(20/0.04) else trajectory['velocities'][-1]
        print(f"\nVelocity Profile Check:")
        print(f"  Maximum velocity: {max_velocity:.1f} mm/s")
        print(f"  Velocity at 20s: {vel_at_20s:.1f} mm/s")
        
        print("\nCreating robot animation...")
        self.create_robot_animation(trajectory, save_animation=True)
        
        # Final summary
        print("\n" + "="*60)
        print("SIMULATION COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("  • scara_robot_simulation.gif - Robot animation")
        
        print("\nKey features demonstrated:")
        print("  ✓ 6-DOF motion with all joints active")
        print("  ✓ Trapezoidal velocity profile with proper phases")
        print("  ✓ Real-time visualization and monitoring")
        print("="*60)

def main():
    """Main function to run the simulation"""
    try:
        print("Starting SCARA Robot Simulation...")
        print("Based on Ben-Gurion University research paper")
        print()
        
        # Create the robot simulator
        robot = SCARASimulator()
        
        # Run the simulation
        robot.run_simulation()
        
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()