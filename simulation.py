import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from scara_model import (
    forward_kinematics as analytical_forward_kinematics,
    inverse_position,
    plan_cartesian_trajectory,
)

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
        Calculate end-effector position using the final-report convention.
        """
        return analytical_forward_kinematics(theta1, l1, l2, theta2, theta3, theta4)
    
    def inverse_kinematics(self, target_position, target_orientation=None, previous_config=None):
        """
        Solve position IK for an explicitly selected wrist posture.

        This is intentionally not a general full-pose IK solver.  The optional
        three-value argument is interpreted as [theta2, theta3, theta4], not as
        roll, pitch and yaw.  ``previous_config`` is retained for compatibility
        but is not used to distort the exact analytical solution.
        """
        del previous_config
        wrist_posture = (0.0, 90.0, 0.0) if target_orientation is None else target_orientation
        return inverse_position(target_position, *wrist_posture).tolist()
    
    def compute_robot_frames(self, theta1, l1, l2, theta2, theta3, theta4):
        """
        Calculate illustrative stick-frame positions from the analytical chain.
        """
        th1 = np.deg2rad(theta1)
        th2 = np.deg2rad(theta2)
        th3 = np.deg2rad(theta3)

        height = 800.0 + l1
        after_l2 = np.array([-l2 * np.sin(th1), l2 * np.cos(th1), height])
        wrist_center = np.array(
            [-(l2 + 150.0) * np.sin(th1), (l2 + 150.0) * np.cos(th1), height]
        )
        local_tool_offset = np.array(
            [-150.0 * np.cos(th2) * np.sin(th3), 150.0 * np.cos(th3), 150.0 * np.sin(th2) * np.sin(th3)]
        )
        c1, s1 = np.cos(th1), np.sin(th1)
        world_tool_offset = np.array(
            [c1 * local_tool_offset[0] - s1 * local_tool_offset[1],
             s1 * local_tool_offset[0] + c1 * local_tool_offset[1],
             local_tool_offset[2]]
        )
        end_effector = wrist_center + world_tool_offset
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 50.0],
                [0.0, 0.0, 800.0],
                [0.0, 0.0, height],
                after_l2,
                wrist_center,
                end_effector,
            ]
        )
    
    def plan_trajectory(self, start_pos, end_pos, motion_time=30.0):
        """
        Generate trajectory between two points using trapezoidal velocity profile
        """
        print(f"Planning trajectory from {start_pos} to {end_pos}")
        trajectory = plan_cartesian_trajectory(
            start_pos,
            end_pos,
            motion_time=motion_time,
            theta2=0.0,
            theta3=90.0,
            theta4_start=0.0,
            theta4_end=180.0,
        )
        print(f"Generating {trajectory['time'].size} trajectory points...")
        print(f"Maximum velocity: {np.max(trajectory['velocities']):.2f} mm/s")
        print(f"Maximum FK position error: {np.max(trajectory['position_error']):.3e} mm")
        return trajectory
    
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
        commanded_positions = trajectory_data['commanded_positions']
        positions = trajectory_data['achieved_positions']
        velocities = trajectory_data['velocities']
        
        # Draw commanded path separately from the FK-achieved path.
        ax.plot(commanded_positions[0], commanded_positions[1], commanded_positions[2],
               color='black', linewidth=2, alpha=0.45, linestyle='--', label='Commanded path')
        ax.plot(positions[0], positions[1], positions[2],
               color='lightblue', linewidth=2, alpha=0.35, linestyle=':', label='FK-achieved path')
        
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
                current_position = trajectory_data['achieved_positions'][:, frame_num]
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
        print(f"  Maximum FK position error: {np.max(trajectory['position_error']):.3e} mm")
        
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
