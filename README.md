# Optimal Missile Guidance Using LQG Control

This project implements an optimal missile guidance system based on the Linear Quadratic Gaussian framework.  
The goal is to guide a pursuer toward a maneuvering target while minimizing the final miss distance and limiting control effort.

The system combines optimal control, stochastic modeling, Riccati equation solvers, and Kalman filtering for state estimation under noisy measurements.

![Missile Guidance Animation](media/missile_guidance.gif)

## Project Overview

The engagement is modeled as a planar pursuit problem between a missile and a maneuvering target.  
The target acceleration is treated as a stochastic process, while the missile receives noisy line-of-sight measurements.

The solution follows the LQG framework:

- A control Riccati equation is solved backward in time to compute the optimal feedback gains.
- A filtering Riccati equation is solved forward in time to compute the Kalman gains.
- A Kalman filter estimates the unmeasured states.
- The optimal control law uses the estimated state to generate the pursuer acceleration command.

## Main Features

- Linear stochastic state-space model
- Time-varying LQG control law
- Backward Riccati solver for optimal feedback control
- Forward Riccati solver for Kalman state estimation
- Closed-loop simulation of missile-target engagement
- Visualization of state trajectories, gains, control input, and cost-to-go
- Animated engagement scenario

## Repository Structure

```text
optimal-missile-guidance-lqg/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── optimal_missile_guidance.py
│   └── animation_missile_guidance.py
│
├── media/
│   └── missile_guidance.gif
│
└── report/
    └── Optimal_Control_Project.pdf
```

## How to Run

Install the required Python packages:

```bash
pip install numpy matplotlib pillow
```

Run the main simulation:

```bash
python src/optimal_missile_guidance.py
```

Run the animation script:

```bash
python src/animation_missile_guidance.py
```

## Project Report

The full mathematical derivation, numerical implementation, and result analysis are available here:

[Open the full project report](report/Optimal_Control_Project.pdf)

## Methods Used

- Linear Quadratic Gaussian control
- Kalman filtering
- Riccati differential equations
- Stochastic state-space modeling
- Monte Carlo simulation
- Python-based numerical integration and visualization

## Technologies

- Python
- NumPy
- Matplotlib
- Pillow

## Author

Yuval Marmor
