# Mechanical Structure (Quaternion Formulation)

The proposed aerial manipulator system (AMS) consists of a fully-actuated aerial platform equipped with a 4-DOF robotic arm. The platform uses eight tilted rotors (with inclination angle $\beta_i < 60^\circ$), enabling generation of forces and torques in all directions, thus achieving full actuation.


## Coordinate Systems and Pose Representation

Three coordinate systems are defined:

- Inertial frame: $\{W\}$
- Aerial platform frame: $\{A\}$
- Manipulator joint frames: $\{M_i\}$

The orientation of the aerial platform is represented using a unit quaternion:

$$
\mathbf{q}_A = [q_x, q_y, q_z, q_w]^T, \quad \|\mathbf{q}_A\| = 1
$$

The homogeneous transformation matrix becomes:

$$
{}^{0}T_A =
\begin{bmatrix}
{}^{0}R_A(\mathbf{q}_A) & {}^{0}p_A \\
\mathbf{0}_{1\times3} & 1
\end{bmatrix}
$$

where the rotation matrix is obtained from the quaternion:

$$
{}^{0}R_A(\mathbf{q}_A) =
\begin{bmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_z q_w) & 2(q_x q_z + q_y q_w) \\
2(q_x q_y + q_z q_w) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_x q_w) \\
2(q_x q_z - q_y q_w) & 2(q_y q_z + q_x q_w) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}
$$

## Manipulator Kinematics (Craig Parameters)

The transformation between two adjacent joints is:

$$
{}^{i-1}T_i =
\begin{bmatrix}
{}^{i-1}R_i & {}^{i-1}p_i \\
\mathbf{0}_{1\times3} & 1
\end{bmatrix}
$$

Using Craig parameters $\langle \alpha, a, d, \theta \rangle$, the forward kinematics is:

$$
{}^{0}T_i = {}^{0}T_A \prod_{k=1}^{i} {}^{k-1}T_k
$$

The position relationships are:

$$
\begin{cases}
{}^{0}p_i = {}^{0}p_{i-1} + {}^{0}R_{i-1} \, {}^{i-1}p_i \\
{}^{0}p_{c_i} = {}^{0}p_i + {}^{0}R_i \, {}^{i}p_{c_i}
\end{cases}
$$

## Quaternion Kinematics

The quaternion kinematics replaces the Euler angle rate equation.

Define angular velocity:

$$
{}^{0}\omega_A = [\omega_x, \omega_y, \omega_z]^T
$$

Quaternion derivative:

$$
\dot{\mathbf{q}}_A = \frac{1}{2} \mathbf{\Omega}({}^{0}\omega_A)\mathbf{q}_A
$$

where:

$$
\mathbf{\Omega}({}^{0}\omega_A) =
\begin{bmatrix}
0 & \omega_z & -\omega_y & \omega_x \\
-\omega_z & 0 & \omega_x & \omega_y \\
\omega_y & -\omega_x & 0 & \omega_z \\
-\omega_x & -\omega_y & -\omega_z & 0
\end{bmatrix}
$$

## Velocity Recursion

Angular velocity recursion:

$$
{}^{0}\omega_i = {}^{0}\omega_{i-1} + {}^{0}R_i \, \dot{\theta}_i \, \mathbf{z}_0
$$

Linear velocity recursion:

$$
\begin{cases}
{}^{0}\dot{p}_i = {}^{0}\dot{p}_{i-1} + {}^{0}\omega_{i-1} \times ({}^{0}R_{i-1} \, {}^{i-1}p_i) \\
{}^{0}\dot{p}_{c_i} = {}^{0}\dot{p}_i + {}^{0}\omega_i \times ({}^{0}R_i \, {}^{i}p_{c_i})
\end{cases}
$$

## Acceleration Recursion

Angular acceleration:

$$
{}^{0}\dot{\omega}_i =
{}^{0}\dot{\omega}_{i-1}
+ {}^{0}\omega_i \times ({}^{0}R_i \dot{\theta}_i \mathbf{z}_0)
+ {}^{0}R_i \ddot{\theta}_i \mathbf{z}_0
$$

Linear acceleration:

$$
\begin{cases}
{}^{0}\ddot{p}_i =
{}^{0}\ddot{p}_{i-1}
+ {}^{0}\dot{\omega}_{i-1} \times ({}^{0}R_{i-1} {}^{i-1}p_i)
+ {}^{0}\omega_{i-1} \times \left({}^{0}\omega_{i-1} \times ({}^{0}R_{i-1} {}^{i-1}p_i)\right) \\
{}^{0}\ddot{p}_{c_i} =
{}^{0}\ddot{p}_i
+ {}^{0}\dot{\omega}_i \times ({}^{0}R_i {}^{i}p_{c_i})
+ {}^{0}\omega_i \times \left({}^{0}\omega_i \times ({}^{0}R_i {}^{i}p_{c_i})\right)
\end{cases}
$$

## Dynamics of the Aerial Platform

Using Newton–Euler formulation:

Rotational dynamics:

$$
{}^{0}I_A \, {}^{0}\dot{\omega}_A
+ {}^{0}\omega_A \times ({}^{0}I_A \, {}^{0}\omega_A)
= {}^{0}\tau
- {}^{0}\tau_1
- {}^{0}r^{out}_A \times {}^{0}f_1
$$

Translational dynamics:

$$
m_A \, {}^{0}\ddot{p}_A
= {}^{0}f
- {}^{0}f_1
+ {}^{0}g_A
$$

## Dynamics of Manipulator Links

For link $i$:

$$
{}^{0}I_i \, {}^{0}\dot{\omega}_i =
{}^{0}\tau_i
+ {}^{0}r^{in}_i \times {}^{0}f_i
- {}^{0}\tau_{i+1}
- {}^{0}r^{out}_i \times {}^{0}f_{i+1}
- {}^{0}\omega_i \times ({}^{0}I_i {}^{0}\omega_i)
$$

$$
m_i \, {}^{0}\ddot{p}_{c_i}
= {}^{0}f_i
- {}^{0}f_{i+1}
+ {}^{0}g_i
$$

Force arm definitions:

$$
\begin{cases}
{}^{0}r^{in}_i = {}^{0}p_i - {}^{0}p_{c_i} \\
{}^{0}r^{out}_i = {}^{0}p_{i+1} - {}^{0}p_{c_i}
\end{cases}
$$

## Key Insight

Equations (1)--(11) establish:

- Full kinematic relationships between platform and manipulator
- Recursive velocity and acceleration propagation
- Coupled dynamics between aerial platform and robotic arm


The system is a strongly coupled, nonlinear, multi-body system where interaction forces between the manipulator and aerial platform significantly influence stability and control.

## Key Modification Summary

- Euler angle representation is fully replaced by quaternion $\mathbf{q}_A$
- Euler rate mapping is removed and replaced by quaternion differential equation
- Rotation matrix is now computed from quaternion
- All dynamic equations remain structurally unchanged but are driven by quaternion-based attitude representation