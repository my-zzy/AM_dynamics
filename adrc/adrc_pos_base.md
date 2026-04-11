# ADRC for Platform Position Control — Summary

This summarizes the **ADRC (Active Disturbance Rejection Control)** used for platform position control, including all key equations and concise explanations.

---

## 1. System Model (After Compensation)

The platform dynamics are simplified to a second-order system:

$$
\dot{x}_1 = x_2
$$
$$
\dot{x}_2 = \frac{1}{m_A} u
$$

- $x_1$: position  
- $x_2$: velocity  
- $u$: control input  

---

## 2. Introduce Total Disturbance

Real system includes unknown disturbance:

$$
\dot{x}_2 = \frac{1}{m_A} u + f(x,t)
$$

- $f(x,t)$: total disturbance (unknown dynamics + external effects)

---

## 3. Extended State Formulation

Define an extended state:

$$
x_3 = f(x,t)
$$

Then:

$$
\dot{x}_1 = x_2
$$
$$
\dot{x}_2 = \frac{1}{m_A} u + x_3
$$
$$
\dot{x}_3 = \dot{f}(x,t)
$$

👉 Converts unknown dynamics into a state to be estimated.

---

## 4. Tracking Differentiator (TD)

Generates smooth reference signals:

$$
v_1 \rightarrow x_d, \quad v_2 \rightarrow \dot{x}_d
$$

Discrete form:

$$
v_1^{k+1} = v_1^k + h v_2^k
$$
$$
v_2^{k+1} = v_2^k + h \cdot fhan(v_1^k - x_d, v_2^k)
$$

👉 Purpose:
- smooth reference input
- avoid noise from direct differentiation

---

## 5. Extended State Observer (ESO)

Estimate states and disturbance:

$$
z_1 \approx x_1,\quad z_2 \approx x_2,\quad z_3 \approx x_3
$$

Discrete form:

$$
e = z_1 - y
$$

$$
z_1^{k+1} = z_1^k + h (z_2^k - \beta_1 e)
$$

$$
z_2^{k+1} = z_2^k + h (z_3^k - \beta_2 fal(e, \alpha_1, \delta) + b_0 u)
$$

$$
z_3^{k+1} = z_3^k - h \beta_3 fal(e, \alpha_2, \delta)
$$

- $y = x_1$: measured output  
- $z_3$: estimated disturbance  

---

## 6. Nonlinear Function

$$
fal(e, \alpha, \delta) =
\begin{cases}
|e|^\alpha \operatorname{sgn}(e), & |e| > \delta \\
\frac{e}{\delta^{1-\alpha}}, & |e| \le \delta
\end{cases}
$$

👉 Properties:
- small error → high gain (precision)
- large error → low gain (stability)

---

## 7. Tracking Errors

$$
e_1 = v_1 - z_1
$$
$$
e_2 = v_2 - z_2
$$

---

## 8. Nonlinear Feedback Control Law

$$
u_0 = k_1 fal(e_1, \alpha_1, \delta) + k_2 fal(e_2, \alpha_2, \delta)
$$

👉 Equivalent to nonlinear PD control.

---

## 9. Disturbance Compensation (Core Step)

Final control input:

$$
u = \frac{u_0 - z_3}{b_0}
$$

- $z_3$: estimated disturbance  
- $b_0 \approx \frac{1}{m_A}$

---

## 10. Closed-Loop Result

Substitute into dynamics:

$$
\dot{x}_2 = \frac{1}{m_A} u + f(x,t)
$$

$$
\dot{x}_2 \approx u_0
$$

👉 Disturbance is canceled.

---

## 11. Final System

$$
\dot{x}_1 = x_2
$$
$$
\dot{x}_2 = u_0
$$

👉 Reduced to a disturbance-free double integrator.

---

## 12. Key Interpretation

- TD → smooth reference  
- ESO → estimate disturbance  
- Feedback → track reference  
- Compensation → cancel disturbance  

---

## 13. Final Control Structure

$$
u =
\frac{
k_1 fal(v_1 - z_1) + k_2 fal(v_2 - z_2) - z_3
}{b_0}
$$

---

## 14. Core Insight

ADRC transforms:

$$
\text{Nonlinear disturbed system}
$$

into:

$$
\text{Simple double integrator}
$$

by:

- estimating disturbance
- canceling it in real time