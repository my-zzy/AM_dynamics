# Backstepping controller for platform attitude

## 1. Start from the System Form

The paper rewrites the attitude dynamics into a general second-order nonlinear system:

$$
\dot{x}_1 = f_1(x_1) + g_1(x_1)x_2
$$
$$
\dot{x}_2 = f_2(x_1, x_2) + g_2(x_1, x_2)u
$$

For the aerial platform:

* $x_1 = \text{Euler angles} = \phi, \psi, \gamma$
* $x_2 = \omega$ (angular velocity)
* $u = \tau$ (control torque)

---

## 2. Define Tracking Error

Let desired attitude be (x_{1d}).

$$
e_1 = x_{1d} - x_1
$$

---

## 3. Step 1 — Virtual Control Design

We treat $x_2$ as a **virtual control input** for the first equation.

---

### Lyapunov Function

$$
V_1 = \frac{1}{2} e_1^T e_1
$$

---

### Time Derivative

$$
\dot{V}_1 = e_1^T \dot{e}_1
$$

$$
\dot{e}*1 = \dot{x}*{1d} - \dot{x}_1
$$

Substitute system:

$$
\dot{e}*1 = \dot{x}*{1d} - f_1(x_1) - g_1(x_1)x_2
$$

---

### Choose Virtual Control

Define desired virtual control $x_2^*$:

$$
x_2^* = g_1^{-1}(x_1)\left(\dot{x}_{1d} - f_1(x_1) + \alpha_1 e_1 \right)
$$

where:

* $\alpha_1 > 0$

---

### Result

Substitute into $\dot{V}_1$:

$$
\dot{V}_1 = -\alpha_1 e_1^T e_1 < 0
$$

👉 First subsystem is stabilized.

---

## 4. Step 2 — Define Second Error

Now define:

$$
e_2 = x_2 - x_2^*
$$

---

## 5. Step 2 — Full Lyapunov Function

$$
V_2 = \frac{1}{2}(e_1^T e_1 + e_2^T e_2)
$$

---

## 6. Derivative of $e_2$

$$
\dot{e}_2 = \dot{x}_2 - \dot{x}_2^*
$$

Substitute system:

$$
\dot{e}_2 = f_2(x_1,x_2) + g_2(x_1,x_2)u - \dot{x}_2^*
$$

---

## 7. Time Derivative of $V_2$

$$
\dot{V}_2 = e_1^T \dot{e}_1 + e_2^T \dot{e}_2
$$

Substitute previous results:

$$
\dot{V}_2 = -\alpha_1 e_1^T e_1 + e_2^T (f_2 + g_2 u - \dot{x}_2^*)
$$

---

## 8. Design Control Input $u$

Choose:

$$
u = g_2^{-1}(x_1,x_2)\left(
\dot{x}_2^* - f_2(x_1,x_2)

* \alpha_2 e_2
* g_1^T(x_1)e_1
  \right)
  $$

where:

* $\alpha_2 > 0$

---

## 9. Substitute Back

Then:

$$
\dot{V}_2 =
-\alpha_1 e_1^T e_1
-\alpha_2 e_2^T e_2
< 0
$$

👉 System is asymptotically stable.

---

## 10. Final Control Law

$$
u =
g_2^{-1}
\left(
\dot{x}_2^* - f_2

* \alpha_2 e_2
* g_1^T e_1
  \right)
  $$

---

## 11. Mapping to the Paper’s Form

The paper simplifies this into:

$$
u = g_2^{-1}(\dot{\alpha} - f_2 + \alpha_2 e_2 + e_1^T e_1)
$$

👉 Same idea:

* stabilize $e_1$
* stabilize $e_2$
* ensure Lyapunov decrease

---

## 12. Key Intuition

Backstepping works by:

1. Stabilizing position error $e_1$
2. Forcing velocity $x_2$ to follow desired $x_2^*$
3. Designing torque $u$ to enforce that

---

## 13. Why Backstepping is Used Here

Because attitude dynamics:

* are nonlinear
* strongly coupled
* not well handled by linear controllers

Backstepping:

* guarantees stability
* works directly on nonlinear system

---

## 14. One-Line Summary

Backstepping designs the torque $u$ by:

> first stabilizing attitude error with a virtual angular velocity, then forcing the real angular velocity to follow it using Lyapunov-based design.

---

If you want, I can next:

* map this derivation **exactly to Euler-angle dynamics in the paper (Eq. 34)**
* or convert this into a **quaternion-based version (better for drones)**
