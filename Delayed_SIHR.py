#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

lambda_ = 0.8
alpha = 0.3
beta = 0.2
delta = 0.6
xi = 0.5
eta = 0.5
zeta = 0.3

k_bar = 10.0       # <-- CHANGE THIS to your chosen average degree
tau = 2.0

# External event parameters
gamma_e = 0.1     # provisional test value only
t_event = 5.0     # provisional event onset time

# ============================================================
# NUMERICAL SETTINGS
# ============================================================

h = 0.01
T = 30.0

time = np.arange(0, T + h, h)
n = len(time)

# ============================================================
# INITIAL CONDITIONS
# ============================================================

N = 1000000

I0 = (N - 1) / N
S0 = 1 / N
H0 = 0.0
R0 = 0.0

# ============================================================
# STORAGE
# ============================================================

I = np.zeros(n)
S = np.zeros(n)
H = np.zeros(n)
R = np.zeros(n)

I[0] = I0
S[0] = S0
H[0] = H0
R[0] = R0

# ============================================================
# HISTORY FUNCTIONS
# ============================================================
# Constant history on [-tau, 0].
# This is consistent with the initial state.

def history_I(t):
    return I0

def history_S(t):
    return S0

def history_H(t):
    return H0

def history_R(t):
    return R0


# ============================================================
# EXTERNAL EVENT FUNCTION
# ============================================================

def gamma_function(t):
    if t < t_event:
        return 0.0
    else:
        return gamma_e


# ============================================================
# DELAYED S(t - tau)
# ============================================================

def delayed_S(t, current_index):
    delayed_time = t - tau

    # Before t = 0: use prescribed history
    if delayed_time <= 0:
        return history_S(delayed_time)

    # For t > 0: interpolate from already computed solution
    return np.interp(
        delayed_time,
        time[:current_index + 1],
        S[:current_index + 1]
    )


# ============================================================
# RIGHT-HAND SIDE
# ============================================================

def derivatives(t, I_val, S_val, H_val, R_val, S_delay):

    gamma_t = gamma_function(t)

    dI = -(lambda_ + beta) * k_bar * I_val * S_val

    dS = (
        lambda_ * k_bar * I_val * S_val
        - alpha * k_bar * S_val * (S_val + H_val + R_val)
        - delta * S_val
        + xi * H_val
        + eta * k_bar * H_val * S_val
        + zeta * k_bar * H_val * S_delay
        + gamma_t * H_val
    )

    dH = (
        delta * S_val
        - xi * H_val
        - eta * k_bar * H_val * S_val
        - zeta * k_bar * H_val * S_delay
        - gamma_t * H_val
    )

    dR = (
        beta * k_bar * I_val * S_val
        + alpha * k_bar * S_val * (S_val + H_val + R_val)
    )

    return dI, dS, dH, dR


# ============================================================
# RK4 METHOD
# ============================================================

for j in range(n - 1):

    t = time[j]

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------
    S_delay_1 = delayed_S(t, j)

    k1_I, k1_S, k1_H, k1_R = derivatives(
        t,
        I[j],
        S[j],
        H[j],
        R[j],
        S_delay_1
    )

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------
    t2 = t + h / 2

    S_delay_2 = delayed_S(t2, j)

    I2 = I[j] + (h / 2) * k1_I
    S2 = S[j] + (h / 2) * k1_S
    H2 = H[j] + (h / 2) * k1_H
    R2 = R[j] + (h / 2) * k1_R

    k2_I, k2_S, k2_H, k2_R = derivatives(
        t2,
        I2,
        S2,
        H2,
        R2,
        S_delay_2
    )

    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------
    t3 = t + h / 2

    S_delay_3 = delayed_S(t3, j)

    I3 = I[j] + (h / 2) * k2_I
    S3 = S[j] + (h / 2) * k2_S
    H3 = H[j] + (h / 2) * k2_H
    R3 = R[j] + (h / 2) * k2_R

    k3_I, k3_S, k3_H, k3_R = derivatives(
        t3,
        I3,
        S3,
        H3,
        R3,
        S_delay_3
    )

    # --------------------------------------------------------
    # Stage 4
    # --------------------------------------------------------
    t4 = t + h

    S_delay_4 = delayed_S(t4, j)

    I4 = I[j] + h * k3_I
    S4 = S[j] + h * k3_S
    H4 = H[j] + h * k3_H
    R4 = R[j] + h * k3_R

    k4_I, k4_S, k4_H, k4_R = derivatives(
        t4,
        I4,
        S4,
        H4,
        R4,
        S_delay_4
    )

    # --------------------------------------------------------
    # RK4 UPDATE
    # --------------------------------------------------------

    I[j + 1] = I[j] + (h / 6) * (
        k1_I + 2*k2_I + 2*k3_I + k4_I
    )

    S[j + 1] = S[j] + (h / 6) * (
        k1_S + 2*k2_S + 2*k3_S + k4_S
    )

    H[j + 1] = H[j] + (h / 6) * (
        k1_H + 2*k2_H + 2*k3_H + k4_H
    )

    R[j + 1] = R[j] + (h / 6) * (
        k1_R + 2*k2_R + 2*k3_R + k4_R
    )


# ============================================================
# CHECK 1: TOTAL POPULATION
# ============================================================

total = I + S + H + R

print("Initial total population =", total[0])
print("Final total population   =", total[-1])
print("Maximum conservation error =",
      np.max(np.abs(total - total[0])))


# ============================================================
# CHECK 2: NON-NEGATIVITY
# ============================================================

print("\nMinimum values:")
print("I minimum =", np.min(I))
print("S minimum =", np.min(S))
print("H minimum =", np.min(H))
print("R minimum =", np.min(R))


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(time, I, label="I(t)")
plt.plot(time, S, label="S(t)")
plt.plot(time, H, label="H(t)")
plt.plot(time, R, label="R(t)")

plt.axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

plt.xlabel("Time(days)")
plt.ylabel("Population proportion")
plt.title("Delayed SIHR Rumour Model with External Reactivation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

lambda_ = 0.8
alpha = 0.3
beta = 0.2
delta = 0.6
xi = 0.5
eta = 0.5
zeta = 0.3

k_bar = 10.0       # <-- CHANGE THIS to your chosen average degree
tau = 2.0

# External event parameters
gamma_e = 0.1     # provisional test value only
t_event = 5.0     # provisional event onset time

# ============================================================
# NUMERICAL SETTINGS
# ============================================================

h = 0.005
T = 30.0

time = np.arange(0, T + h, h)
n = len(time)

# ============================================================
# INITIAL CONDITIONS
# ============================================================

N = 1000000

I0 = (N - 1) / N
S0 = 1 / N
H0 = 0.0
R0 = 0.0

# ============================================================
# STORAGE
# ============================================================

I = np.zeros(n)
S = np.zeros(n)
H = np.zeros(n)
R = np.zeros(n)

I[0] = I0
S[0] = S0
H[0] = H0
R[0] = R0

# ============================================================
# HISTORY FUNCTIONS
# ============================================================
# Constant history on [-tau, 0].
# This is consistent with the initial state.

def history_I(t):
    return I0

def history_S(t):
    return S0

def history_H(t):
    return H0

def history_R(t):
    return R0


# ============================================================
# EXTERNAL EVENT FUNCTION
# ============================================================

def gamma_function(t):
    if t < t_event:
        return 0.0
    else:
        return gamma_e


# ============================================================
# DELAYED S(t - tau)
# ============================================================

def delayed_S(t, current_index):
    delayed_time = t - tau

    # Before t = 0: use prescribed history
    if delayed_time <= 0:
        return history_S(delayed_time)

    # For t > 0: interpolate from already computed solution
    return np.interp(
        delayed_time,
        time[:current_index + 1],
        S[:current_index + 1]
    )


# ============================================================
# RIGHT-HAND SIDE
# ============================================================

def derivatives(t, I_val, S_val, H_val, R_val, S_delay):

    gamma_t = gamma_function(t)

    dI = -(lambda_ + beta) * k_bar * I_val * S_val

    dS = (
        lambda_ * k_bar * I_val * S_val
        - alpha * k_bar * S_val * (S_val + H_val + R_val)
        - delta * S_val
        + xi * H_val
        + eta * k_bar * H_val * S_val
        + zeta * k_bar * H_val * S_delay
        + gamma_t * H_val
    )

    dH = (
        delta * S_val
        - xi * H_val
        - eta * k_bar * H_val * S_val
        - zeta * k_bar * H_val * S_delay
        - gamma_t * H_val
    )

    dR = (
        beta * k_bar * I_val * S_val
        + alpha * k_bar * S_val * (S_val + H_val + R_val)
    )

    return dI, dS, dH, dR


# ============================================================
# RK4 METHOD
# ============================================================

for j in range(n - 1):

    t = time[j]

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------
    S_delay_1 = delayed_S(t, j)

    k1_I, k1_S, k1_H, k1_R = derivatives(
        t,
        I[j],
        S[j],
        H[j],
        R[j],
        S_delay_1
    )

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------
    t2 = t + h / 2

    S_delay_2 = delayed_S(t2, j)

    I2 = I[j] + (h / 2) * k1_I
    S2 = S[j] + (h / 2) * k1_S
    H2 = H[j] + (h / 2) * k1_H
    R2 = R[j] + (h / 2) * k1_R

    k2_I, k2_S, k2_H, k2_R = derivatives(
        t2,
        I2,
        S2,
        H2,
        R2,
        S_delay_2
    )

    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------
    t3 = t + h / 2

    S_delay_3 = delayed_S(t3, j)

    I3 = I[j] + (h / 2) * k2_I
    S3 = S[j] + (h / 2) * k2_S
    H3 = H[j] + (h / 2) * k2_H
    R3 = R[j] + (h / 2) * k2_R

    k3_I, k3_S, k3_H, k3_R = derivatives(
        t3,
        I3,
        S3,
        H3,
        R3,
        S_delay_3
    )

    # --------------------------------------------------------
    # Stage 4
    # --------------------------------------------------------
    t4 = t + h

    S_delay_4 = delayed_S(t4, j)

    I4 = I[j] + h * k3_I
    S4 = S[j] + h * k3_S
    H4 = H[j] + h * k3_H
    R4 = R[j] + h * k3_R

    k4_I, k4_S, k4_H, k4_R = derivatives(
        t4,
        I4,
        S4,
        H4,
        R4,
        S_delay_4
    )

    # --------------------------------------------------------
    # RK4 UPDATE
    # --------------------------------------------------------

    I[j + 1] = I[j] + (h / 6) * (
        k1_I + 2*k2_I + 2*k3_I + k4_I
    )

    S[j + 1] = S[j] + (h / 6) * (
        k1_S + 2*k2_S + 2*k3_S + k4_S
    )

    H[j + 1] = H[j] + (h / 6) * (
        k1_H + 2*k2_H + 2*k3_H + k4_H
    )

    R[j + 1] = R[j] + (h / 6) * (
        k1_R + 2*k2_R + 2*k3_R + k4_R
    )


# ============================================================
# CHECK 1: TOTAL POPULATION
# ============================================================

total = I + S + H + R

print("Initial total population =", total[0])
print("Final total population   =", total[-1])
print("Maximum conservation error =",
      np.max(np.abs(total - total[0])))


# ============================================================
# CHECK 2: NON-NEGATIVITY
# ============================================================

print("\nMinimum values:")
print("I minimum =", np.min(I))
print("S minimum =", np.min(S))
print("H minimum =", np.min(H))
print("R minimum =", np.min(R))


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(time, I, label="I(t)")
plt.plot(time, S, label="S(t)")
plt.plot(time, H, label="H(t)")
plt.plot(time, R, label="R(t)")

plt.axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

plt.xlabel("Time(days)")
plt.ylabel("Population proportion")
plt.title("Delayed SIHR Rumour Model with External Reactivation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

lambda_ = 0.8
alpha = 0.3
beta = 0.2
delta = 0.6
xi = 0.5
eta = 0.5
zeta = 0.3

N = 1000000
k_bar = 10

tau = 2.0          # delay in days
t_event = 2.0      # external event begins at day 2

# Values for sensitivity analysis
gamma_values = [0.0, 0.1, 0.2, 0.5]

# ============================================================
# NUMERICAL SETTINGS
# ============================================================

h = 0.01           # time step in days
T = 10.0           # simulation time in days

time = np.arange(0, T + h, h)
n = len(time)

# ============================================================
# INITIAL CONDITIONS
# ============================================================

I0 = (N - 1) / N
S0 = 1 / N
H0 = 0.0
R0 = 0.0


# ============================================================
# EXTERNAL EVENT FUNCTION
# ============================================================

def gamma_function(t, gamma_e):
    if t < t_event:
        return 0.0
    else:
        return gamma_e


# ============================================================
# RUN MODEL FOR ONE gamma_e
# ============================================================

def run_model(gamma_e):

    # Storage
    I = np.zeros(n)
    S = np.zeros(n)
    H = np.zeros(n)
    R = np.zeros(n)

    # Initial conditions
    I[0] = I0
    S[0] = S0
    H[0] = H0
    R[0] = R0

    # --------------------------------------------------------
    # History function for S(t), -tau <= t <= 0
    # --------------------------------------------------------

    def history_S(t):
        return S0

    # --------------------------------------------------------
    # Delayed S(t - tau)
    # --------------------------------------------------------

    def delayed_S(t, current_index):

        delayed_time = t - tau

        # Use prescribed history before t = 0
        if delayed_time <= 0:
            return history_S(delayed_time)

        # Interpolate previously computed S values
        return np.interp(
            delayed_time,
            time[:current_index + 1],
            S[:current_index + 1]
        )

    # --------------------------------------------------------
    # Right-hand side
    # --------------------------------------------------------

    def derivatives(t, I_val, S_val, H_val, R_val, S_delay):

        gamma_t = gamma_function(t, gamma_e)

        dI = (
            -(lambda_ + beta)
            * k_bar
            * I_val
            * S_val
        )

        dS = (
            lambda_ * k_bar * I_val * S_val
            - alpha * k_bar * S_val
              * (S_val + H_val + R_val)
            - delta * S_val
            + xi * H_val
            + eta * k_bar * H_val * S_val
            + zeta * k_bar * H_val * S_delay
            + gamma_t * H_val
        )

        dH = (
            delta * S_val
            - xi * H_val
            - eta * k_bar * H_val * S_val
            - zeta * k_bar * H_val * S_delay
            - gamma_t * H_val
        )

        dR = (
            beta * k_bar * I_val * S_val
            + alpha * k_bar * S_val
              * (S_val + H_val + R_val)
        )

        return dI, dS, dH, dR

    # ========================================================
    # RK4
    # ========================================================

    for j in range(n - 1):

        t = time[j]

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        S_delay_1 = delayed_S(t, j)

        k1_I, k1_S, k1_H, k1_R = derivatives(
            t,
            I[j],
            S[j],
            H[j],
            R[j],
            S_delay_1
        )

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        t2 = t + h / 2

        S_delay_2 = delayed_S(t2, j)

        I2 = I[j] + (h / 2) * k1_I
        S2 = S[j] + (h / 2) * k1_S
        H2 = H[j] + (h / 2) * k1_H
        R2 = R[j] + (h / 2) * k1_R

        k2_I, k2_S, k2_H, k2_R = derivatives(
            t2,
            I2,
            S2,
            H2,
            R2,
            S_delay_2
        )

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        t3 = t + h / 2

        S_delay_3 = delayed_S(t3, j)

        I3 = I[j] + (h / 2) * k2_I
        S3 = S[j] + (h / 2) * k2_S
        H3 = H[j] + (h / 2) * k2_H
        R3 = R[j] + (h / 2) * k2_R

        k3_I, k3_S, k3_H, k3_R = derivatives(
            t3,
            I3,
            S3,
            H3,
            R3,
            S_delay_3
        )

        # ----------------------------------------------------
        # Stage 4
        # ----------------------------------------------------

        t4 = t + h

        S_delay_4 = delayed_S(t4, j)

        I4 = I[j] + h * k3_I
        S4 = S[j] + h * k3_S
        H4 = H[j] + h * k3_H
        R4 = R[j] + h * k3_R

        k4_I, k4_S, k4_H, k4_R = derivatives(
            t4,
            I4,
            S4,
            H4,
            R4,
            S_delay_4
        )

        # ----------------------------------------------------
        # RK4 update
        # ----------------------------------------------------

        I[j + 1] = I[j] + (h / 6) * (
            k1_I + 2*k2_I + 2*k3_I + k4_I
        )

        S[j + 1] = S[j] + (h / 6) * (
            k1_S + 2*k2_S + 2*k3_S + k4_S
        )

        H[j + 1] = H[j] + (h / 6) * (
            k1_H + 2*k2_H + 2*k3_H + k4_H
        )

        R[j + 1] = R[j] + (h / 6) * (
            k1_R + 2*k2_R + 2*k3_R + k4_R
        )

    return I, S, H, R


# ============================================================
# RUN ALL GAMMA_e SCENARIOS
# ============================================================

results = {}

for gamma_e in gamma_values:

    I, S, H, R = run_model(gamma_e)

    results[gamma_e] = {
        "I": I,
        "S": S,
        "H": H,
        "R": R
    }


# ============================================================
# PRINT NUMERICAL CHECKS
# ============================================================

print("Sensitivity analysis for gamma_e")
print("=" * 60)

for gamma_e in gamma_values:

    I = results[gamma_e]["I"]
    S = results[gamma_e]["S"]
    H = results[gamma_e]["H"]
    R = results[gamma_e]["R"]

    total = I + S + H + R

    print(f"\ngamma_e = {gamma_e}")
    print("Final total population   =", total[-1])
    print("Maximum conservation error =",
          np.max(np.abs(total - 1.0)))

    print("Minimum I =", np.min(I))
    print("Minimum S =", np.min(S))
    print("Minimum H =", np.min(H))
    print("Minimum R =", np.min(R))


# ============================================================
# PLOT: SPREADERS
# ============================================================

plt.figure(figsize=(10, 6))

for gamma_e in gamma_values:
    plt.plot(
        time,
        results[gamma_e]["S"],
        label=rf"$\gamma_e={gamma_e}$"
    )

plt.axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

plt.xlabel("Time (days)")
plt.ylabel("Spreader proportion")
plt.title("Effect of External Reactivation Strength on Spreaders")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT: HIBERNATORS
# ============================================================

plt.figure(figsize=(10, 6))

for gamma_e in gamma_values:
    plt.plot(
        time,
        results[gamma_e]["H"],
        label=rf"$\gamma_e={gamma_e}$"
    )

plt.axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

plt.xlabel("Time (days)")
plt.ylabel("Hibernator proportion")
plt.title("Effect of External Reactivation Strength on Hibernators")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT: STIFLERS
# ============================================================

plt.figure(figsize=(10, 6))

for gamma_e in gamma_values:
    plt.plot(
        time,
        results[gamma_e]["R"],
        label=rf"$\gamma_e={gamma_e}$"
    )

plt.axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

plt.xlabel("Time (days)")
plt.ylabel("Stifler proportion")
plt.title("Effect of External Reactivation Strength on Stiflers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

lambda_ = 0.8
alpha = 0.3
beta = 0.2
delta = 0.6
xi = 0.5
eta = 0.5
zeta = 0.3

k_bar = 10.0       # <-- CHANGE THIS to your chosen average degree
tau = 2.0

# External event parameters
gamma_e = 0.1     # provisional test value only
t_event = 2.0     # provisional event onset time

# ============================================================
# NUMERICAL SETTINGS
# ============================================================

h = 0.005
T = 10.0

time = np.arange(0, T + h, h)
n = len(time)

# ============================================================
# INITIAL CONDITIONS
# ============================================================

N = 1000000

I0 = (N - 1) / N
S0 = 1 / N
H0 = 0.0
R0 = 0.0

# ============================================================
# STORAGE
# ============================================================

I = np.zeros(n)
S = np.zeros(n)
H = np.zeros(n)
R = np.zeros(n)

I[0] = I0
S[0] = S0
H[0] = H0
R[0] = R0

# ============================================================
# HISTORY FUNCTIONS
# ============================================================
# Constant history on [-tau, 0].
# This is consistent with the initial state.

def history_I(t):
    return I0

def history_S(t):
    return S0

def history_H(t):
    return H0

def history_R(t):
    return R0


# ============================================================
# EXTERNAL EVENT FUNCTION
# ============================================================

def gamma_function(t):
    if t < t_event:
        return 0.0
    else:
        return gamma_e


# ============================================================
# DELAYED S(t - tau)
# ============================================================

def delayed_S(t, current_index):
    delayed_time = t - tau

    # Before t = 0: use prescribed history
    if delayed_time <= 0:
        return history_S(delayed_time)

    # For t > 0: interpolate from already computed solution
    return np.interp(
        delayed_time,
        time[:current_index + 1],
        S[:current_index + 1]
    )


# ============================================================
# RIGHT-HAND SIDE
# ============================================================

def derivatives(t, I_val, S_val, H_val, R_val, S_delay):

    gamma_t = gamma_function(t)

    dI = -(lambda_ + beta) * k_bar * I_val * S_val

    dS = (
        lambda_ * k_bar * I_val * S_val
        - alpha * k_bar * S_val * (S_val + H_val + R_val)
        - delta * S_val
        + xi * H_val
        + eta * k_bar * H_val * S_val
        + zeta * k_bar * H_val * S_delay
        + gamma_t * H_val
    )

    dH = (
        delta * S_val
        - xi * H_val
        - eta * k_bar * H_val * S_val
        - zeta * k_bar * H_val * S_delay
        - gamma_t * H_val
    )

    dR = (
        beta * k_bar * I_val * S_val
        + alpha * k_bar * S_val * (S_val + H_val + R_val)
    )

    return dI, dS, dH, dR


# ============================================================
# RK4 METHOD
# ============================================================

for j in range(n - 1):

    t = time[j]

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------
    S_delay_1 = delayed_S(t, j)

    k1_I, k1_S, k1_H, k1_R = derivatives(
        t,
        I[j],
        S[j],
        H[j],
        R[j],
        S_delay_1
    )

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------
    t2 = t + h / 2

    S_delay_2 = delayed_S(t2, j)

    I2 = I[j] + (h / 2) * k1_I
    S2 = S[j] + (h / 2) * k1_S
    H2 = H[j] + (h / 2) * k1_H
    R2 = R[j] + (h / 2) * k1_R

    k2_I, k2_S, k2_H, k2_R = derivatives(
        t2,
        I2,
        S2,
        H2,
        R2,
        S_delay_2
    )

    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------
    t3 = t + h / 2

    S_delay_3 = delayed_S(t3, j)

    I3 = I[j] + (h / 2) * k2_I
    S3 = S[j] + (h / 2) * k2_S
    H3 = H[j] + (h / 2) * k2_H
    R3 = R[j] + (h / 2) * k2_R

    k3_I, k3_S, k3_H, k3_R = derivatives(
        t3,
        I3,
        S3,
        H3,
        R3,
        S_delay_3
    )

    # --------------------------------------------------------
    # Stage 4
    # --------------------------------------------------------
    t4 = t + h

    S_delay_4 = delayed_S(t4, j)

    I4 = I[j] + h * k3_I
    S4 = S[j] + h * k3_S
    H4 = H[j] + h * k3_H
    R4 = R[j] + h * k3_R

    k4_I, k4_S, k4_H, k4_R = derivatives(
        t4,
        I4,
        S4,
        H4,
        R4,
        S_delay_4
    )

    # --------------------------------------------------------
    # RK4 UPDATE
    # --------------------------------------------------------

    I[j + 1] = I[j] + (h / 6) * (
        k1_I + 2*k2_I + 2*k3_I + k4_I
    )

    S[j + 1] = S[j] + (h / 6) * (
        k1_S + 2*k2_S + 2*k3_S + k4_S
    )

    H[j + 1] = H[j] + (h / 6) * (
        k1_H + 2*k2_H + 2*k3_H + k4_H
    )

    R[j + 1] = R[j] + (h / 6) * (
        k1_R + 2*k2_R + 2*k3_R + k4_R
    )


# ============================================================
# CHECK 1: TOTAL POPULATION
# ============================================================

total = I + S + H + R

print("Initial total population =", total[0])
print("Final total population   =", total[-1])
print("Maximum conservation error =",
      np.max(np.abs(total - total[0])))


# ============================================================
# CHECK 2: NON-NEGATIVITY
# ============================================================

print("\nMinimum values:")
print("I minimum =", np.min(I))
print("S minimum =", np.min(S))
print("H minimum =", np.min(H))
print("R minimum =", np.min(R))


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(time, I, label="I(t)")
plt.plot(time, S, label="S(t)")
plt.plot(time, H, label="H(t)")
plt.plot(time, R, label="R(t)")

plt.axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

plt.xlabel("Time(days)")
plt.ylabel("Population proportion")
plt.title("Delayed SIHR Rumour Model with External Reactivation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[6]:


# ============================================================
# VALUES AT DAYS 1–5
# ============================================================

from tabulate import tabulate

print("\nNumerical solution at days 1–5")

data = []

for day in range(1, 6):
    idx = np.argmin(np.abs(time - day))

    total_day = I[idx] + S[idx] + H[idx] + R[idx]

    data.append([
        f"{time[idx]:.2f}",
        f"{I[idx]:.10f}",
        f"{S[idx]:.10f}",
        f"{H[idx]:.10f}",
        f"{R[idx]:.10f}",
        f"{total_day:.10f}"
    ])

print(tabulate(
    data,
    headers=[
        "Time (days)",
        "Ignorants (I)",
        "Spreaders (S)",
        "Hibernators (H)",
        "Stiflers (R)",
        "Total"
    ],
    tablefmt="grid"
))


# In[7]:


import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

lambda_ = 0.8
alpha = 0.3
beta = 0.2
delta = 0.6
xi = 0.5
eta = 0.5

k_bar = 10.0
tau = 2.0

# External event parameters
gamma_e = 0.1
t_event = 2.0

# ============================================================
# NUMERICAL SETTINGS
# ============================================================

h = 0.02
T = 10.0

time = np.arange(0, T + h, h)
n = len(time)

# ============================================================
# INITIAL CONDITIONS
# ============================================================

N = 1000000

I0 = (N - 1) / N
S0 = 1 / N
H0 = 0.0
R0 = 0.0

# ============================================================
# VALUES OF ZETA TO BE SIMULATED
# ============================================================

zeta_values = [0.1, 0.3, 0.7]

# ============================================================
# EXTERNAL EVENT FUNCTION
# ============================================================

def gamma_function(t):
    if t < t_event:
        return 0.0
    else:
        return gamma_e

# ============================================================
# SIMULATION FUNCTION
# ============================================================

def simulate_zeta(zeta):

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    I = np.zeros(n)
    S = np.zeros(n)
    H = np.zeros(n)
    R = np.zeros(n)

    I[0] = I0
    S[0] = S0
    H[0] = H0
    R[0] = R0

    # --------------------------------------------------------
    # HISTORY FUNCTIONS
    # --------------------------------------------------------

    def history_S(t):
        return S0

    # --------------------------------------------------------
    # DELAYED S(t - tau)
    # --------------------------------------------------------

    def delayed_S(t, current_index):

        delayed_time = t - tau

        # Before t = 0: use prescribed history
        if delayed_time <= 0:
            return history_S(delayed_time)

        # For t > 0: interpolate previously computed S
        return np.interp(
            delayed_time,
            time[:current_index + 1],
            S[:current_index + 1]
        )

    # --------------------------------------------------------
    # RIGHT-HAND SIDE
    # --------------------------------------------------------

    def derivatives(t, I_val, S_val, H_val, R_val, S_delay):

        gamma_t = gamma_function(t)

        dI = (
            -(lambda_ + beta)
            * k_bar
            * I_val
            * S_val
        )

        dS = (
            lambda_ * k_bar * I_val * S_val
            - alpha * k_bar * S_val
              * (S_val + H_val + R_val)
            - delta * S_val
            + xi * H_val
            + eta * k_bar * H_val * S_val
            + zeta * k_bar * H_val * S_delay
            + gamma_t * H_val
        )

        dH = (
            delta * S_val
            - xi * H_val
            - eta * k_bar * H_val * S_val
            - zeta * k_bar * H_val * S_delay
            - gamma_t * H_val
        )

        dR = (
            beta * k_bar * I_val * S_val
            + alpha * k_bar * S_val
              * (S_val + H_val + R_val)
        )

        return dI, dS, dH, dR

    # ========================================================
    # RK4 METHOD
    # ========================================================

    for j in range(n - 1):

        t = time[j]

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        S_delay_1 = delayed_S(t, j)

        k1_I, k1_S, k1_H, k1_R = derivatives(
            t,
            I[j],
            S[j],
            H[j],
            R[j],
            S_delay_1
        )

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        t2 = t + h / 2

        S_delay_2 = delayed_S(t2, j)

        I2 = I[j] + (h / 2) * k1_I
        S2 = S[j] + (h / 2) * k1_S
        H2 = H[j] + (h / 2) * k1_H
        R2 = R[j] + (h / 2) * k1_R

        k2_I, k2_S, k2_H, k2_R = derivatives(
            t2,
            I2,
            S2,
            H2,
            R2,
            S_delay_2
        )

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        t3 = t + h / 2

        S_delay_3 = delayed_S(t3, j)

        I3 = I[j] + (h / 2) * k2_I
        S3 = S[j] + (h / 2) * k2_S
        H3 = H[j] + (h / 2) * k2_H
        R3 = R[j] + (h / 2) * k2_R

        k3_I, k3_S, k3_H, k3_R = derivatives(
            t3,
            I3,
            S3,
            H3,
            R3,
            S_delay_3
        )

        # ----------------------------------------------------
        # Stage 4
        # ----------------------------------------------------

        t4 = t + h

        S_delay_4 = delayed_S(t4, j)

        I4 = I[j] + h * k3_I
        S4 = S[j] + h * k3_S
        H4 = H[j] + h * k3_H
        R4 = R[j] + h * k3_R

        k4_I, k4_S, k4_H, k4_R = derivatives(
            t4,
            I4,
            S4,
            H4,
            R4,
            S_delay_4
        )

        # ----------------------------------------------------
        # RK4 UPDATE
        # ----------------------------------------------------

        I[j + 1] = I[j] + (h / 6) * (
            k1_I + 2*k2_I + 2*k3_I + k4_I
        )

        S[j + 1] = S[j] + (h / 6) * (
            k1_S + 2*k2_S + 2*k3_S + k4_S
        )

        H[j + 1] = H[j] + (h / 6) * (
            k1_H + 2*k2_H + 2*k3_H + k4_H
        )

        R[j + 1] = R[j] + (h / 6) * (
            k1_R + 2*k2_R + 2*k3_R + k4_R
        )

    return I, S, H, R


# ============================================================
# RUN ALL ZETA VALUES
# ============================================================

results = {}

for zeta in zeta_values:

    I, S, H, R = simulate_zeta(zeta)

    results[zeta] = {
        "I": I,
        "S": S,
        "H": H,
        "R": R
    }

# ============================================================
# PLOT: SIHR COMPARTMENTS — 2 × 2 COMBINED FIGURE
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ============================================================
# PLOT: IGNORANTS
# ============================================================

for zeta in zeta_values:
    axes[0, 0].plot(
        time,
        results[zeta]["I"],
        label=f"$\\zeta$ = {zeta}"
    )

axes[0, 0].axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

axes[0, 0].set_xlabel("Time (days)")
axes[0, 0].set_ylabel("Ignorants, I(t)")
axes[0, 0].set_title("Effect of Delayed Remembering ($\\zeta$) on Ignorants")
axes[0, 0].legend()
axes[0, 0].grid(True)


# ============================================================
# PLOT: SPREADERS
# ============================================================

for zeta in zeta_values:
    axes[0, 1].plot(
        time,
        results[zeta]["S"],
        label=f"$\\zeta$ = {zeta}"
    )

axes[0, 1].axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

axes[0, 1].set_xlabel("Time (days)")
axes[0, 1].set_ylabel("Spreaders, S(t)")
axes[0, 1].set_title("Effect of Delayed Remembering ($\\zeta$) on Spreaders")
axes[0, 1].legend()
axes[0, 1].grid(True)


# ============================================================
# PLOT: HIBERNATORS
# ============================================================

for zeta in zeta_values:
    axes[1, 0].plot(
        time,
        results[zeta]["H"],
        label=f"$\\zeta$ = {zeta}"
    )

axes[1, 0].axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

axes[1, 0].set_xlabel("Time (days)")
axes[1, 0].set_ylabel("Hibernators, H(t)")
axes[1, 0].set_title("Effect of Delayed Remembering ($\\zeta$) on Hibernators")
axes[1, 0].legend()
axes[1, 0].grid(True)


# ============================================================
# PLOT: STIFLERS
# ============================================================

for zeta in zeta_values:
    axes[1, 1].plot(
        time,
        results[zeta]["R"],
        label=f"$\\zeta$ = {zeta}"
    )

axes[1, 1].axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

axes[1, 1].set_xlabel("Time (days)")
axes[1, 1].set_ylabel("Stiflers, R(t)")
axes[1, 1].set_title("Effect of Delayed Remembering ($\\zeta$) on Stiflers")
axes[1, 1].legend()
axes[1, 1].grid(True)


# ============================================================
# ADJUST LAYOUT
# ============================================================

plt.tight_layout()

# ============================================================
# EXPORT THE WHOLE 2 × 2 FIGURE AS ONE FILE
# ============================================================

plt.savefig(
    "Effect_of_Delayed_Remembering_zeta_2x2.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "Effect_of_Delayed_Remembering_zeta_2x2.pdf",
    bbox_inches="tight"
)

plt.show()


# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

lambda_ = 0.8
alpha = 0.3
beta = 0.2
delta = 0.6
xi = 0.5
eta = 0.5
zeta = 0.3 

k_bar = 10.0

# External event
gamma_e = 0.1
t_event = 2.0

# Baseline delay
tau = 2.0

# ============================================================
# NUMERICAL SETTINGS
# ============================================================

h = 0.01
T = 10.0

time = np.arange(0, T + h, h)
n = len(time)

# ============================================================
# INITIAL CONDITIONS
# ============================================================

N = 1000000

I0 = (N - 1) / N
S0 = 1 / N
H0 = 0.0
R0 = 0.0


# ============================================================
# ============================================================
# PART A: EFFECT OF EXTERNAL REACTIVATION gamma(t)
# ============================================================
# ============================================================

gamma_values = [0.0, 0.1, 0.2, 0.5]


# ============================================================
# SIMULATION FUNCTION FOR GAMMA
# ============================================================

def simulate_gamma(gamma_value):

    I = np.zeros(n)
    S = np.zeros(n)
    H = np.zeros(n)
    R = np.zeros(n)

    I[0] = I0
    S[0] = S0
    H[0] = H0
    R[0] = R0

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    def history_S(t):
        return S0

    # --------------------------------------------------------
    # EXTERNAL REACTIVATION FUNCTION
    # --------------------------------------------------------

    def gamma_function(t):

        if t < t_event:
            return 0.0
        else:
            return gamma_value

    # --------------------------------------------------------
    # DELAYED S(t - tau)
    # --------------------------------------------------------

    def delayed_S(t, current_index):

        delayed_time = t - tau

        if delayed_time <= 0:
            return history_S(delayed_time)

        return np.interp(
            delayed_time,
            time[:current_index + 1],
            S[:current_index + 1]
        )

    # --------------------------------------------------------
    # RIGHT-HAND SIDE
    # --------------------------------------------------------

    def derivatives(
        t,
        I_val,
        S_val,
        H_val,
        R_val,
        S_delay
    ):

        gamma_t = gamma_function(t)

        dI = (
            -(lambda_ + beta)
            * k_bar
            * I_val
            * S_val
        )

        dS = (
            lambda_ * k_bar * I_val * S_val
            - alpha * k_bar * S_val
            * (S_val + H_val + R_val)
            - delta * S_val
            + xi * H_val
            + eta * k_bar * H_val * S_val
            + zeta * k_bar * H_val * S_delay
            + gamma_t * H_val
        )

        dH = (
            delta * S_val
            - xi * H_val
            - eta * k_bar * H_val * S_val
            - zeta * k_bar * H_val * S_delay
            - gamma_t * H_val
        )

        dR = (
            beta * k_bar * I_val * S_val
            + alpha * k_bar * S_val
            * (S_val + H_val + R_val)
        )

        return dI, dS, dH, dR

    # ========================================================
    # RK4 METHOD
    # ========================================================

    for j in range(n - 1):

        t = time[j]

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        S_delay_1 = delayed_S(t, j)

        k1_I, k1_S, k1_H, k1_R = derivatives(
            t,
            I[j],
            S[j],
            H[j],
            R[j],
            S_delay_1
        )

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        t2 = t + h / 2

        S_delay_2 = delayed_S(t2, j)

        I2 = I[j] + (h / 2) * k1_I
        S2 = S[j] + (h / 2) * k1_S
        H2 = H[j] + (h / 2) * k1_H
        R2 = R[j] + (h / 2) * k1_R

        k2_I, k2_S, k2_H, k2_R = derivatives(
            t2,
            I2,
            S2,
            H2,
            R2,
            S_delay_2
        )

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        t3 = t + h / 2

        S_delay_3 = delayed_S(t3, j)

        I3 = I[j] + (h / 2) * k2_I
        S3 = S[j] + (h / 2) * k2_S
        H3 = H[j] + (h / 2) * k2_H
        R3 = R[j] + (h / 2) * k2_R

        k3_I, k3_S, k3_H, k3_R = derivatives(
            t3,
            I3,
            S3,
            H3,
            R3,
            S_delay_3
        )

        # ----------------------------------------------------
        # Stage 4
        # ----------------------------------------------------

        t4 = t + h

        S_delay_4 = delayed_S(t4, j)

        I4 = I[j] + h * k3_I
        S4 = S[j] + h * k3_S
        H4 = H[j] + h * k3_H
        R4 = R[j] + h * k3_R

        k4_I, k4_S, k4_H, k4_R = derivatives(
            t4,
            I4,
            S4,
            H4,
            R4,
            S_delay_4
        )

        # ----------------------------------------------------
        # RK4 UPDATE
        # ----------------------------------------------------

        I[j + 1] = I[j] + (h / 6) * (
            k1_I + 2*k2_I + 2*k3_I + k4_I
        )

        S[j + 1] = S[j] + (h / 6) * (
            k1_S + 2*k2_S + 2*k3_S + k4_S
        )

        H[j + 1] = H[j] + (h / 6) * (
            k1_H + 2*k2_H + 2*k3_H + k4_H
        )

        R[j + 1] = R[j] + (h / 6) * (
            k1_R + 2*k2_R + 2*k3_R + k4_R
        )

    return I, S, H, R


# ============================================================
# RUN ALL GAMMA VALUES
# ============================================================

results_gamma = {}

for gamma_value in gamma_values:

    I, S, H, R = simulate_gamma(gamma_value)

    results_gamma[gamma_value] = {
        "I": I,
        "S": S,
        "H": H,
        "R": R
    }


# ============================================================
# ============================================================
# PART B: EFFECT OF DELAY tau
# ============================================================
# ============================================================

tau_values = [0.5, 1.0, 2.0, 3.0]


# ============================================================
# SIMULATION FUNCTION FOR TAU
# ============================================================

def simulate_tau(tau_value):

    I = np.zeros(n)
    S = np.zeros(n)
    H = np.zeros(n)
    R = np.zeros(n)

    I[0] = I0
    S[0] = S0
    H[0] = H0
    R[0] = R0

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    def history_S(t):
        return S0

    # --------------------------------------------------------
    # EXTERNAL REACTIVATION
    # gamma_e remains fixed at 0.1
    # --------------------------------------------------------

    def gamma_function(t):

        if t < t_event:
            return 0.0
        else:
            return gamma_e

    # --------------------------------------------------------
    # DELAYED S(t - tau)
    # --------------------------------------------------------

    def delayed_S(t, current_index):

        delayed_time = t - tau_value

        if delayed_time <= 0:
            return history_S(delayed_time)

        return np.interp(
            delayed_time,
            time[:current_index + 1],
            S[:current_index + 1]
        )

    # --------------------------------------------------------
    # RIGHT-HAND SIDE
    # --------------------------------------------------------

    def derivatives(
        t,
        I_val,
        S_val,
        H_val,
        R_val,
        S_delay
    ):

        gamma_t = gamma_function(t)

        dI = (
            -(lambda_ + beta)
            * k_bar
            * I_val
            * S_val
        )

        dS = (
            lambda_ * k_bar * I_val * S_val
            - alpha * k_bar * S_val
            * (S_val + H_val + R_val)
            - delta * S_val
            + xi * H_val
            + eta * k_bar * H_val * S_val
            + zeta * k_bar * H_val * S_delay
            + gamma_t * H_val
        )

        dH = (
            delta * S_val
            - xi * H_val
            - eta * k_bar * H_val * S_val
            - zeta * k_bar * H_val * S_delay
            - gamma_t * H_val
        )

        dR = (
            beta * k_bar * I_val * S_val
            + alpha * k_bar * S_val
            * (S_val + H_val + R_val)
        )

        return dI, dS, dH, dR

    # ========================================================
    # RK4 METHOD
    # ========================================================

    for j in range(n - 1):

        t = time[j]

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        S_delay_1 = delayed_S(t, j)

        k1_I, k1_S, k1_H, k1_R = derivatives(
            t,
            I[j],
            S[j],
            H[j],
            R[j],
            S_delay_1
        )

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        t2 = t + h / 2

        S_delay_2 = delayed_S(t2, j)

        I2 = I[j] + (h / 2) * k1_I
        S2 = S[j] + (h / 2) * k1_S
        H2 = H[j] + (h / 2) * k1_H
        R2 = R[j] + (h / 2) * k1_R

        k2_I, k2_S, k2_H, k2_R = derivatives(
            t2,
            I2,
            S2,
            H2,
            R2,
            S_delay_2
        )

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        t3 = t + h / 2

        S_delay_3 = delayed_S(t3, j)

        I3 = I[j] + (h / 2) * k2_I
        S3 = S[j] + (h / 2) * k2_S
        H3 = H[j] + (h / 2) * k2_H
        R3 = R[j] + (h / 2) * k2_R

        k3_I, k3_S, k3_H, k3_R = derivatives(
            t3,
            I3,
            S3,
            H3,
            R3,
            S_delay_3
        )

        # ----------------------------------------------------
        # Stage 4
        # ----------------------------------------------------

        t4 = t + h

        S_delay_4 = delayed_S(t4, j)

        I4 = I[j] + h * k3_I
        S4 = S[j] + h * k3_S
        H4 = H[j] + h * k3_H
        R4 = R[j] + h * k3_R

        k4_I, k4_S, k4_H, k4_R = derivatives(
            t4,
            I4,
            S4,
            H4,
            R4,
            S_delay_4
        )

        # ----------------------------------------------------
        # RK4 UPDATE
        # ----------------------------------------------------

        I[j + 1] = I[j] + (h / 6) * (
            k1_I + 2*k2_I + 2*k3_I + k4_I
        )

        S[j + 1] = S[j] + (h / 6) * (
            k1_S + 2*k2_S + 2*k3_S + k4_S
        )

        H[j + 1] = H[j] + (h / 6) * (
            k1_H + 2*k2_H + 2*k3_H + k4_H
        )

        R[j + 1] = R[j] + (h / 6) * (
            k1_R + 2*k2_R + 2*k3_R + k4_R
        )

    return I, S, H, R


# ============================================================
# RUN ALL TAU VALUES
# ============================================================

results_tau = {}

for tau_value in tau_values:

    I, S, H, R = simulate_tau(tau_value)

    results_tau[tau_value] = {
        "I": I,
        "S": S,
        "H": H,
        "R": R
    }


# ============================================================
# 2 × 2 FIGURE
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))


# ============================================================
# TOP LEFT: GAMMA — SPREADERS
# ============================================================

for gamma_value in gamma_values:

    axes[0, 0].plot(
        time,
        results_gamma[gamma_value]["S"],
        label=f"$\\gamma_e$ = {gamma_value}"
    )

axes[0, 0].axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

axes[0, 0].set_xlabel("Time (days)")
axes[0, 0].set_ylabel("Spreaders, S(t)")
axes[0, 0].set_title(
    "Effect of External Reactivation ($\\gamma(t)$) on Spreaders"
)
axes[0, 0].legend()
axes[0, 0].grid(True)


# ============================================================
# TOP RIGHT: GAMMA — HIBERNATORS
# ============================================================

for gamma_value in gamma_values:

    axes[0, 1].plot(
        time,
        results_gamma[gamma_value]["H"],
        label=f"$\\gamma_e$ = {gamma_value}"
    )

axes[0, 1].axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

axes[0, 1].set_xlabel("Time (days)")
axes[0, 1].set_ylabel("Hibernators, H(t)")
axes[0, 1].set_title(
    "Effect of External Reactivation ($\\gamma(t)$) on Hibernators"
)
axes[0, 1].legend()
axes[0, 1].grid(True)


# ============================================================
# BOTTOM LEFT: TAU — SPREADERS
# ============================================================

for tau_value in tau_values:

    axes[1, 0].plot(
        time,
        results_tau[tau_value]["S"],
        label=f"$\\tau$ = {tau_value}"
    )

axes[1, 0].axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

axes[1, 0].set_xlabel("Time (days)")
axes[1, 0].set_ylabel("Spreaders, S(t)")
axes[1, 0].set_title(
    "Effect of Delay ($\\tau$) on Spreaders"
)
axes[1, 0].legend()
axes[1, 0].grid(True)


# ============================================================
# BOTTOM RIGHT: TAU — HIBERNATORS
# ============================================================

for tau_value in tau_values:

    axes[1, 1].plot(
        time,
        results_tau[tau_value]["H"],
        label=f"$\\tau$ = {tau_value}"
    )

axes[1, 1].axvline(
    t_event,
    linestyle="--",
    label="External event onset"
)

axes[1, 1].set_xlabel("Time (days)")
axes[1, 1].set_ylabel("Hibernators, H(t)")
axes[1, 1].set_title(
    "Effect of Delay ($\\tau$) on Hibernators"
)
axes[1, 1].legend()
axes[1, 1].grid(True)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# EXPORT AS ONE FIGURE
# ============================================================

plt.savefig(
    "External_Reactivation_and_Delay_2x2.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "External_Reactivation_and_Delay_2x2.pdf",
    bbox_inches="tight"
)

plt.show()


# In[9]:


import numpy as np

# ============================================================
# PARAMETERS
# ============================================================

lambda_ = 0.8
alpha = 0.3
beta = 0.2
delta = 0.6
xi = 0.5
eta = 0.5
zeta = 0.3

k_bar = 10.0
tau = 2.0

gamma_e = 0.1
t_event = 2.0

# ============================================================
# INITIAL CONDITIONS
# ============================================================

N = 1000000

I0 = (N - 1) / N
S0 = 1 / N
H0 = 0.0
R0 = 0.0


# ============================================================
# EXTERNAL EVENT
# ============================================================

def gamma_function(t):
    if t < t_event:
        return 0.0
    else:
        return gamma_e


# ============================================================
# RK4 SIMULATION FUNCTION
# ============================================================

def run_simulation(h):

    T = 10.0

    time = np.arange(0, T + h, h)
    n = len(time)

    I = np.zeros(n)
    S = np.zeros(n)
    H = np.zeros(n)
    R = np.zeros(n)

    I[0] = I0
    S[0] = S0
    H[0] = H0
    R[0] = R0

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    def history_S(t):
        return S0

    # --------------------------------------------------------
    # DELAYED S(t - tau)
    # --------------------------------------------------------

    def delayed_S(t, current_index):

        delayed_time = t - tau

        # History region
        if delayed_time <= 0:
            return history_S(delayed_time)

        # Previously computed solution
        return np.interp(
            delayed_time,
            time[:current_index + 1],
            S[:current_index + 1]
        )

    # --------------------------------------------------------
    # RIGHT-HAND SIDE
    # --------------------------------------------------------

    def derivatives(t, I_val, S_val, H_val, R_val, S_delay):

        gamma_t = gamma_function(t)

        dI = -(lambda_ + beta) * k_bar * I_val * S_val

        dS = (
            lambda_ * k_bar * I_val * S_val
            - alpha * k_bar * S_val * (S_val + H_val + R_val)
            - delta * S_val
            + xi * H_val
            + eta * k_bar * H_val * S_val
            + zeta * k_bar * H_val * S_delay
            + gamma_t * H_val
        )

        dH = (
            delta * S_val
            - xi * H_val
            - eta * k_bar * H_val * S_val
            - zeta * k_bar * H_val * S_delay
            - gamma_t * H_val
        )

        dR = (
            beta * k_bar * I_val * S_val
            + alpha * k_bar * S_val * (S_val + H_val + R_val)
        )

        return dI, dS, dH, dR

    # ========================================================
    # RK4 LOOP
    # ========================================================

    for j in range(n - 1):

        t = time[j]

        # ---------------- Stage 1 ----------------
        S_delay_1 = delayed_S(t, j)

        k1_I, k1_S, k1_H, k1_R = derivatives(
            t, I[j], S[j], H[j], R[j], S_delay_1
        )

        # ---------------- Stage 2 ----------------
        t2 = t + h / 2

        S_delay_2 = delayed_S(t2, j)

        I2 = I[j] + (h / 2) * k1_I
        S2 = S[j] + (h / 2) * k1_S
        H2 = H[j] + (h / 2) * k1_H
        R2 = R[j] + (h / 2) * k1_R

        k2_I, k2_S, k2_H, k2_R = derivatives(
            t2, I2, S2, H2, R2, S_delay_2
        )

        # ---------------- Stage 3 ----------------
        t3 = t + h / 2

        S_delay_3 = delayed_S(t3, j)

        I3 = I[j] + (h / 2) * k2_I
        S3 = S[j] + (h / 2) * k2_S
        H3 = H[j] + (h / 2) * k2_H
        R3 = R[j] + (h / 2) * k2_R

        k3_I, k3_S, k3_H, k3_R = derivatives(
            t3, I3, S3, H3, R3, S_delay_3
        )

        # ---------------- Stage 4 ----------------
        t4 = t + h

        S_delay_4 = delayed_S(t4, j)

        I4 = I[j] + h * k3_I
        S4 = S[j] + h * k3_S
        H4 = H[j] + h * k3_H
        R4 = R[j] + h * k3_R

        k4_I, k4_S, k4_H, k4_R = derivatives(
            t4, I4, S4, H4, R4, S_delay_4
        )

        # ---------------- RK4 update ----------------

        I[j + 1] = I[j] + (h / 6) * (
            k1_I + 2*k2_I + 2*k3_I + k4_I
        )

        S[j + 1] = S[j] + (h / 6) * (
            k1_S + 2*k2_S + 2*k3_S + k4_S
        )

        H[j + 1] = H[j] + (h / 6) * (
            k1_H + 2*k2_H + 2*k3_H + k4_H
        )

        R[j + 1] = R[j] + (h / 6) * (
            k1_R + 2*k2_R + 2*k3_R + k4_R
        )

    return time, I, S, H, R


# ============================================================
# CONVERGENCE TEST
# ============================================================

h_values = [0.02, 0.01, 0.005]

results = {}

for h in h_values:

    time, I, S, H, R = run_simulation(h)

    results[h] = {
        "time": time,
        "I": I,
        "S": S,
        "H": H,
        "R": R
    }


# ============================================================
# EXTRACT VALUES AT t = 3
# ============================================================

print("\nRK4 STEP-SIZE CONVERGENCE")
print("=" * 70)

print(
    f"{'h':<10}"
    f"{'S(3)':<18}"
    f"{'H(3)':<18}"
    f"{'Max S':<18}"
    f"{'t at Max S':<15}"
)

for h in h_values:

    time = results[h]["time"]
    S = results[h]["S"]
    H = results[h]["H"]

    index_3 = np.argmin(np.abs(time - 3.0))

    S3 = S[index_3]
    H3 = H[index_3]

    max_index = np.argmax(S)
    S_max = S[max_index]
    t_max = time[max_index]

    print(
        f"{h:<10.3f}"
        f"{S3:<18.10f}"
        f"{H3:<18.10f}"
        f"{S_max:<18.10f}"
        f"{t_max:<15.4f}"
    )


# ============================================================
# COMPARISON: h = 0.01 AGAINST h = 0.005
# ============================================================

S_01 = results[0.01]["S"]
H_01 = results[0.01]["H"]

S_005 = results[0.005]["S"]
H_005 = results[0.005]["H"]

# Since the meshes differ, compare only at common times.
# Every 0.01-day point exists in the h=0.005 solution.

S_005_common = S_005[::2]
H_005_common = H_005[::2]

abs_diff_S = np.max(np.abs(S_01 - S_005_common))
abs_diff_H = np.max(np.abs(H_01 - H_005_common))

rel_diff_S = abs_diff_S / np.max(np.abs(S_005_common))
rel_diff_H = abs_diff_H / np.max(np.abs(H_005_common))

print("\n")
print("Comparison: h = 0.01 versus h = 0.005")
print("=" * 70)

print("Maximum absolute difference in S =", abs_diff_S)
print("Maximum absolute difference in H =", abs_diff_H)

print("Maximum relative difference in S =", rel_diff_S)
print("Maximum relative difference in H =", rel_diff_H)


# ============================================================
# CONSERVATION CHECK
# ============================================================

print("\n")
print("Maximum conservation error")
print("=" * 70)

for h in h_values:

    I = results[h]["I"]
    S = results[h]["S"]
    H = results[h]["H"]
    R = results[h]["R"]

    total = I + S + H + R

    conservation_error = np.max(
        np.abs(total - 1.0)
    )

    print(
        f"h = {h:.3f} : "
        f"{conservation_error:.3e}"
    )


# In[ ]:




