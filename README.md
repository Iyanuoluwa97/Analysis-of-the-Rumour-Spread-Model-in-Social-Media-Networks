# Analysis-of-a-delayed-SIHR-rumour-spread-model
This repository contains the numerical implementation and analysis of a delayed SIHR rumour-spreading model in social media networks. It includes Python and Maple codes, numerical simulations, parameter sensitivity analysis, and comparisons of the Differential Transform Method and fourth-order Runge–Kutta method.

# Analysis of the Rumour Spread Model in Social Media Networks

## About the Project

This repository contains the numerical implementation of a delayed SIHR (Ignorant–Spreader–Hibernator–Stifler) model developed to investigate rumour spreading and reactivation in social media networks.

The study extends the classical rumour-spreading framework by incorporating delayed remembering and external reactivation. It examines how individuals who temporarily stop spreading a rumour may become active again through different remembering mechanisms or external influences.

The main objective of the computational work is to investigate the dynamics of the model, examine the influence of selected parameters, and compare numerical solutions obtained using two different methods.

## Model Description

The model consists of four compartments:

- Ignorants (I): Individuals who have not yet received or become aware of the rumour.
- Spreaders (S): Individuals who are actively spreading the rumour.
- Hibernators (H): Individuals who have temporarily stopped spreading the rumour but may become active again.
- Stiflers (R): Individuals who are no longer spreading or interested in spreading the rumour.

The model incorporates spontaneous remembering, interaction-based remembering, delayed remembering and external reactivation. These mechanisms allow the study of how rumour activity may persist or re-emerge over time.

The delayed remembering process is represented by a delay term involving the past spreader population, while external reactivation accounts for the influence of events or interventions that may encourage previously inactive individuals to spread the rumour again.

## Numerical Methods

Two numerical approaches are implemented:

1. **Fourth-Order Runge–Kutta (RK4) Method:** Implemented in Python to obtain numerical solutions of the delayed SIHR system using the method of steps.
2. **Differential Transform Method (DTM):** Implemented in Maple using local Taylor expansions to approximate the solutions of the model.

The numerical solutions obtained using both methods are compared to examine their agreement.

## Parameter Analysis

The computational experiments investigate the influence of key model parameters, particularly:

- Delayed remembering rate (\(\zeta\)).
- External reactivation strength (\(\gamma\)).
- Delay duration (\(\tau\)).

The simulations examine changes in the compartment populations and the persistence of rumour spreading under different parameter settings.

## Software Requirements

- Python 3.11 or later
- NumPy
- SciPy
- Matplotlib
- Maple (for the DTM implementation)

## Running the Codes

The Python implementation can be run using a Python environment with the required packages installed.

The Maple implementation can be opened and executed using Maple. The source code is provided in the Maple directory.

## Research and Reproducibility

The codes are provided to support the computational study of the delayed SIHR rumour-spreading model and allow interested researchers to explore its numerical behaviour.

The associated research manuscript is currently unpublished. The manuscript itself is therefore not included in this repository.

## Author and Acknowledgement

This repository accompanies research on the mathematical modelling and numerical analysis of rumour spreading in social media networks.

Further details about the associated research will be provided when the manuscript becomes publicly available.
