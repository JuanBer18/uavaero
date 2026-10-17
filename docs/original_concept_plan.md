> **Historical concept document — preserved unchanged below.**
>
> This was the original broad proposal. It is retained for its research, equations, solver rationale,
> validation ideas, and feature catalogue. It is **not** the V2 implementation authority and must not
> reintroduce sizing, requirements, mission, or the obsolete phase order. The authoritative plan is
> [../vspaero_uav_aero_dynamics_tool_plan.md](../vspaero_uav_aero_dynamics_tool_plan.md).

# VSPAERO-Based UAV Aerodynamics & Flight-Dynamics Design Tool

## 1. Project Goal

Build a custom UAV preliminary-design and optimisation application with:

- **OpenVSP** as the parametric aircraft geometry engine.
- **VSPAERO** as the main 3D aerodynamic solver.
- **XFoil-derived sectional viscous data** as an additional low-Reynolds-number correction method.
- **OpenVSP Parasite Drag** as another independent drag-estimation method.
- **VSPAERO's own viscous/drag corrections** retained as a comparison method.
- **Flight-dynamics post-processing** built around VSPAERO aerodynamic and stability derivatives.
- **PySide6 / Qt** as a clean, purpose-built interface.
- Automated design-space exploration and optimisation instead of manually rebuilding wings and rerunning analyses.

The objective is **not** to reproduce the OpenVSP GUI.

The objective is to create a much more focused UAV design workflow where the engineer changes a small number of meaningful design variables and the software automatically:

1. Builds the aircraft.
2. Runs aerodynamic analyses.
3. Calculates multiple drag estimates.
4. Evaluates stability and dynamic modes.
5. Compares candidate designs.
6. Optimises the aircraft against mission requirements.

---

# 2. Why Use OpenVSP/VSPAERO as the Main Backend?

We compared the XFLR5/Flow5 family with OpenVSP/VSPAERO.

## Flow5 strengths

Flow5 is particularly attractive for:

- Low-Reynolds-number aircraft.
- Integrated XFoil viscous airfoil analysis.
- Sectional airfoil polar interpolation.
- Wing profile-drag estimation.
- Good stability visualisation.
- Root-locus plots.
- Eigenmode visualisation.
- Time-response plots.
- Clean aerodynamic visualisations.

Its integrated XFoil treatment is particularly useful for small UAVs because airfoil drag, transition and separation behaviour can change significantly with Reynolds number.

## OpenVSP/VSPAERO strengths

OpenVSP/VSPAERO is a better base for this custom application because it has:

- A mature Python API.
- Headless operation.
- Parametric geometry generation.
- Whole-aircraft geometry.
- Wing, tail, fuselage and boom modelling.
- Control surfaces.
- Propeller/rotor support.
- Aircraft interference effects.
- Stability derivatives.
- Unsteady P/Q/R analyses.
- Automation support.
- Multiple simultaneous OpenVSP instances.
- Strong support for design studies and optimisation.
- Better long-term flexibility for unconventional UAV configurations.

Therefore the chosen direction is:

> **Use OpenVSP/VSPAERO as the single main aircraft backend and add the missing low-Re viscous treatment around it.**

Flow5 should be treated as a useful reference/validation tool rather than the primary backend.

---

# 3. Scope

## Included

### Aircraft design

- Wing geometry.
- Horizontal tail.
- Vertical tail.
- Fuselage / booms where required.
- Control surfaces.
- Propellers if needed.
- CG position.
- Mass and inertia properties.

### Aerodynamics

- Lift.
- Drag.
- Pitching moment.
- Side force.
- Rolling moment.
- Yawing moment.
- Spanwise loading.
- Pressure distribution.
- Induced drag.
- Profile drag.
- Parasite drag.
- Reynolds-number effects.
- Airfoil-specific viscous effects.
- Control-surface derivatives.
- Propeller-airframe interaction where applicable.

### Stability and dynamics

- Static margin.
- Neutral point.
- Longitudinal static stability.
- Directional static stability.
- Lateral static stability.
- Aerodynamic stability derivatives.
- Dynamic derivatives.
- Short-period mode.
- Phugoid mode.
- Dutch-roll mode.
- Roll subsidence.
- Spiral mode.
- Damping ratios.
- Natural frequencies.
- Time constants.
- Mode time responses.
- Root-locus style visualisation.
- Optional 3D mode animation.

### Optimisation

- Parametric geometry.
- Automated candidate generation.
- Batch VSPAERO execution.
- Parallel candidate evaluation.
- Single-objective optimisation.
- Multi-objective optimisation.
- Engineering constraints.
- Pareto-front comparison.
- Coarse-to-fine fidelity levels.

---

# 4. Explicitly Out of Scope

The following should **not** be part of this project at present:

- Structural FEA.
- Rib/spar optimisation.
- CalculiX.
- NASTRAN.
- Structural stress analysis.
- Aeroelasticity.
- Full Navier-Stokes CFD solver development.
- Replacing VSPAERO with a custom CFD solver.

CFD may later be used only as an **external validation reference** for selected final designs.

---

# 5. Overall Software Architecture

```text
                           CUSTOM UAV DESIGNER
                                  |
                                  |
                             PySide6 / Qt
                                  |
                    Aircraft Design Data Model
                                  |
             +--------------------+--------------------+
             |                                         |
             |                                         |
        OpenVSP Geometry                           Mission Model
             |                                         |
             +--------------------+--------------------+
                                  |
                                  v
                               VSPAERO
                                  |
                    3D aerodynamic solution
                                  |
           +----------------------+----------------------+
           |                      |                      |
           v                      v                      v
     Aerodynamic loads       Stability data       Spanwise loads
           |                      |                      |
           |                      |                      v
           |                      |                XFoil polar DB
           |                      |                      |
           |                      |                 viscous CD
           |                      |                      |
           +----------+-----------+----------------------+
                      |
                      v
                Aero Postprocessor
                      |
       +--------------+----------------+
       |                               |
       v                               v
 Multiple drag models           Flight dynamics
       |                               |
       v                               v
 Performance metrics       Eigenvalues / modes / response
       |                               |
       +---------------+---------------+
                       |
                       v
                   Optimiser
                       |
                       v
                Candidate database
                       |
                       v
                  Results / plots
```

---

# 6. Core Design Philosophy

The GUI should expose **engineering design variables**, not raw OpenVSP internals.

Bad UI:

```text
SectTess_U
X_Rel_Location
XSecCurve
Group_4832
SectTess_W
```

Preferred UI:

```text
Wing span
Aspect ratio
Taper ratio
Sweep
Dihedral
Tip twist
Root airfoil
Tip airfoil
Wing position
```

The software should internally convert engineering variables into OpenVSP parameters.

---

# 7. Aircraft Parameterisation

The optimiser should manipulate a limited number of independent variables.

## Example wing variables

- Span, \(b\)
- Aspect ratio, \(AR\)
- Taper ratio, \(\lambda\)
- Sweep, \(\Lambda\)
- Dihedral
- Tip twist
- Root airfoil
- Tip airfoil
- Wing longitudinal position

Derived quantities should be calculated automatically.

For example:

\[
S = \frac{b^2}{AR}
\]

For a simple tapered wing:

\[
c_{root} =
\frac{2S}{b(1+\lambda)}
\]

\[
c_{tip} = \lambda c_{root}
\]

Do not ask the optimiser to independently change span, area, root chord, tip chord, aspect ratio and taper simultaneously when several of those variables are mathematically dependent.

## Example tail variables

- Horizontal-tail area.
- Horizontal-tail aspect ratio.
- Tail arm.
- Horizontal-tail incidence.
- Horizontal-tail position.
- Vertical-tail area.
- Vertical-tail aspect ratio.
- Vertical-tail position.

## Example control variables

- Elevator chord fraction.
- Rudder chord fraction.
- Aileron span fraction.
- Aileron chord fraction.
- Control deflection limits.

---

# 8. Aerodynamic Analysis Pipeline

## Step 1 — Generate geometry

The aircraft model is created or modified through the OpenVSP Python API.

The user should not need to manually edit the OpenVSP model.

## Step 2 — Generate VSPAERO analysis geometry

Automatically:

- Update geometry.
- Generate analysis mesh.
- Set aerodynamic analysis parameters.
- Configure reference quantities.
- Set Mach number.
- Set Reynolds-relevant atmosphere information where required.
- Set angle-of-attack sweep.
- Set sideslip.
- Set control deflections.
- Run VSPAERO.

## Step 3 — Collect main results

Store:

- \(C_L\)
- \(C_D\)
- \(C_{D_i}\)
- \(C_m\)
- \(C_l\)
- \(C_n\)
- \(C_Y\)
- L/D
- spanwise lift distribution
- sectional lift coefficients
- local chord
- surface pressures
- aerodynamic derivatives

---

# 9. Multiple Drag / Viscosity Models

A major feature of the program should be to **retain several independent viscous/drag approximations instead of forcing a single result**.

This allows later comparison against:

- CFD.
- Wind-tunnel measurements.
- Flight-test results.

The software should present each model separately.

---

## Drag Model A — VSPAERO Native

Keep VSPAERO's own viscous / empirical drag treatment.

Example output:

```text
VSPAERO total CD
VSPAERO induced CD
VSPAERO viscous contribution
```

This provides the baseline result.

---

## Drag Model B — OpenVSP Parasite Drag

Run the OpenVSP Parasite Drag analysis.

Useful for:

- Fuselage.
- Booms.
- Nacelles.
- Landing-gear approximations if relevant.
- Miscellaneous wetted bodies.
- Empirical wing drag estimate as an independent comparison.

Keep its wing result even if another wing model is believed to be better.

The point is comparison.

---

## Drag Model C — VSPAERO + XFoil Sectional Viscous Drag

This should be the principal low-Reynolds-number correction method.

VSPAERO provides the 3D wing loading.

For each spanwise strip:

1. Obtain local chord \(c_i\).
2. Obtain local sectional lift coefficient \(C_{l,i}\).
3. Calculate local Reynolds number:

\[
Re_i =
\frac{\rho V c_i}{\mu}
\]

4. Determine the local airfoil.
5. Query the XFoil polar database at:

\[
C_d =
f(C_l,Re,\text{airfoil})
\]

6. Integrate profile drag across the wing.

Approximate wing profile drag:

\[
C_{D,profile}
=
\frac{1}{S_{ref}}
\sum_i C_{d,i}\Delta S_i
\]

Then construct:

\[
C_{D,total}^{XFoil}
=
C_{D_i}^{VSPAERO}
+
C_{D,profile}^{XFoil}
+
C_{D,body}^{Parasite}
+
C_{D,other}
\]

This gives a more airfoil-specific low-Reynolds-number result.

---

# 10. XFoil Polar Database

XFoil should **not** necessarily run during every optimisation iteration.

Instead, generate and cache polar databases.

Example Reynolds-number grid:

```text
50,000
75,000
100,000
125,000
150,000
200,000
250,000
300,000
400,000
500,000
750,000
1,000,000
```

For each airfoil store:

- \(C_l(\alpha)\)
- \(C_d(\alpha)\)
- \(C_m(\alpha)\)
- transition data if required
- \(C_{l,max}\)
- stall-related limits
- convergence status

Equivalent interpolation surface:

\[
C_d = f(C_l, Re)
\]

Optional future dimensions:

\[
C_d = f(C_l, Re, N_{crit}, x_{trip})
\]

The optimiser should query this database rather than repeatedly rerunning XFoil.

---

# 11. Viscosity Model Comparison UI

The results page should make model disagreement visible.

Example:

```text
AERODYNAMIC DRAG COMPARISON
-------------------------------------------------

                    CD       L/D
VSPAERO native     0.0341    15.2
Parasite model     0.0378    13.8
XFoil corrected    0.0356    14.6

Spread             0.0037
Relative spread    10.4 %
```

Plots should allow:

- \(C_D\) vs \(C_L\)
- L/D vs speed
- drag breakdown vs speed
- each model on the same graph
- model difference vs Reynolds number

Once CFD or experimental data exists, add:

```text
CFD
Wind tunnel
Flight test
```

to the same comparison.

---

# 12. Model Calibration

Later validation should not overwrite older models.

Instead, maintain:

```text
VSPAERO native
OpenVSP parasite
XFoil corrected
CFD
Experiment
```

Then calculate errors such as:

\[
\epsilon_{CD}
=
\frac{C_{D,predicted}-C_{D,reference}}
{C_{D,reference}}
\]

The tool can eventually identify which model is more reliable for:

- cruise.
- high lift.
- different Reynolds-number regimes.
- different UAV configurations.

---

# 13. Stall Handling

XFoil sectional data can be used to identify likely local stall.

For each strip:

\[
C_{l,i}
\]

is compared with:

\[
C_{l,max}(Re_i)
\]

The initial implementation should produce a warning such as:

```text
TIP SECTION STALL LIMIT EXCEEDED
Station: 82 % semispan
Local Cl: 1.28
XFoil Clmax: 1.13
Margin: -0.15
```

Do **not** initially attempt a fully coupled nonlinear stall solver.

The first version should use this as:

- a design constraint.
- a stall warning.
- an optimisation penalty.

A true iterative viscous/inviscid stall coupling can be investigated later.

---

# 14. Aerodynamic Analysis Modes

The program should support several analysis levels.

## Quick analysis

For optimisation screening:

- coarse mesh.
- one or a few angles of attack.
- minimal output.
- no expensive visualisation.

## Standard analysis

For normal design work:

- normal VSPAERO mesh.
- alpha sweep.
- drag comparison.
- stability derivatives.
- span loading.

## High-fidelity VSPAERO analysis

For finalists:

- refined mesh.
- detailed alpha sweep.
- more wake iterations.
- unsteady derivatives where required.
- detailed visualisation.

---

# 15. Flight Stability

VSPAERO should provide the aerodynamic derivatives.

Important examples include:

### Longitudinal

\[
C_{L_\alpha}
\]

\[
C_{m_\alpha}
\]

\[
C_{m_q}
\]

\[
C_{m_{\delta_e}}
\]

### Lateral-directional

\[
C_{Y_\beta}
\]

\[
C_{l_\beta}
\]

\[
C_{n_\beta}
\]

\[
C_{l_p}
\]

\[
C_{n_r}
\]

\[
C_{l_{\delta_a}}
\]

\[
C_{n_{\delta_r}}
\]

These should then be converted into dimensional stability derivatives.

---

# 16. Static-Stability Outputs

The software should calculate and display:

- Neutral point.
- CG.
- Static margin.
- \(C_{m_\alpha}\)
- \(C_{n_\beta}\)
- \(C_{l_\beta}\)
- Elevator effectiveness.
- Rudder effectiveness.
- Aileron effectiveness.

Example:

```text
STATIC STABILITY

CG                    31.2 % MAC
Neutral point         42.8 % MAC
Static margin         11.6 %

Longitudinal          STABLE
Directional           STABLE
Lateral               STABLE
```

---

# 17. Flight-Dynamics Model

VSPAERO should provide the aerodynamic derivatives.

The custom program should perform the actual linear flight-dynamics post-processing.

## Longitudinal state-space model

Typical state vector:

\[
x_{long}
=
\begin{bmatrix}
u & w & q & \theta
\end{bmatrix}^{T}
\]

with:

\[
\dot{x}=Ax+Bu
\]

## Lateral-directional state-space model

Typical state vector:

\[
x_{lat}
=
\begin{bmatrix}
v & p & r & \phi
\end{bmatrix}^{T}
\]

The exact formulation may later include:

- \(\alpha\)
- \(\beta\)
- altitude
- heading

depending on the chosen flight-dynamics convention.

---

# 18. Dynamic Modes

Use:

```python
numpy.linalg.eig(A)
```

to obtain:

\[
Av=\lambda v
\]

where:

\[
\lambda = \sigma \pm i\omega_d
\]

From each eigenvalue calculate:

\[
\omega_n =
\sqrt{\sigma^2+\omega_d^2}
\]

\[
\zeta =
-\frac{\sigma}{\omega_n}
\]

\[
T =
\frac{2\pi}{\omega_d}
\]

where appropriate.

Automatically classify:

### Longitudinal

- Short period.
- Phugoid.

### Lateral-directional

- Dutch roll.
- Roll subsidence.
- Spiral.

---

# 19. Dynamics Results UI

Example:

```text
DYNAMIC STABILITY
-------------------------------------------------

LONGITUDINAL

Short Period
Status              STABLE
Damping ratio       0.61
Natural frequency   5.82 rad/s
Settling time       1.4 s

Phugoid
Status              STABLE
Damping ratio       0.07
Period              14.6 s


LATERAL-DIRECTIONAL

Dutch Roll
Status              STABLE
Damping ratio       0.28
Period              2.8 s

Roll Subsidence
Status              STABLE
Time constant       0.42 s

Spiral
Status              UNSTABLE
Doubling time       36.2 s
```

---

# 20. Root-Locus / Eigenvalue Visualisation

Flow5 is a useful inspiration here.

Provide an eigenvalue plot:

```text
Imaginary
   ^
   |
   |      x Dutch roll
   |
---+------------------------------> Real
   |
   | x short period
   |
```

Features:

- Hover mode identification.
- Stable/unstable region.
- Click an eigenvalue to inspect the mode.
- Display damping ratio.
- Display natural frequency.
- Display dominant state components.

---

# 21. Time-Response Simulation

For a selected mode or control input, integrate:

\[
\dot{x}=Ax+Bu
\]

using SciPy.

Examples:

- Elevator step.
- Aileron step.
- Rudder step.
- Initial pitch disturbance.
- Initial sideslip.
- Initial roll disturbance.
- Eigenmode excitation.

Plots could include:

- pitch angle.
- roll angle.
- yaw rate.
- pitch rate.
- angle of attack.
- sideslip.
- velocity perturbation.
- altitude response.

---

# 22. Optional 3D Flight-Mode Animation

This is not required for the first version but is desirable later.

Use the eigenvector for each mode to animate:

- aircraft translation.
- roll.
- pitch.
- yaw.

Examples:

- Phugoid.
- Short period.
- Dutch roll.
- Roll subsidence.

This could provide the same strong intuitive value as Flow5's stability animation while remaining inside the custom tool.

---

# 23. VSPAERO Unsteady Analysis

Later versions can expose VSPAERO's:

- P analysis.
- Q analysis.
- R analysis.
- Time-accurate analysis.
- Moving-body analysis.
- Propeller / rotor unsteady effects.

These analyses may provide improved dynamic derivatives and unsteady aerodynamic information.

They should not be required for every optimisation candidate because they are more expensive.

---

# 24. Optimisation Philosophy

The optimiser should not blindly manipulate the entire OpenVSP parameter set.

Use approximately **8–20 meaningful design variables**.

Example vector:

\[
x =
[
b,\,
AR,\,
\lambda,\,
\Lambda,\,
\theta_{tip},\,
S_H,\,
l_H,\,
S_V
]
\]

---

# 25. Example Optimisation Objectives

Possible single-objective problems:

### Maximum aerodynamic efficiency

\[
\max \frac{L}{D}
\]

### Minimum cruise drag

\[
\min D_{cruise}
\]

### Minimum cruise power

\[
\min P_{cruise}
\]

### Maximum endurance

\[
\max t_{endurance}
\]

### Maximum range

\[
\max R
\]

---

# 26. Multi-Objective Optimisation

Useful combinations include:

\[
\max L/D
\]

while:

\[
\max \zeta_{DutchRoll}
\]

or:

\[
\min P_{cruise}
\]

while:

\[
\max SM
\]

A Pareto front should be produced instead of pretending there is always one universally optimal design.

---

# 27. Example Constraints

Possible constraints:

```text
Maximum span
Maximum wing area
Minimum tip chord
Minimum static margin
Maximum static margin
Maximum stall speed
Minimum Clmax margin
Minimum Dutch-roll damping
Minimum short-period damping
Maximum spiral divergence
Maximum control deflection
Maximum pitching moment
Manufacturing limits
```

Example:

\[
b \leq 1.5 \text{ m}
\]

\[
8\% \leq SM \leq 15\%
\]

\[
V_{stall} \leq 10 \text{ m/s}
\]

\[
C_{l,local} < 0.9 C_{l,max}
\]

---

# 28. Optimisation Algorithms

## Initial implementation

Use SciPy:

- differential evolution.
- SLSQP where appropriate.
- least-squares methods for trim problems.

Differential evolution is attractive for aerodynamic geometry because the design space can be nonlinear and contain solver failures.

## Later implementation

Use `pymoo` for:

- NSGA-II.
- NSGA-III.
- multi-objective optimisation.
- Pareto-front generation.

Optuna can also be considered for design-space exploration.

---

# 29. Parallel Optimisation

OpenVSP's multi-instance/facade capability should eventually be used to evaluate several aircraft simultaneously.

Concept:

```text
                    OPTIMISER
                        |
       +----------------+----------------+
       |                |                |
       v                v                v
   Candidate A      Candidate B      Candidate C
       |                |                |
     VSP #1           VSP #2           VSP #3
       |                |                |
   VSPAERO A        VSPAERO B        VSPAERO C
       |                |                |
       +----------------+----------------+
                        |
                        v
                    RESULTS
```

The number of simultaneous workers should be configurable based on CPU and memory.

---

# 30. Coarse-to-Fine Optimisation

Do not run expensive analyses for every candidate.

Recommended pipeline:

```text
Stage 0
Analytical geometry / sanity checks
        |
        v
Stage 1
Very coarse VSPAERO
        |
        v
Stage 2
Normal VSPAERO
        |
        v
Stage 3
XFoil-corrected drag + stability
        |
        v
Stage 4
Unsteady / refined VSPAERO
        |
        v
Finalists
External CFD / experiment
```

Example:

```text
10,000 generated designs
    ↓
2,000 valid geometries
    ↓
500 coarse VSPAERO candidates
    ↓
50 standard candidates
    ↓
10 detailed candidates
    ↓
3 finalists
```

---

# 31. Caching

Every simulation should be uniquely identified from:

- geometry.
- airfoil.
- atmosphere.
- speed.
- angle of attack.
- control state.
- VSPAERO settings.
- mesh settings.

Before running a simulation:

```text
if result_exists(hash):
    load result
else:
    run solver
```

This avoids wasting time rerunning identical designs.

SQLite is suitable for metadata/results.

Large arrays can be stored separately as:

- NumPy.
- HDF5.
- Parquet.

---

# 32. Proposed Python Project Structure

```text
uav_designer/
|
+-- main.py
|
+-- gui/
|   +-- main_window.py
|   +-- aircraft_panel.py
|   +-- wing_panel.py
|   +-- tail_panel.py
|   +-- mission_panel.py
|   +-- aero_panel.py
|   +-- dynamics_panel.py
|   +-- optimisation_panel.py
|   +-- results_panel.py
|
+-- aircraft/
|   +-- aircraft.py
|   +-- wing.py
|   +-- tail.py
|   +-- controls.py
|   +-- mass_properties.py
|   +-- mission.py
|
+-- geometry/
|   +-- openvsp_model.py
|   +-- parameter_mapping.py
|
+-- solvers/
|   +-- solver_base.py
|   +-- vspaero_solver.py
|   +-- parasite_drag.py
|   +-- xfoil_runner.py
|
+-- aero/
|   +-- polar_database.py
|   +-- viscous_correction.py
|   +-- drag_models.py
|   +-- spanload.py
|   +-- performance.py
|
+-- dynamics/
|   +-- derivatives.py
|   +-- dimensional_derivatives.py
|   +-- longitudinal.py
|   +-- lateral.py
|   +-- eigenmodes.py
|   +-- time_response.py
|
+-- optimisation/
|   +-- design_variables.py
|   +-- objectives.py
|   +-- constraints.py
|   +-- optimizer.py
|   +-- population.py
|
+-- results/
|   +-- database.py
|   +-- cache.py
|   +-- exporter.py
|
+-- plotting/
    +-- aero_plots.py
    +-- stability_plots.py
    +-- pareto_plots.py
    +-- aircraft_viewer.py
```

---

# 33. Important Software Abstraction

The GUI and optimiser should not call VSPAERO directly everywhere.

Use a solver interface.

Example:

```python
class AeroSolver:
    def evaluate(self, aircraft, condition):
        raise NotImplementedError
```

Implementation:

```python
class VSPAeroSolver(AeroSolver):
    def evaluate(self, aircraft, condition):
        ...
```

This prevents the UI from becoming tightly coupled to VSPAERO internals.

---

# 34. Main Results Object

Each aircraft evaluation should return a structured object.

Example:

```python
AeroResult(
    CL=...,
    CD_vspaero=...,
    CD_parasite=...,
    CD_xfoil_corrected=...,
    CDi=...,
    Cm=...,
    LD_vspaero=...,
    LD_xfoil=...,
    spanloads=...,
    stability_derivatives=...,
    dynamic_modes=...
)
```

---

# 35. Proposed Main UI

## Design tab

```text
AIRCRAFT
------------------------------------------------

Wing
Span
Aspect ratio
Taper
Sweep
Dihedral
Twist
Airfoil

Horizontal Tail
Area
Aspect ratio
Tail arm
Incidence

Vertical Tail
Area
Aspect ratio

CG / mass
```

---

## Mission tab

```text
MISSION / OPERATING CONDITION
------------------------------------------------

Altitude
Cruise speed
Stall-speed target
Maximum speed
Air density
Temperature
Mass
CG
```

---

## Aerodynamics tab

```text
AERODYNAMICS
------------------------------------------------

CL
Cm
CDi

Drag model:
[ VSPAERO ]
[ Parasite ]
[ XFoil corrected ]

L/D

Alpha sweep
Drag polar
Lift curve
Moment curve
Span loading
Cp distribution
```

---

## Stability tab

```text
STATIC STABILITY
------------------------------------------------

CG
Neutral point
Static margin

Cm_alpha
Cn_beta
Cl_beta

Control derivatives
```

---

## Dynamics tab

```text
DYNAMIC STABILITY
------------------------------------------------

Short period
Phugoid
Dutch roll
Roll
Spiral

Root locus

Mode:
[ Dutch Roll ▼ ]

Damping ratio
Natural frequency
Period
Settling time

[ Plot response ]
[ Animate mode ]
```

---

## Optimisation tab

```text
OPTIMISATION
------------------------------------------------

Objective
[ Minimum Cruise Power ▼ ]

Design Variables
                MIN        MAX
Span
Aspect Ratio
Taper
Sweep
Twist
Tail Area
Tail Arm

Constraints
[ ] Maximum span
[ ] Stall speed
[ ] Static margin
[ ] Dutch-roll damping
[ ] Short-period damping

Algorithm
[ Differential Evolution ▼ ]

Population
Generations

[ START OPTIMISATION ]
```

---

# 36. Results Dashboard

For each candidate display:

```text
DESIGN #184

Geometry
------------------------------------------------
Span                    1.42 m
AR                      10.2
Taper                    0.53
Twist                   -2.1 deg

Aerodynamics
------------------------------------------------
CL cruise                0.51
CDi                      0.012
CD VSPAERO               0.034
CD Parasite              0.038
CD XFoil corrected       0.036

L/D VSPAERO              15.0
L/D XFoil                14.2

Stability
------------------------------------------------
Static margin            11.4 %

Short-period zeta         0.63
Phugoid zeta              0.08
Dutch-roll zeta           0.31
Spiral                    stable

Constraints
------------------------------------------------
Span                      PASS
Stall                     PASS
Static margin             PASS
Dynamics                  PASS
```

---

# 37. Visualisation Goals

The custom UI should match or exceed the useful visual features of Flow5.

Important plots:

### Aerodynamic

- \(C_L\) vs \(\alpha\)
- \(C_D\) vs \(C_L\)
- \(C_m\) vs \(\alpha\)
- L/D vs \(\alpha\)
- L/D vs speed
- drag breakdown
- drag-model comparison
- spanwise \(C_l\)
- spanwise Reynolds number
- spanwise \(C_d\)
- spanwise stall margin

### Dynamics

- eigenvalue plot.
- root locus.
- time response.
- mode properties.
- optional 3D mode animation.

### Optimisation

- objective history.
- parameter convergence.
- Pareto front.
- constraint violations.
- candidate ranking.
- sensitivity plots.

---

# 38. Recommended Development Order

## Phase 1 — VSPAERO command-line proof of concept

Build a script that:

1. Creates a simple wing in OpenVSP.
2. Runs VSPAERO.
3. Reads:
   - \(C_L\)
   - \(C_D\)
   - \(C_{D_i}\)
   - \(C_m\)
   - spanwise loading.
4. Saves results to Python objects.

No GUI yet.

### Success criterion

```python
result = solver.evaluate(aircraft, condition)
```

returns valid aerodynamic results.

---

## Phase 2 — XFoil viscous correction

Implement:

1. XFoil polar generation.
2. Polar database.
3. Local Reynolds-number calculation.
4. VSPAERO span-load extraction.
5. Sectional \(C_d\) interpolation.
6. Wing profile-drag integration.

### Output

```text
CDi
CD VSPAERO
CD XFoil profile
CD XFoil total
```

---

## Phase 3 — Three-model drag comparison

Integrate OpenVSP Parasite Drag.

Produce:

```text
VSPAERO drag
Parasite drag
XFoil-corrected drag
```

and plot all three.

This becomes the main aerodynamic validation framework.

---

## Phase 4 — Aircraft parameterisation

Create Python objects for:

- wing.
- tail.
- controls.
- fuselage.
- mass / CG.

Connect them cleanly to OpenVSP.

---

## Phase 5 — PySide6 design UI

Create only the basic tabs:

- Aircraft.
- Mission.
- Aerodynamics.
- Results.

Avoid optimisation initially.

---

## Phase 6 — Static stability

Run VSPAERO stability analyses.

Display:

- neutral point.
- static margin.
- stability derivatives.
- control derivatives.

---

## Phase 7 — Dynamic stability

Build:

- dimensional derivative conversion.
- longitudinal A matrix.
- lateral-directional A matrix.
- eigenvalue extraction.
- automatic mode classification.

Output:

- short period.
- phugoid.
- Dutch roll.
- roll.
- spiral.

---

## Phase 8 — Dynamics visualisation

Add:

- eigenvalue plot.
- root locus.
- damping / frequency table.
- time response.

Optional later:

- 3D eigenmode animation.

---

## Phase 9 — Optimisation

Start with:

```python
scipy.optimize.differential_evolution
```

Optimise only 4–6 variables initially.

Example:

- span.
- AR.
- taper.
- twist.
- tail area.
- tail arm.

---

## Phase 10 — Parallelisation

After the optimiser works reliably:

- OpenVSP Facade.
- Multiple OpenVSP processes.
- parallel VSPAERO workers.
- result caching.

---

## Phase 11 — Multi-objective optimisation

Add pymoo / NSGA-II.

Generate Pareto fronts for:

- drag vs stability.
- endurance vs manoeuvrability.
- power vs static margin.
- performance vs geometric restrictions.

---

## Phase 12 — Validation

Compare selected designs against:

1. Flow5.
2. CFD.
3. Wind-tunnel data if available.
4. Flight-test data if available.

Use the differences to assess which drag model is most reliable.

---

# 39. First Minimum Viable Product

The MVP should **not** attempt everything.

It should do only:

### Geometry

- simple wing.
- horizontal tail.
- vertical tail.
- CG.

### Aero

- VSPAERO alpha sweep.
- lift.
- induced drag.
- pitching moment.
- span loading.

### Viscosity

- VSPAERO result.
- Parasite Drag result.
- XFoil-corrected result.

### Stability

- static margin.
- major stability derivatives.

### Dynamics

- eigenvalues.
- short period.
- phugoid.
- Dutch roll.
- roll.
- spiral.
- root-locus plot.

### Optimisation

- span.
- aspect ratio.
- taper.
- twist.
- tail size.
- tail arm.

### GUI

- design.
- aero.
- dynamics.
- optimisation.
- results.

That is already a very powerful UAV preliminary-design tool.

---

# 40. Main Technical Risks

## VSPAERO result parsing

Need robust access to:

- spanwise loads.
- derivative outputs.
- control derivative outputs.

Prefer API access where possible rather than fragile text parsing.

---

## XFoil convergence

XFoil can fail at:

- very low Reynolds number.
- high angle of attack.
- strongly separated flow.

The polar database must store convergence state.

Interpolation must never silently cross large invalid regions.

---

## Stall

Do not interpret potential-flow VSPAERO results as accurate deep-stall prediction.

Use sectional XFoil \(C_{l,max}\) initially as a warning/constraint.

---

## Dynamics conventions

Carefully control:

- axes.
- derivative signs.
- dimensional derivative equations.
- inertia conventions.
- reference quantities.

Validate the state-space implementation against known aircraft examples before trusting optimisation results.

---

## Optimiser exploiting solver weaknesses

Optimisers frequently discover physically meaningless solutions.

Every design should pass sanity checks before VSPAERO is run.

Example checks:

- positive chord.
- valid taper.
- reasonable sweep.
- valid tail arm.
- minimum Reynolds number.
- manufacturable geometry.
- acceptable static margin.

---

# 41. Recommended Validation Strategy

Before optimising a real UAV, validate individual modules separately.

## Aerodynamics

Known wing:

```text
VSPAERO
vs
Flow5
vs
published / experimental result
```

## Profile drag

Known airfoil:

```text
XFoil
vs
wind-tunnel polar
```

## Drag integration

Simple finite wing:

```text
VSPAERO CDi + XFoil profile drag
vs
Flow5
```

## Dynamics

Known conventional aircraft:

```text
VSPAERO derivatives
→ custom state-space model
→ compare modes with published solution
```

Only after these tests should automatic optimisation be trusted.

---

# 42. Final Recommended Architecture

```text
                    PySide6 UAV DESIGNER
                           |
                      Aircraft model
                           |
                     Mission condition
                           |
                           v
                        OpenVSP
                           |
                           v
                        VSPAERO
                           |
           +---------------+----------------+
           |               |                |
           v               v                v
       aero loads     spanwise loads   aero derivatives
           |               |                |
           |               v                v
           |          XFoil polar DB    state-space
           |               |                |
           |         profile viscous CD     |
           |               |                |
           +-------+-------+                |
                   |                        |
                   v                        v
              drag comparison          dynamics
                   |                        |
          +--------+--------+       +-------+-------+
          |        |        |       |       |       |
       VSPAERO  Parasite  XFoil   modes   plots  response
          |        |        |       |       |       |
          +--------+--------+-------+-------+-------+
                           |
                           v
                       optimiser
                           |
                           v
                    candidate database
                           |
                           v
                      final designs
```

---

# 43. Project Direction in One Sentence

> **Build a focused PySide6 UAV design environment that uses OpenVSP/VSPAERO for geometry and 3D aerodynamics, compares three independent drag/viscous models including an XFoil sectional correction, converts VSPAERO derivatives into full aircraft dynamic modes, and automatically searches the design space for aerodynamically efficient and dynamically acceptable UAV configurations.**

---

# 44. Immediate Next Step

Before building more GUI functionality, implement one end-to-end aerodynamic proof of concept:

```text
parameterised wing
        ↓
OpenVSP
        ↓
VSPAERO
        ↓
spanwise Cl + chord
        ↓
local Reynolds numbers
        ↓
XFoil polar interpolation
        ↓
three drag estimates
        ↓
CL / CD / L/D
```

Once this works reliably, add VSPAERO stability derivatives and the state-space dynamics module.

Only after the aerodynamic and dynamics core is validated should the optimisation UI become the main development focus.
