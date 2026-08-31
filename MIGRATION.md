# FlightLxx migration map

| Concern | FlightLxx source | Isaac Lab migration |
|---|---|---|
| Mass | `rl_env.yaml`: 0.78 kg | Custom rigid object, 0.78 kg |
| Arm length | 0.125 m | 0.25 m span placeholder geometry |
| Inertia | FlightLxx derived diagonal | PhysX tensor diagonal `(0.00228515625, 0.00228515625, 0.0035546875)` kg m² |
| Control period | 0.02 s | Physics 0.01 s, decimation 2 |
| Motor response | 0.033 s | Configurable first-order wrench response |
| Positive thrust maximum | 54.813 N from four 29,400 rpm motors | CTBR collective saturation at 54.813 N |
| Old action | Four independent signed motor thrusts | CTBR `[collective, p, q, r]` with rate inner loop |
| Attitude actor input | Euler angles | Canonical WXYZ quaternion error |
| Integration/graphics | Custom RK4 and Unity | PhysX rigid body and Isaac Sim headless |
| Learning | PPO2 | Official FastTD3 plus minimal TCN model extension |

The first phase is positive-thrust CTBR. Bidirectional thrust maps, Unity transport, the old RK4 integrator, vision, trajectory tracking, hard-coded inverted reset, and impulse auxiliary prediction are intentionally not enabled.

The rate inner loop retains FlightLxx's diagonal inertia-scaled P behavior and the `ω × Jω` gyroscopic compensation term. Its actuator state initializes at hover thrust on reset instead of ramping from zero.

The target is a fixed hover position with identity quaternion and zero linear/angular velocity. Policy observation is 625-D: current state 13, fast raw history `4×17`, and slow raw history `32×17`. The causal TCN compresses only the slow part inside Actor/Critic networks. The critic receives the 625-D policy prefix plus 16 privileged values.

Disturbance hooks cover timed force/torque, wind-compatible sustained force, velocity impulse, and a 0.6 kg physical sphere launch. A disturbance never clears history.

`training_stage` separates the curriculum into `small_recovery` (±30°, ±1 m/s, ±2 rad/s), `large_recovery` (up to ±90°, ±3 m/s, ±6 rad/s), and `disturbance` (starts exactly at hover and applies the perturbation only after 0.5–1.5 s of accumulated history). Recovery success requires remaining inside the recovery set for 0.3 s.
