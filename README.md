# Speed-driven-transitions-between-discrete-and-rhythmic-dynamics-in-walking
This repository contains the scripts used in the paper  "Speed-driven transitions between discrete and rhythmic dynamics in walking revealed by kinematic smoothness and muscle synergies"

KinematicAnalysisRAMP.m - [Kinematics] Loads the converted RAMP/RAMPDOWN trials; computes spatiotemporal gait parameters and the LDJ/SAL smoothness metrics; detects constant-speed blocks; segments everything by treadmill speed-change indices.

EMG_RAMP_preproc.m - [EMG preprocessing] Loads raw EMG together with the segmented kinematics; high-pass + notch filtering, envelope extraction, global and segment-wise spike cleaning, stride and speed segmentation, channel-quality GUI, and per-segment normalisation.

EMG_GroupSyn_NNMF.m - [Synergies] Group-level synergy extraction across subjects via NNMF; VAF sweep and synergy-number selection.

EMG_RAMP_synxsubj.m - [Synergies] Subject-level NNMF synergy extractions.
