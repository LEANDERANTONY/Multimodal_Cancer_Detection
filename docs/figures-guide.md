# Figures Guide

This document explains how figures in this repository should be treated.

## Purpose

The `figures/` directory is for curated, lightweight, presentation- or report-worthy outputs.

It is not intended to hold every intermediate notebook plot generated during exploration.

## What Should Be Tracked

Good candidates for tracked figures:

- final comparison plots used in reports
- representative Grad-CAM outputs
- key calibration, fusion, or sanity-check visuals
- figures that support the thesis narrative or repository README/docs

## What Should Stay Local

Keep these local unless there is a deliberate reason to track them:

- repeated exploratory plot variants
- temporary debugging figures
- bulk image dumps from notebook experiments
- large sets of intermediate visual audit outputs

## Naming Guidance

Prefer descriptive, stable names such as:

- `fusion_decision_level_auc_vs_weight.png`
- `fusion_decision_sanity_auc.png`
- `biomarker_calibration.png`
- `ct_cluster0_gradcam.png`

Avoid names that depend on vague words like:

- `final_final_plot.png`
- `test2.png`
- `new_graph.png`

## Relationship To Reports

When possible, tracked figures should line up with tracked report outputs in `reports/`.

Examples:

- a tracked CSV summary in `reports/`
- a matching explanatory figure in `figures/`

That pairing makes the repo easier to understand without access to the raw data or notebook kernel state.

## Reproducibility Guidance

When a figure is important enough to track, the logic that produces it should ideally be:

- generated from the main notebook with stable cell flow, or
- backed by reusable helpers in `src/`

The long-term direction is to reduce one-off figure-generation logic that only exists in an untracked kernel state.

## Current Reality

This repo still contains a mix of:

- curated final figures worth tracking
- thesis-era artifacts preserved during consolidation

As the repo matures, the goal is to keep only figures that materially help:

- understand the experimental outcomes
- support the thesis or publication narrative
- document the methodology clearly
