# Neutron Reconstruction Calibration Plan

## Goal

Replace the placeholder `0.03` with a neutron reconstruction calibration that matches what this repo is trying to simulate.

Define `rec_E` as reconstructed neutron energy for detected events only. Keep detection efficiency as a separate metric.

## Scope

The useful quantity here is not a generic sampling fraction from literature. It is an effective neutron response scale for each geometry in this repo, applied to the detected neutron sample.

The calibration target is:

`f_n = <sim_E> / <mc_E>`

and the reconstructed energy stays:

`rec_E = sim_E / f_n`

if a single constant is good enough.

This is a neutron reconstruction calibration, not a textbook muon or MIP sampling fraction.

## Plan

1. Keep the meaning of `rec_E` fixed downstream.

   `rec_E` should represent reconstructed neutron energy after the detector has passed the detection requirement. Detection efficiency should remain a separate observable.

2. Use neutron gun samples for calibration.

   Derive the calibration from neutron runs, not muons. Muons are still useful for threshold studies, but they do not set the neutron energy scale.

3. Calibrate from existing event observables.

   Use the `events.root` tree that already contains `mc_E`, `sim_E`, and `start_layer`. No new detector observable is needed for the first pass.

4. Measure the neutron response per geometry on detected events.

   For each `geometry_id`, compute the detected-sample response with:

   `f_n = mean(sim_E) / mean(mc_E)`

   and also inspect the event-by-event `sim_E / mc_E` distribution to see how wide and asymmetric it is.

5. Test whether one constant is enough.

   Run several neutron energies in the range that matters for this study. If `mean(sim_E) / mean(mc_E)` is stable across energy, keep one constant per geometry. If it changes with energy, use an energy-dependent calibration instead.

6. Check for an offset term.

   Fit `sim_E = alpha * mc_E + beta` across neutron energies.

   If `beta` is small, store `alpha` as the calibration constant.

   If `beta` is not small, replace the single-fraction model with:

   `rec_E = (sim_E - beta) / alpha`

7. Separate response calibration from detection efficiency.

   Calibrate `rec_E` using only detected neutron events.

   Keep detection efficiency as its own metric over all valid neutron events.

   Do not fold missed neutrons into the reconstructed energy scale, because that would mix detection failure with energy response.

8. Store calibration beside run products.

   Write one calibration JSON per geometry, with the fitted neutron response value and an explicit note that the fit used detected neutron events.

9. Connect the calibration to processing.

   Update the processing flow so the calibrated neutron response can be loaded from JSON instead of relying on the hard-coded `0.03`, while keeping `--sampling` as a manual override.

10. Validate the calibrated response.

    After calibration, check that reconstructed neutron energy for detected events is less biased relative to truth and that the result is stable across the tested neutron energies.

11. Keep the muon estimate separate.

    If needed, compute a muon or MIP sampling fraction only as a geometry sanity check. Do not use it as the main neutron reconstruction scale.

## Expected Outcome

The repo will likely need a neutron reconstruction calibration per geometry, and possibly per energy range if the response is nonlinear. A single literature value is unlikely to be justified for all generated layouts.

## First Implementation Target

Add a small calibration tool that reads `events.root`, computes the neutron response scale from `mc_E` and `sim_E`, and writes a JSON file per geometry. After that, wire the result into the existing processing pipeline.
