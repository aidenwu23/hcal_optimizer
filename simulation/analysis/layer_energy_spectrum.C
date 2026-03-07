/*
Take one event, sum the HCal_Readout energy in each logical HCal layer (which is just the energy in that 
layer’s scintillator layer), keep each nonzero layer sum as one entry, repeat over all events, and 
histogram those layer-energy values.

Basically a graph of how many scintillator layers had what amount of energy deposited.

root -l -b -q 'simulation/analysis/layer_energy_spectrum.C("data/raw/<geom_id>/<run_id>.edm4hep.root","layer_energy_spectrum.root")'

*/
#include <TH1D.h>
#include <TFile.h>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include "edm4hep/SimCalorimeterHitCollection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kHcalLayerCount = 10;
constexpr int kLayerBitOffset = 8;
constexpr std::uint64_t kLayerMask = 0xFF;
constexpr int kDefaultBinCount = 80;

// The readout encodes the logical sampling-layer index in bits 8..15.
int decode_hcal_layer(std::uint64_t cell_id) {
  return static_cast<int>((cell_id >> kLayerBitOffset) & kLayerMask);
}

// Log-spaced bins make the low-energy structure visible over a wide range.
std::vector<double> build_log_edges(double minimum, double maximum, int n_bins) {
  std::vector<double> edges(static_cast<std::size_t>(n_bins + 1), minimum);
  const double log_min = std::log10(minimum);
  const double log_max = std::log10(maximum);
  const double step = (log_max - log_min) / static_cast<double>(n_bins);
  for (int bin_index = 0; bin_index <= n_bins; ++bin_index) {
    edges[static_cast<std::size_t>(bin_index)] =
        std::pow(10.0, log_min + step * static_cast<double>(bin_index));
  }
  return edges;
}

// Quantiles.
double quantile(std::vector<double> values, double probability) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const double index = probability * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(index));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
  if (lower == upper) {
    return values[lower];
  }
  const double fraction = index - static_cast<double>(lower);
  return values[lower] + fraction * (values[upper] - values[lower]);
}

std::string pick_category(podio::ROOTReader& reader) {
  std::string category = "events";
  try {
    auto categories = reader.getAvailableCategories();
    if (!categories.empty() &&
        std::find(categories.begin(), categories.end(), "events") == categories.end()) {
      category = categories.front();
    }
  } catch (...) {
  }
  return category;
}

}  // namespace

void layer_energy_spectrum(const char* raw_events_path_cstr,
                           const char* out_root_cstr = "layer_energy_spectrum.root") {
  if (!raw_events_path_cstr || std::string(raw_events_path_cstr).empty()) {
    std::cerr << "[layer_energy_spectrum] Raw EDM4hep path is required.\n";
    return;
  }

  const std::string raw_events_path(raw_events_path_cstr);
  const std::string out_root =
      (out_root_cstr && std::string(out_root_cstr).size())
          ? std::string(out_root_cstr)
          : std::string("layer_energy_spectrum.root");
  const std::string collection_name = "HCal_Readout";

  // Read the raw EDM4hep file.
  podio::ROOTReader reader;
  try {
    reader.openFile(raw_events_path);
  } catch (const std::exception& error) {
    std::cerr << "[layer_energy_spectrum] Failed to open " << raw_events_path
              << ": " << error.what() << ".\n";
    return;
  }

  const std::string category = pick_category(reader);
  const std::size_t entry_count = reader.getEntries(category);

  // This spectrum treats every nonzero active-layer deposit as one entry.
  std::vector<double> nonzero_layer_energies_GeV;

  // Allocate space up front for a rough expected number of nonzero layer-energy entries.
  nonzero_layer_energies_GeV.reserve(entry_count * static_cast<std::size_t>(kHcalLayerCount) / 2U);

  long long event_count = 0;
  long long active_layer_count = 0;
  long long zero_layer_count = 0;

  // Loop through all events
  for (std::size_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    auto frame_data = reader.readEntry(category, entry_index);
    podio::Frame frame(std::move(frame_data));

    const edm4hep::SimCalorimeterHitCollection* sim_collection = nullptr;
    try {
      sim_collection = &frame.get<edm4hep::SimCalorimeterHitCollection>(collection_name);
    } catch (...) {
      continue;
    }
    if (!sim_collection) {
      continue;
    }

    ++event_count;
    // Sum all hit energy in the same logical sampling layer for this event.
    std::array<double, kHcalLayerCount> layer_energy_GeV {};
    for (const auto& hit : *sim_collection) {

      const int layer_index = decode_hcal_layer(static_cast<std::uint64_t>(hit.getCellID()));

      if (layer_index < 0 || layer_index >= kHcalLayerCount) {
        std::cerr << "[layer_energy_spectrum] Decoded HCAL layer index is out of range: "
                  << layer_index << ".\n";
        return;
      }

      // hit.getEnergy() only returns the energy of the active readout layer, which is attached to the
      // scintillator layer in this pipeline.
      layer_energy_GeV[static_cast<std::size_t>(layer_index)] += hit.getEnergy();
    }

    // Keep only nonzero layers because the threshold question is about visible deposits.
    for (double layer_energy : layer_energy_GeV) {
      if (layer_energy > 0.0) {
        nonzero_layer_energies_GeV.push_back(layer_energy);
        ++active_layer_count;
      } else {
        ++zero_layer_count;
      }
    }
  }

  if (nonzero_layer_energies_GeV.empty()) {
    std::cerr << "[layer_energy_spectrum] No nonzero layer energies found.\n";
    return;
  }

  // Track minimum and maximum energies to make bounds for the histogram.
  const auto [min_it, max_it] =
      std::minmax_element(nonzero_layer_energies_GeV.begin(), nonzero_layer_energies_GeV.end());
  const double minimum_energy_GeV = *min_it;
  double maximum_energy_GeV = *max_it;
  if (!(maximum_energy_GeV > minimum_energy_GeV)) {
    maximum_energy_GeV = minimum_energy_GeV * 10.0;
  }

  // Create histogram.
  const std::vector<double> edges =
      build_log_edges(minimum_energy_GeV, maximum_energy_GeV * 1.05, kDefaultBinCount);
  TH1D histogram(
      "h_layer_energy",
      "Per-layer active energy spectrum;Layer energy [GeV];Layer count",
      kDefaultBinCount,
      edges.data());
  for (double layer_energy : nonzero_layer_energies_GeV) {
    histogram.Fill(layer_energy);
  }
  histogram.SetDirectory(nullptr);
  histogram.SetStats(false);

  TFile output_file(out_root.c_str(), "RECREATE");
  if (output_file.IsZombie()) {
    std::cerr << "[layer_energy_spectrum] Failed to open " << out_root
              << " for writing.\n";
    return;
  }
  histogram.Write();
  output_file.Close();

  std::cout << "[layer_energy_spectrum] Wrote " << out_root << "\n";
  std::cout << "[layer_energy_spectrum] Events=" << event_count
            << " nonzero_layers=" << active_layer_count
            << " zero_layers=" << zero_layer_count << "\n";
  std::cout << "[layer_energy_spectrum] q10=" << quantile(nonzero_layer_energies_GeV, 0.10)
            << " GeV q50=" << quantile(nonzero_layer_energies_GeV, 0.50)
            << " GeV q90=" << quantile(nonzero_layer_energies_GeV, 0.90) << " GeV\n";
}
