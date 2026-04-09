/*
Write segment Landau-fit histograms from a raw muon EDM4hep file.
Example:
root -l -b -q 'analysis/simulation/plot_landau.C("data/raw/04e3fdfb/run_mu_ctrl_10k/run_mu_ctrl.edm4hep.root","landau_plots.root")'
*/

#include <TF1.h>
#include <TH1D.h>
#include <TFile.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include "edm4hep/SimCalorimeterHitCollection.h"

namespace {

constexpr int kLayerCount = 10;
constexpr int kSegmentCount = 3;
constexpr int kEnergyBinCount = 400;
constexpr int kLayerBitOffset = 8;
constexpr std::uint64_t kLayerMask = 0xFF;
constexpr double kRangeQuantile = 0.995;

int decode_layer_index(std::uint64_t cell_id) {
  return static_cast<int>((cell_id >> kLayerBitOffset) & kLayerMask);
}

int layer_to_segment(int layer_index) {
  if (layer_index < 3) {
    return 0;
  }
  if (layer_index < 6) {
    return 1;
  }
  return 2;
}

std::string sibling_path(const std::string& path, const std::string& basename) {
  const auto separator_index = path.find_last_of("/\\");
  if (separator_index == std::string::npos) {
    return basename;
  }
  return path.substr(0, separator_index + 1) + basename;
}

double peak_bin_center(const TH1D& histogram) {
  if (histogram.GetEntries() <= 0.0) {
    return 0.0;
  }
  return histogram.GetBinCenter(histogram.GetMaximumBin());
}

double quantile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }

  const double clamped_fraction = std::clamp(fraction, 0.0, 1.0);
  const std::size_t index = static_cast<std::size_t>(
      clamped_fraction * static_cast<double>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
  return values[index];
}

double histogram_xmax(const std::vector<double>& values) {
  if (values.empty()) {
    return 1.0;
  }

  const double median = quantile(values, 0.50);
  const double upper_quantile = quantile(values, kRangeQuantile);
  const double scaled_quantile = upper_quantile > 0.0 ? upper_quantile * 1.15 : 0.0;
  const double scaled_median = median > 0.0 ? median * 6.0 : 0.0;
  return std::max({scaled_quantile, scaled_median, 1e-6});
}

// Read raw calorimeter hits and collect one per-layer summed energy per event.
bool collect_segment_layer_energies(
    const std::string& events_path,
    std::array<std::vector<double>, kSegmentCount>& segment_values) {
  podio::ROOTReader reader;
  try {
    reader.openFile(events_path);
  } catch (const std::exception& error) {
    std::cerr << "[plot_landau] Failed to open " << events_path << ": "
              << error.what() << ".\n";
    return false;
  }

  std::string category = "events";
  try {
    const auto categories = reader.getAvailableCategories();
    if (!categories.empty() &&
        std::find(categories.begin(), categories.end(), "events") == categories.end()) {
      category = categories.front();
    }
  } catch (...) {
  }

  const size_t entry_count = reader.getEntries(category);
  for (std::size_t segment_index = 0; segment_index < segment_values.size(); ++segment_index) {
    const int layer_count = segment_index < 2 ? 3 : 4;
    segment_values[segment_index].reserve(entry_count * static_cast<std::size_t>(layer_count));
  }

  for (size_t event_index = 0; event_index < entry_count; ++event_index) {
    auto frame_data = reader.readEntry(category, event_index);
    podio::Frame frame(std::move(frame_data));

    std::array<double, kLayerCount> layer_energy {};
    layer_energy.fill(0.0);

    const edm4hep::SimCalorimeterHitCollection* sim_collection = nullptr;
    try {
      sim_collection = &frame.get<edm4hep::SimCalorimeterHitCollection>("HCal_Readout");
    } catch (...) {
      sim_collection = nullptr;
    }
    if (sim_collection) {
      // Straight muons make cell-level spectra pile up near zero in off-track edge cells.
      // Sum by layer so the plots match the calibration observable instead.
      for (const auto& hit : *sim_collection) {
        const int layer_index = decode_layer_index(static_cast<std::uint64_t>(hit.getCellID()));
        if (layer_index < 0 || layer_index >= kLayerCount) {
          continue;
        }
        layer_energy[static_cast<std::size_t>(layer_index)] += static_cast<double>(hit.getEnergy());
      }
    }

    // Keep one per-layer energy entry per event, pooled by segment.
    for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
      const double value = layer_energy[static_cast<std::size_t>(layer_index)];
      if (!std::isfinite(value) || value < 0.0) {
        continue;
      }
      const int segment_index = layer_to_segment(layer_index);
      segment_values[static_cast<std::size_t>(segment_index)].push_back(value);
    }
  }

  return true;
}

}  // namespace

void plot_landau(const char* events_path_cstr, const char* out_path_cstr = "") {
  if (!events_path_cstr || std::string(events_path_cstr).empty()) {
    std::cerr << "[plot_landau] Raw EDM4hep path is required.\n";
    return;
  }

  const std::string events_path(events_path_cstr);
  const std::string out_path =
      (out_path_cstr && std::string(out_path_cstr).size())
          ? std::string(out_path_cstr)
          : sibling_path(events_path, "landau_plots.root");

  std::array<std::vector<double>, kSegmentCount> segment_values;
  if (!collect_segment_layer_energies(events_path, segment_values)) {
    return;
  }

  std::array<double, kSegmentCount> segment_xmax {};
  for (std::size_t segment_index = 0; segment_index < segment_values.size(); ++segment_index) {
    segment_xmax[segment_index] = histogram_xmax(segment_values[segment_index]);
  }

  std::array<TH1D, kSegmentCount> segment_histograms = {
      TH1D("h_seg1_mip", "Segment 1 muon layer deposits;Layer energy [GeV];Layer count", kEnergyBinCount, 0.0, segment_xmax[0]),
      TH1D("h_seg2_mip", "Segment 2 muon layer deposits;Layer energy [GeV];Layer count", kEnergyBinCount, 0.0, segment_xmax[1]),
      TH1D("h_seg3_mip", "Segment 3 muon layer deposits;Layer energy [GeV];Layer count", kEnergyBinCount, 0.0, segment_xmax[2]),
  };

  for (TH1D& histogram : segment_histograms) {
    histogram.SetDirectory(nullptr);
    histogram.SetStats(false);
  }

  for (std::size_t segment_index = 0; segment_index < segment_values.size(); ++segment_index) {
    for (const double value : segment_values[segment_index]) {
      segment_histograms[segment_index].Fill(value);
    }
  }

  std::array<std::unique_ptr<TF1>, kSegmentCount> landau_fits;
  for (int segment_index = 0; segment_index < kSegmentCount; ++segment_index) {
    TH1D& histogram = segment_histograms[static_cast<std::size_t>(segment_index)];
    const double fallback_mpv = peak_bin_center(histogram);
    const double xmin = std::max(0.0, 0.25 * fallback_mpv);
    const double xmax = std::min(histogram.GetXaxis()->GetXmax(), 3.0 * fallback_mpv);
    const std::string fit_name = "landau_fit_seg" + std::to_string(segment_index + 1);
    landau_fits[static_cast<std::size_t>(segment_index)] = std::make_unique<TF1>(
        fit_name.c_str(), "landau", xmin, xmax);

    TF1& landau_fit = *landau_fits[static_cast<std::size_t>(segment_index)];
    landau_fit.SetParameters(
        histogram.GetMaximum(),
        fallback_mpv,
        std::max(1e-6, fallback_mpv * 0.25));

    const int fit_status = histogram.Fit(&landau_fit, "QR0");
    const double mu = landau_fit.GetParameter(1);
    const double sigma = landau_fit.GetParameter(2);
    const double fitted_mu = (fit_status == 0 && std::isfinite(mu)) ? mu : fallback_mpv;
    double fitted_mpv = fallback_mpv;
    if (fit_status == 0 && std::isfinite(mu) && std::isfinite(sigma) && sigma > 0.0) {
      const double mpv = mu - 0.22278298 * sigma;
      if (std::isfinite(mpv) && mpv >= 0.0) {
        fitted_mpv = mpv;
      }
    }
    const double median = quantile(segment_values[static_cast<std::size_t>(segment_index)], 0.50);
    const double p90 = quantile(segment_values[static_cast<std::size_t>(segment_index)], 0.90);
    const double p99 = quantile(segment_values[static_cast<std::size_t>(segment_index)], 0.99);

    std::cout << "[plot_landau] segment " << (segment_index + 1)
              << " entries=" << histogram.GetEntries()
              << " median=" << median
              << " p90=" << p90
              << " p99=" << p99
              << " xmax=" << histogram.GetXaxis()->GetXmax()
              << " fallback_mpv=" << fallback_mpv
              << " fitted_mu=" << fitted_mu
              << " fitted_sigma=" << sigma
              << " fitted_mpv=" << fitted_mpv
              << " fit_status=" << fit_status
              << " chi2_ndf=" << landau_fit.GetChisquare() << "/" << landau_fit.GetNDF()
              << "\n";
  }

  TFile output_file(out_path.c_str(), "RECREATE");
  if (output_file.IsZombie()) {
    std::cerr << "[plot_landau] Failed to open " << out_path << " for writing.\n";
    return;
  }

  for (int segment_index = 0; segment_index < kSegmentCount; ++segment_index) {
    segment_histograms[static_cast<std::size_t>(segment_index)].Write();
    landau_fits[static_cast<std::size_t>(segment_index)]->Write();
  }

  output_file.Close();
  std::cout << "[plot_landau] Wrote " << out_path << ".\n";
}
