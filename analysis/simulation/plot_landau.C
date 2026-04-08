/*
Write segment Landau-fit histograms from a muon control events.root file.
Example:
root -l -b -q 'analysis/simulation/plot_landau.C("data/processed/04e3fdfb/run_mu_ctrl/events.root","landau_plots.root")'
*/

#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TTree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kLayerCount = 10;
constexpr int kSegmentCount = 3;
constexpr int kEnergyBinCount = 400;
constexpr double kRangeQuantile = 0.995;

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
  const double x_max = std::max({scaled_quantile, scaled_median, 1e-6});
  return x_max;
}

}  // namespace

void plot_landau(const char* events_path_cstr, const char* out_path_cstr = "") {
  if (!events_path_cstr || std::string(events_path_cstr).empty()) {
    std::cerr << "[plot_landau] events.root path is required.\n";
    return;
  }

  const std::string events_path(events_path_cstr);
  const std::string out_path =
      (out_path_cstr && std::string(out_path_cstr).size())
          ? std::string(out_path_cstr)
          : sibling_path(events_path, "landau_plots.root");

  TFile input_file(events_path.c_str(), "READ");
  if (input_file.IsZombie()) {
    std::cerr << "[plot_landau] Failed to open " << events_path << ".\n";
    return;
  }

  TTree* tree = nullptr;
  input_file.GetObject("events", tree);
  if (!tree) {
    std::cerr << "[plot_landau] Tree 'events' not found in " << events_path << ".\n";
    return;
  }

  std::array<float, kLayerCount> layer_energy {};
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    const std::string branch_name = "layer_" + std::to_string(layer_index) + "_E";
    if (tree->GetBranch(branch_name.c_str()) == nullptr) {
      std::cerr << "[plot_landau] Branch '" << branch_name << "' not found in "
                << events_path << ".\n";
      return;
    }
    tree->SetBranchAddress(branch_name.c_str(), &layer_energy[static_cast<std::size_t>(layer_index)]);
  }

  std::array<std::vector<double>, kSegmentCount> segment_values;
  const Long64_t entry_count = tree->GetEntries();
  for (std::size_t segment_index = 0; segment_index < segment_values.size(); ++segment_index) {
    const int layer_count = segment_index < 2 ? 3 : 4;
    segment_values[segment_index].reserve(static_cast<std::size_t>(entry_count) * layer_count);
  }

  for (Long64_t event_index = 0; event_index < entry_count; ++event_index) {
    tree->GetEntry(event_index);

    // Collect one value per layer so the histogram range can ignore rare outliers.
    for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
      const double value = static_cast<double>(layer_energy[static_cast<std::size_t>(layer_index)]);
      if (!std::isfinite(value) || value < 0.0) {
        continue;
      }
      const int segment_index = layer_to_segment(layer_index);
      segment_values[static_cast<std::size_t>(segment_index)].push_back(value);
    }
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

  for (Long64_t event_index = 0; event_index < entry_count; ++event_index) {
    tree->GetEntry(event_index);

    // Fill one histogram per segment with all same-segment layer deposits.
    for (int layer_index = 0; layer_index < kLayerCount; ++layer_index) {
      const double value = static_cast<double>(layer_energy[static_cast<std::size_t>(layer_index)]);
      if (!std::isfinite(value) || value < 0.0) {
        continue;
      }
      const int segment_index = layer_to_segment(layer_index);
      segment_histograms[static_cast<std::size_t>(segment_index)].Fill(value);
    }
  }

  std::array<std::unique_ptr<TF1>, kSegmentCount> landau_fits;
  for (int segment_index = 0; segment_index < kSegmentCount; ++segment_index) {
    TH1D& histogram = segment_histograms[static_cast<std::size_t>(segment_index)];
    const double fallback_mpv = peak_bin_center(histogram);
    const std::string fit_name = "landau_fit_seg" + std::to_string(segment_index + 1);
    landau_fits[static_cast<std::size_t>(segment_index)] = std::make_unique<TF1>(
        fit_name.c_str(), "landau", 0.0, histogram.GetXaxis()->GetXmax());

    TF1& landau_fit = *landau_fits[static_cast<std::size_t>(segment_index)];
    landau_fit.SetParameters(
        histogram.GetMaximum(),
        fallback_mpv,
        std::max(1e-6, fallback_mpv * 0.25));

    const int fit_status = histogram.Fit(&landau_fit, "Q0R");
    const double fitted_mpv = landau_fit.GetParameter(1);
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
