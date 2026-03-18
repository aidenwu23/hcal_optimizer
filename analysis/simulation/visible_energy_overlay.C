/*
Overlay the visible-energy distributions for a muon control sample and a neutron signal sample.

Both input files must contain an "events" TTree with a "visible_E" branch (output of processor.cc).
The two histograms are normalised to unit area so their shapes are directly comparable.

root -l -b -q 'analysis/simulation/visible_energy_overlay.C("muon/events.root","neutron/events.root","out.root")'
root -l -b -q 'analysis/simulation/visible_energy_overlay.C("data/processed/1144444a/run_mu_ctrl/events.root","data/processed/1144444a/runa1be6b3be8/events.root","data/result_validation/visible_energy_overlay.root")'
root -l -b -q 'analysis/simulation/visible_energy_overlay.C("data/processed/04e3fdfb/run_mu_ctrl/events.root","data/processed/04e3fdfb/run7f378b22da/events.root","data/result_validation/visible_energy_overlay_baseline.root")'
*/

#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBinCount = 60;
constexpr const char* kBranchName = "visible_E";
constexpr const char* kTreeName = "events";

// Read every finite visible_E value from the named tree into a vector.
std::vector<double> read_visible_e(const std::string& file_path, const char* label) {
  TFile input(file_path.c_str(), "READ");
  if (input.IsZombie()) {
    std::cerr << "[visible_energy_overlay] Failed to open " << label << " at " << file_path << ".\n";
    return {};
  }

  TTree* tree = nullptr;
  input.GetObject(kTreeName, tree);
  if (!tree) {
    std::cerr << "[visible_energy_overlay] Tree '" << kTreeName << "' not found in " << file_path << ".\n";
    return {};
  }

  float visible_e = 0.0F;
  tree->SetBranchAddress(kBranchName, &visible_e);

  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(tree->GetEntries()));
  const Long64_t entry_count = tree->GetEntries();
  for (Long64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    tree->GetEntry(entry_index);
    const double value = static_cast<double>(visible_e);
    if (std::isfinite(value) && value >= 0.0) {
      values.push_back(value);
    }
  }

  return values;
}

// Build a fixed-bin histogram over a shared axis and normalise it to unit area.
TH1D* build_histogram(
    const char* name,
    const char* title,
    const std::vector<double>& values,
    double x_min,
    double x_max) {
  TH1D* histogram = new TH1D(name, title, kBinCount, x_min, x_max);
  histogram->SetDirectory(nullptr);
  histogram->SetStats(false);
  for (double value : values) {
    histogram->Fill(value);
  }
  if (histogram->Integral() > 0.0) {
    histogram->Scale(1.0 / histogram->Integral());
  }
  return histogram;
}

}  // namespace

void visible_energy_overlay(
    const char* muon_events_path_cstr,
    const char* neutron_events_path_cstr,
    const char* out_root_cstr = "visible_energy_overlay.root") {
  if (!muon_events_path_cstr || std::string(muon_events_path_cstr).empty()) {
    std::cerr << "[visible_energy_overlay] Muon events path is required.\n";
    return;
  }
  if (!neutron_events_path_cstr || std::string(neutron_events_path_cstr).empty()) {
    std::cerr << "[visible_energy_overlay] Neutron events path is required.\n";
    return;
  }

  const std::string muon_path(muon_events_path_cstr);
  const std::string neutron_path(neutron_events_path_cstr);
  const std::string out_path(
      (out_root_cstr && std::string(out_root_cstr).size())
          ? std::string(out_root_cstr)
          : std::string("visible_energy_overlay.root"));

  const std::vector<double> muon_values = read_visible_e(muon_path, "muon");
  const std::vector<double> neutron_values = read_visible_e(neutron_path, "neutron");

  if (muon_values.empty()) {
    std::cerr << "[visible_energy_overlay] No valid muon visible_E values.\n";
    return;
  }
  if (neutron_values.empty()) {
    std::cerr << "[visible_energy_overlay] No valid neutron visible_E values.\n";
    return;
  }

  // Derive a shared x-axis range from both samples combined.
  const double global_max = std::max(
      *std::max_element(muon_values.begin(), muon_values.end()),
      *std::max_element(neutron_values.begin(), neutron_values.end()));
  const double x_min = 0.0;
  const double x_max = global_max * 1.05;

  TH1D* muon_histogram = build_histogram(
      "muon_visible_E",
      "Muon visible energy;Visible energy [GeV];Normalised counts",
      muon_values,
      x_min,
      x_max);
  muon_histogram->SetLineColor(kBlue + 1);
  muon_histogram->SetLineWidth(2);

  TH1D* neutron_histogram = build_histogram(
      "neutron_visible_E",
      "Neutron visible energy;Visible energy [GeV];Normalised counts",
      neutron_values,
      x_min,
      x_max);
  neutron_histogram->SetLineColor(kRed + 1);
  neutron_histogram->SetLineWidth(2);
  neutron_histogram->SetLineStyle(7);

  // Overlay canvas.
  TCanvas* overlay_canvas = new TCanvas("visible_energy_overlay", "visible_energy_overlay", 900, 700);
  const double y_max = std::max(muon_histogram->GetMaximum(), neutron_histogram->GetMaximum()) * 1.15;
  muon_histogram->SetMaximum(y_max);
  muon_histogram->SetTitle("Visible energy distribution;Visible energy [GeV];Normalised counts");
  muon_histogram->Draw("HIST");
  neutron_histogram->Draw("HIST SAME");

  TLegend* legend = new TLegend(0.62, 0.75, 0.88, 0.88);
  legend->AddEntry(muon_histogram, "Muon (control)", "l");
  legend->AddEntry(neutron_histogram, "Neutron (signal)", "l");
  legend->Draw();

  TFile output_file(out_path.c_str(), "RECREATE");
  if (output_file.IsZombie()) {
    std::cerr << "[visible_energy_overlay] Failed to open " << out_path << " for writing.\n";
    return;
  }

  muon_histogram->Write();
  neutron_histogram->Write();
  overlay_canvas->Write();
  output_file.Close();

  std::cout << "[visible_energy_overlay] Wrote " << out_path << ".\n";
  std::cout << "[visible_energy_overlay] Muon events=" << muon_values.size()
            << " Neutron events=" << neutron_values.size() << ".\n";
}
