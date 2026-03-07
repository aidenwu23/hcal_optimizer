/* Calibrate the visible-to-truth neutron energy scale on detected events. */
// root -l -b -q 'simulation/calibration/calibrate_neutron_response.C("path/to/events.root",5.0,0.01,"path/to/output.json")'

#include <TFile.h>
#include <TTree.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

void calibrate_neutron_response(const char* events_path_cstr,
                                double gun_energy_GeV,
                                double muon_threshold_GeV,
                                const char* out_json_cstr) {
  if (!events_path_cstr || std::string(events_path_cstr).empty()) {
    std::cerr << "[calibrate_neutron_response] events.root path is required.\n";
    return;
  }
  if (!out_json_cstr || std::string(out_json_cstr).empty()) {
    std::cerr << "[calibrate_neutron_response] Output json path is required.\n";
    return;
  }
  if (!(gun_energy_GeV > 0.0)) {
    std::cerr << "[calibrate_neutron_response] gun_energy_GeV must be positive.\n";
    return;
  }
  if (muon_threshold_GeV < 0.0) {
    std::cerr << "[calibrate_neutron_response] muon_threshold_GeV must be non-negative.\n";
    return;
  }

  const std::string events_path(events_path_cstr);
  const std::string out_json_path(out_json_cstr);
  const double mc_rel_diff_limit = 0.1;

  TFile input_file(events_path.c_str(), "READ");
  if (input_file.IsZombie()) {
    std::cerr << "[calibrate_neutron_response] Failed to open " << events_path << ".\n";
    return;
  }

  TTree* tree = nullptr;
  input_file.GetObject("events", tree);
  if (!tree) {
    std::cerr << "[calibrate_neutron_response] Tree 'events' not found in " << events_path << ".\n";
    return;
  }
  if (tree->GetBranch("mc_E") == nullptr || tree->GetBranch("visible_E") == nullptr) {
    std::cerr << "[calibrate_neutron_response] Branches mc_E and visible_E are required in "
              << events_path << ".\n";
    return;
  }

  float mc_E = 0.0F;
  float visible_E = 0.0F;
  std::string* category_ptr = nullptr;
  tree->SetBranchAddress("mc_E", &mc_E);
  tree->SetBranchAddress("visible_E", &visible_E);
  if (tree->GetBranch("category") != nullptr) {
    tree->SetBranchAddress("category", &category_ptr);
  }

  long long detected_event_count = 0;
  double detected_visible_energy_sum_GeV = 0.0;
  double detected_truth_energy_sum_GeV = 0.0;

  const Long64_t entry_count = tree->GetEntries();
  for (Long64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    tree->GetEntry(entry_index);

    const std::string category = (category_ptr != nullptr) ? *category_ptr : std::string();
    if (!category.empty() && category != "events") {
      continue;
    }

    const double truth_energy_GeV = static_cast<double>(mc_E);
    if (truth_energy_GeV <= 0.0) {
      continue;
    }
    const double relative_difference = std::abs(truth_energy_GeV - gun_energy_GeV) / gun_energy_GeV;
    if (relative_difference > mc_rel_diff_limit) {
      continue;
    }

    const double visible_energy_GeV = static_cast<double>(visible_E);
    if (visible_energy_GeV < muon_threshold_GeV) {
      continue;
    }

    detected_event_count++;
    detected_visible_energy_sum_GeV += visible_energy_GeV;
    detected_truth_energy_sum_GeV += truth_energy_GeV;
  }

  if (detected_event_count <= 0 || detected_truth_energy_sum_GeV <= 0.0) {
    std::cerr << "[calibrate_neutron_response] No detected neutron events passed the calibration selection.\n";
    return;
  }

  // This value will be used as the sampling fraction.
  const double neutron_scale = detected_visible_energy_sum_GeV / detected_truth_energy_sum_GeV;
  if (!std::isfinite(neutron_scale) || neutron_scale <= 0.0) {
    std::cerr << "[calibrate_neutron_response] Computed neutron_scale is not finite and positive.\n";
    return;
  }

  std::ofstream output_file(out_json_path);
  if (!output_file) {
    std::cerr << "[calibrate_neutron_response] Failed to open " << out_json_path
              << " for writing.\n";
    return;
  }
  output_file << "{\n";
  output_file << "  \"neutron_scale\": " << neutron_scale << "\n";
  output_file << "}\n";
  std::cout << "[calibrate_neutron_response] Wrote " << out_json_path << ".\n";
}
