/*
Particle performance summary.
*/

#include <TFile.h>
#include <TTree.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

namespace {

struct PerformanceStats {
  long long valid_event_count = 0;
  long long detected_event_count = 0;
  double reco_energy_sum = 0.0;
  double reco_energy_sum_sq = 0.0;
};

struct BinomialInterval {
  double mean = 0.0;
  double error_low = 0.0;
  double error_high = 0.0;
};

std::string sibling_path(const std::string& path, const std::string& basename) {
  const auto separator_index = path.find_last_of("/\\");
  if (separator_index == std::string::npos) {
    return basename;
  }
  return path.substr(0, separator_index + 1) + basename;
}

std::string string_arg_or_default(const char* value, const std::string& fallback) {
  if (value && std::string(value).size()) {
    return std::string(value);
  }
  return fallback;
}

bool load_json_file(const std::string& path,
                    const char* label,
                    nlohmann::json& json_payload,
                    const char* context_label) {
  std::ifstream input(path);
  if (!input) {
    std::cerr << "[" << context_label << "] Failed to open " << label << " at " << path << ".\n";
    return false;
  }

  try {
    input >> json_payload;
  } catch (const std::exception& error) {
    std::cerr << "[" << context_label << "] Failed to parse " << label << ": "
              << error.what() << ".\n";
    return false;
  }

  return true;
}

BinomialInterval wilson_interval(long long successes, long long trials, double z_value) {
  BinomialInterval interval;
  if (trials <= 0) {
    return interval;
  }
  const double trial_count = static_cast<double>(trials);
  const double fraction = static_cast<double>(successes) / trial_count;
  const double z_squared = z_value * z_value;
  const double denominator = 1.0 + z_squared / trial_count;
  const double center = (fraction + z_squared / (2.0 * trial_count)) / denominator;
  const double margin =
      (z_value / denominator) * std::sqrt((fraction * (1.0 - fraction) / trial_count) + (z_squared / (4.0 * trial_count * trial_count)));
  const double lower = std::max(0.0, center - margin);
  const double upper = std::min(1.0, center + margin);
  interval.mean = fraction;
  interval.error_low = std::max(0.0, fraction - lower);
  interval.error_high = std::max(0.0, upper - fraction);
  return interval;
}

}  // namespace

void performance(const char* events_path_cstr, const char* meta_path_cstr = "",
                 const char* calibration_path_cstr = "", const char* out_path_cstr = "") {
  // Validate the main runtime input first.
  if (!events_path_cstr || std::string(events_path_cstr).empty()) {
    std::cerr << "[performance] events.root path is required.\n";
    return;
  }

  const std::string events_path(events_path_cstr);
  const std::string meta_path = string_arg_or_default(meta_path_cstr, sibling_path(events_path, "meta.json"));
  const std::string calibration_path =
      string_arg_or_default(calibration_path_cstr, sibling_path(events_path, "calibration.json"));
  const std::string out_path =
      string_arg_or_default(out_path_cstr, sibling_path(events_path, "performance.json"));

  nlohmann::json meta_json;
  if (!load_json_file(meta_path, "meta.json", meta_json, "performance")) {
    return;
  }

  // Load the metadata and calibration needed for the event selection.
  nlohmann::json calibration_json;
  if (!load_json_file(
          calibration_path,
          "calibration.json",
          calibration_json,
          "performance")) {
    return;
  }

  // Validate the fields needed for efficiency and optional energy-resolution calculation.
  if (!meta_json.contains("geometry_id") || !meta_json["geometry_id"].is_string()) {
    std::cerr << "[performance] meta.json is missing geometry_id.\n";
    return;
  }
  if (!meta_json.contains("gun_particle") || !meta_json["gun_particle"].is_string()) {
    std::cerr << "[performance] meta.json is missing gun_particle.\n";
    return;
  }
  if (!meta_json.contains("total_energy_GeV") || !meta_json["total_energy_GeV"].is_number()) {
    std::cerr << "[performance] meta.json is missing total_energy_GeV.\n";
    return;
  }
  if (!calibration_json.contains("muon_threshold_GeV") || !calibration_json["muon_threshold_GeV"].is_number()) {
    std::cerr << "[performance] calibration.json is missing muon_threshold_GeV.\n";
    return;
  }

  const std::string geometry_id = meta_json["geometry_id"].get<std::string>();
  const std::string gun_particle = meta_json["gun_particle"].get<std::string>();
  const double total_energy_GeV = meta_json["total_energy_GeV"].get<double>();
  const double muon_threshold_GeV = calibration_json["muon_threshold_GeV"].get<double>();
  const bool has_response_scale =
      calibration_json.contains("response_scale") && calibration_json["response_scale"].is_number();
  const double response_scale =
      has_response_scale ? calibration_json["response_scale"].get<double>() : 0.0;
  const double wilson_z = 1.0;  // Use a 1-sigma Wilson interval for the efficiency error.
  if (muon_threshold_GeV < 0.0 || !std::isfinite(muon_threshold_GeV)) {
    std::cerr << "[performance] muon_threshold_GeV must be finite and non-negative.\n";
    return;
  }
  if (has_response_scale && (!(response_scale > 0.0) || !std::isfinite(response_scale))) {
    std::cerr << "[performance] response_scale must be finite and positive when present.\n";
    return;
  }

  // Open the processed event tree and bind the required branches.
  TFile input_file(events_path.c_str(), "READ");
  if (input_file.IsZombie()) {
    std::cerr << "[performance] Failed to open " << events_path << ".\n";
    return;
  }

  TTree* tree = nullptr;
  input_file.GetObject("events", tree);
  if (!tree) {
    std::cerr << "[performance] Tree 'events' not found in " << events_path << ".\n";
    return;
  }
  if (tree->GetBranch("mc_E") == nullptr) {
    std::cerr << "[performance] Branch mc_E is required in " << events_path << ".\n";
    return;
  }
  if (tree->GetBranch("visible_E") == nullptr) {
    std::cerr << "[performance] Branch visible_E is required in " << events_path << ".\n";
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

  PerformanceStats stats;
  const Long64_t entry_count = tree->GetEntries();
  // Accumulate detection and energy statistics over the processed run.
  for (Long64_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    tree->GetEntry(entry_index);

    const std::string category =
        (category_ptr != nullptr) ? *category_ptr : std::string();
    if (!category.empty() && category != "events") {
      continue;
    }

    const double mc_energy_GeV = static_cast<double>(mc_E);
    if (mc_energy_GeV <= 0.0 || total_energy_GeV <= 0.0) {
      continue;
    }

    stats.valid_event_count++;
    if (static_cast<double>(visible_E) >= muon_threshold_GeV) {
      stats.detected_event_count++;
      if (has_response_scale) {
        const double reconstructed_energy_GeV = static_cast<double>(visible_E) / response_scale;
        stats.reco_energy_sum += reconstructed_energy_GeV;
        stats.reco_energy_sum_sq += reconstructed_energy_GeV * reconstructed_energy_GeV;
      }
    }
  }

  // Convert the accumulated counters into the JSON summary payload.
  nlohmann::json output;
  output["geometry_id"] = geometry_id;
  output["gun_particle"] = gun_particle;
  if (meta_json.contains("kinetic_energy_GeV")) {
    output["kinetic_energy_GeV"] = meta_json["kinetic_energy_GeV"];
  }
  output["total_energy_GeV"] = total_energy_GeV;
  output["muon_threshold_GeV"] = muon_threshold_GeV;
  output["response_scale"] = has_response_scale ? nlohmann::json(response_scale) : nlohmann::json(nullptr);
  output["valid_event_count"] = stats.valid_event_count;
  output["detected_event_count"] = stats.detected_event_count;

  const BinomialInterval efficiency_interval =
      wilson_interval(stats.detected_event_count, stats.valid_event_count, wilson_z);
  if (stats.valid_event_count > 0) {
    output["detection_efficiency"] = efficiency_interval.mean;
    output["eff_lo"] = efficiency_interval.error_low;
    output["eff_hi"] = efficiency_interval.error_high;
  } else {
    output["detection_efficiency"] = nullptr;
    output["eff_lo"] = nullptr;
    output["eff_hi"] = nullptr;
  }

  if (has_response_scale && stats.detected_event_count > 1 && total_energy_GeV > 0.0) {
    const double mean_reconstructed_energy_GeV =
        stats.reco_energy_sum / static_cast<double>(stats.detected_event_count);
    double variance =
        (stats.reco_energy_sum_sq - static_cast<double>(stats.detected_event_count) *
          mean_reconstructed_energy_GeV * mean_reconstructed_energy_GeV) / static_cast<double>(stats.detected_event_count - 1);
    if (variance < 0.0) {
      variance = 0.0;
    }
    output["energy_resolution"] = std::sqrt(variance) / total_energy_GeV;
  } else {
    output["energy_resolution"] = nullptr;
  }

  // Write the final performance summary next to the run outputs.
  std::ofstream output_file(out_path);
  if (!output_file) {
    std::cerr << "[performance] Failed to open " << out_path << " for writing.\n";
    return;
  }
  output_file << output.dump(2) << '\n';
  std::cout << "[performance] Wrote " << out_path << ".\n";
}
