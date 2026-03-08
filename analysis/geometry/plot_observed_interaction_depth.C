/*
Build observed interaction-depth curves from raw HCAL hits.
The macro marks the first layer above a muon-calibrated threshold and writes cumulative curves.

Example:
root -l -b -q 'analysis/geometry/plot_observed_interaction_depth.C("data/raw/2663fc88/run42d6c963ff.edm4hep.root")'
*/

#include <TDirectory.h>
#include <TFile.h>
#include <TFormula.h>
#include <TGraph.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>

#include "edm4hep/MCParticleCollection.h"
#include "edm4hep/SimCalorimeterHitCollection.h"

namespace {

constexpr int kLayerBitOffset = 8;
constexpr std::uint64_t kLayerMask = 0xFF;

struct LayerDepthRow {
  int layer_index = -1;
  double depth_back_mm = 0.0;
};

struct ObservedLayerRow {
  int layer_index = -1;
  double depth_back_mm = 0.0;
  long long observed_start_count = 0;
  double observed_start_fraction = 0.0;
  double cumulative_observed_start_fraction = 0.0;
};

int decode_hcal_layer(std::uint64_t cell_id) {
  return static_cast<int>((cell_id >> kLayerBitOffset) & kLayerMask);
}

std::string project_root_from_macro() {
  std::filesystem::path macro_path(__FILE__);
  return macro_path.parent_path().parent_path().parent_path().string();
}

std::string derive_run_id_from_raw_path(const std::string& raw_path) {
  std::filesystem::path raw_file(raw_path);
  std::string file_name = raw_file.filename().string();
  constexpr const char* suffix = ".edm4hep.root";
  const std::size_t suffix_length = std::char_traits<char>::length(suffix);
  if (file_name.size() > suffix_length &&
      file_name.compare(file_name.size() - suffix_length, suffix_length, suffix) == 0) {
    file_name.erase(file_name.size() - suffix_length);
  }
  return file_name;
}

std::string derive_geometry_id_from_raw_path(const std::string& raw_path) {
  return std::filesystem::path(raw_path).parent_path().filename().string();
}

std::string default_meta_path_from_raw(const std::string& raw_path) {
  const std::filesystem::path project_root(project_root_from_macro());
  const std::string geometry_id = derive_geometry_id_from_raw_path(raw_path);
  const std::string run_id = derive_run_id_from_raw_path(raw_path);
  return (project_root / "data" / "processed" / geometry_id / run_id / "meta.json").string();
}

std::string default_calibration_path_from_raw(const std::string& raw_path) {
  const std::filesystem::path project_root(project_root_from_macro());
  const std::string geometry_id = derive_geometry_id_from_raw_path(raw_path);
  const std::string run_id = derive_run_id_from_raw_path(raw_path);
  return (project_root / "data" / "processed" / geometry_id / run_id / "calibration.json").string();
}

std::string geometry_json_path(const std::string& geometry_id) {
  const std::filesystem::path project_root(project_root_from_macro());
  return (project_root / "geometries" / "generated" / geometry_id / "geometry.json").string();
}

std::string default_layers_csv_path(const std::string& geometry_id, const std::string& run_id) {
  const std::filesystem::path project_root(project_root_from_macro());
  return (
      project_root /
      "data" /
      "geometry_analysis" /
      geometry_id /
      run_id /
      "start_layer_observed_layers.csv").string();
}

std::string default_output_root_path(const std::string& geometry_id, const std::string& run_id) {
  const std::filesystem::path project_root(project_root_from_macro());
  return (
      project_root /
      "data" /
      "geometry_analysis" /
      geometry_id /
      run_id /
      "observed_interaction_depth.root").string();
}

bool read_required_int(const nlohmann::json& payload, const char* key, int& value) {
  if (!payload.contains(key) || !payload[key].is_number_integer()) {
    return false;
  }
  value = payload[key].get<int>();
  return true;
}

void replace_all(std::string& text, const std::string& from, const std::string& to) {
  if (from.empty()) {
    return;
  }
  std::size_t start_index = 0;
  while ((start_index = text.find(from, start_index)) != std::string::npos) {
    text.replace(start_index, from.size(), to);
    start_index += to.size();
  }
}

double eval_length_mm(const nlohmann::json& value) {
  if (value.is_number()) {
    return value.get<double>();
  }
  if (!value.is_string()) {
    return 0.0;
  }

  std::string expression = value.get<std::string>();
  if (expression.empty()) {
    return 0.0;
  }

  replace_all(expression, "mm", "1.0");
  replace_all(expression, "cm", "10.0");
  replace_all(expression, "m", "1000.0");

  static int formula_counter = 0;
  TFormula formula(
      ("length_formula_" + std::to_string(formula_counter++)).c_str(),
      expression.c_str());
  return formula.Eval(0.0);
}

// Build one depth row for each physical HCAL layer.
std::vector<LayerDepthRow> build_layer_depth_rows(const nlohmann::json& geometry_json) {
  int segment_layer_counts[3] = {0, 0, 0};
  if (!read_required_int(geometry_json, "seg1_layers", segment_layer_counts[0]) ||
      !read_required_int(geometry_json, "seg2_layers", segment_layer_counts[1]) ||
      !read_required_int(geometry_json, "seg3_layers", segment_layer_counts[2])) {
    throw std::runtime_error("Geometry JSON is missing one or more segment layer counts.");
  }

  const double spacer_thickness_mm = eval_length_mm(geometry_json.at("t_spacer"));
  std::vector<LayerDepthRow> layer_rows;
  double running_depth_mm = 0.0;
  int layer_index = 0;

  // Read each segment thickness and expand it into physical layers.
  for (int segment_index = 0; segment_index < 3; ++segment_index) {
    const std::string absorber_key = "t_absorber_seg" + std::to_string(segment_index + 1);
    const std::string scintillator_key = "t_scin_seg" + std::to_string(segment_index + 1);
    const double absorber_thickness_mm = eval_length_mm(geometry_json.at(absorber_key));
    const double scintillator_thickness_mm = eval_length_mm(geometry_json.at(scintillator_key));
    const double layer_total_thickness_mm =
        absorber_thickness_mm + scintillator_thickness_mm + 2.0 * spacer_thickness_mm;

    // Add one cumulative depth row for each layer in this segment.
    for (int layer_in_segment = 0; layer_in_segment < segment_layer_counts[segment_index]; ++layer_in_segment) {
      running_depth_mm += layer_total_thickness_mm;
      layer_rows.push_back(LayerDepthRow{
          layer_index,
          running_depth_mm,
      });
      layer_index++;
    }
  }

  return layer_rows;
}

// Read and parse a required JSON file.
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

double energy_from_mc(const edm4hep::MCParticle& particle) {
  const double energy = particle.getEnergy();
  if (energy > 0.0) {
    return energy;
  }
  const auto& momentum = particle.getMomentum();
  const double momentum_sq =
      momentum.x * momentum.x + momentum.y * momentum.y + momentum.z * momentum.z;
  const double mass = particle.getMass();
  return std::sqrt(std::max(0.0, momentum_sq + mass * mass));
}

// Pick the MC particle that best matches the injected beam particle.
std::optional<edm4hep::MCParticle> select_primary_candidate(
    const edm4hep::MCParticleCollection& mc_collection,
    int expected_pdg) {
  std::optional<edm4hep::MCParticle> same_pdg_not_from_sim_candidate;
  std::optional<edm4hep::MCParticle> same_pdg_candidate;
  std::optional<edm4hep::MCParticle> not_from_sim_candidate;
  std::optional<edm4hep::MCParticle> highest_energy_candidate;
  double same_pdg_not_from_sim_candidate_energy = -1.0;
  double same_pdg_candidate_energy = -1.0;
  double not_from_sim_candidate_energy = -1.0;
  double highest_energy_candidate_energy = -1.0;

  // Track fallback candidates while scanning the truth record.
  for (const auto& particle : mc_collection) {
    const double energy = energy_from_mc(particle);
    const bool not_from_sim = !particle.isCreatedInSimulation();
    const int pdg = particle.getPDG();

    if (not_from_sim && energy > not_from_sim_candidate_energy) {
      not_from_sim_candidate = particle;
      not_from_sim_candidate_energy = energy;
    }
    if (energy > highest_energy_candidate_energy) {
      highest_energy_candidate = particle;
      highest_energy_candidate_energy = energy;
    }

    // Keep the best matches to the requested PDG in a separate priority tier.
    if (expected_pdg != 0 && pdg != 0 && std::abs(pdg) == std::abs(expected_pdg)) {
      if (not_from_sim && energy > same_pdg_not_from_sim_candidate_energy) {
        same_pdg_not_from_sim_candidate = particle;
        same_pdg_not_from_sim_candidate_energy = energy;
      }
      if (energy > same_pdg_candidate_energy) {
        same_pdg_candidate = particle;
        same_pdg_candidate_energy = energy;
      }
    }
  }

  if (same_pdg_not_from_sim_candidate) {
    return same_pdg_not_from_sim_candidate;
  }
  if (same_pdg_candidate) {
    return same_pdg_candidate;
  }
  if (not_from_sim_candidate) {
    return not_from_sim_candidate;
  }
  return highest_energy_candidate;
}

// Convert first-hit counts into cumulative observed-start probabilities.
std::vector<ObservedLayerRow> build_observed_layer_rows(
    const std::vector<LayerDepthRow>& layer_depth_rows,
    const std::vector<long long>& start_counts,
    long long valid_event_count) {
  if (layer_depth_rows.size() != start_counts.size()) {
    throw std::runtime_error("Layer depth rows and observed start counts have inconsistent sizes.");
  }

  std::vector<ObservedLayerRow> rows;
  long long cumulative_start_count = 0;

  // Accumulate the per-layer count and cumulative fraction together.
  for (std::size_t index = 0; index < layer_depth_rows.size(); ++index) {
    cumulative_start_count += start_counts[index];
    const double normalization = valid_event_count > 0 ? static_cast<double>(valid_event_count) : 1.0;
    rows.push_back(ObservedLayerRow{
        layer_depth_rows[index].layer_index,
        layer_depth_rows[index].depth_back_mm,
        start_counts[index],
        static_cast<double>(start_counts[index]) / normalization,
        static_cast<double>(cumulative_start_count) / normalization,
    });
  }
  return rows;
}

// Write the layer-by-layer observed-start summary.
void write_layers_csv(const std::string& csv_path, const std::vector<ObservedLayerRow>& rows) {
  const std::filesystem::path output_path(csv_path);
  std::filesystem::create_directories(output_path.parent_path());

  std::ofstream output(csv_path);
  if (!output) {
    throw std::runtime_error("Failed to open observed-interaction CSV for writing.");
  }

  output << "layer_index,depth_back_mm,observed_start_count,observed_start_fraction,"
            "cumulative_observed_start_fraction\n";
  for (const auto& row : rows) {
    output << row.layer_index << ","
           << row.depth_back_mm << ","
           << row.observed_start_count << ","
           << row.observed_start_fraction << ","
           << row.cumulative_observed_start_fraction << "\n";
  }
}

// Write cumulative observed-start graphs vs depth and layer.
void write_root_file(const std::string& root_path, const std::vector<ObservedLayerRow>& rows) {
  const std::filesystem::path output_path(root_path);
  std::filesystem::create_directories(output_path.parent_path());

  TFile output_file(root_path.c_str(), "RECREATE");
  if (output_file.IsZombie()) {
    throw std::runtime_error("Failed to open observed-interaction ROOT file for writing.");
  }

  TDirectory* depth_directory = output_file.mkdir("depth");
  TDirectory* layer_directory = output_file.mkdir("layer");

  TGraph p_start_observed_vs_depth_mm(static_cast<int>(rows.size()));
  p_start_observed_vs_depth_mm.SetName("p_start_observed_vs_depth_mm");
  p_start_observed_vs_depth_mm.SetTitle(
      "Observed cumulative start probability;Depth [mm];P_{start,observed}");
  p_start_observed_vs_depth_mm.SetLineWidth(2);

  TGraph p_start_observed_vs_layer(static_cast<int>(rows.size()));
  p_start_observed_vs_layer.SetName("p_start_observed_vs_layer");
  p_start_observed_vs_layer.SetTitle(
      "Observed cumulative start probability;Layer index;P_{start,observed}");
  p_start_observed_vs_layer.SetLineWidth(2);

  // Fill both graphs from the same cumulative layer rows.
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const auto& row = rows[index];
    p_start_observed_vs_depth_mm.SetPoint(
        static_cast<int>(index),
        row.depth_back_mm,
        row.cumulative_observed_start_fraction);
    p_start_observed_vs_layer.SetPoint(
        static_cast<int>(index),
        static_cast<double>(row.layer_index),
        row.cumulative_observed_start_fraction);
  }

  depth_directory->cd();
  p_start_observed_vs_depth_mm.Write();

  layer_directory->cd();
  p_start_observed_vs_layer.Write();

  output_file.Close();
}

// Find the first layer whose visible energy crosses the observed-start threshold.
int find_observed_start_layer(const std::vector<double>& layer_energy_GeV, double threshold_GeV) {
  for (std::size_t layer_index = 0; layer_index < layer_energy_GeV.size(); ++layer_index) {
    if (layer_energy_GeV[layer_index] > threshold_GeV) {
      return static_cast<int>(layer_index);
    }
  }
  return -1;
}

std::string string_arg_or_default(const char* value, const std::string& fallback) {
  if (value && std::string(value).size()) {
    return std::string(value);
  }
  return fallback;
}

}  // namespace

// Build the observed interaction-depth curves from raw HCAL hits.
void plot_observed_interaction_depth(const char* raw_path_cstr,
                                     const char* meta_path_cstr = "",
                                     const char* calibration_path_cstr = "",
                                     const char* out_csv_cstr = "",
                                     const char* out_root_cstr = "",
                                     double threshold_scale = 0.2,
                                     int expected_pdg = 0,
                                     const char* hit_collection_cstr = "HCal_Readout") {
  // Validate the main runtime inputs first.
  if (!raw_path_cstr || std::string(raw_path_cstr).empty()) {
    std::cerr << "[plot_observed_interaction_depth] raw EDM4hep path is required.\n";
    return;
  }
  if (threshold_scale <= 0.0) {
    std::cerr << "[plot_observed_interaction_depth] threshold_scale must be positive.\n";
    return;
  }

  const std::string raw_path(raw_path_cstr);
  const std::string geometry_id = derive_geometry_id_from_raw_path(raw_path);
  const std::string run_id = derive_run_id_from_raw_path(raw_path);
  const std::string meta_path = string_arg_or_default(meta_path_cstr, default_meta_path_from_raw(raw_path));
  const std::string calibration_path =
      string_arg_or_default(calibration_path_cstr, default_calibration_path_from_raw(raw_path));
  const std::string out_csv_path =
      string_arg_or_default(out_csv_cstr, default_layers_csv_path(geometry_id, run_id));
  const std::string out_root_path =
      string_arg_or_default(out_root_cstr, default_output_root_path(geometry_id, run_id));
  const std::string hit_collection = string_arg_or_default(hit_collection_cstr, "HCal_Readout");

  // Load the run metadata and calibration inputs.
  nlohmann::json meta_json;
  if (!load_json_file(meta_path, "meta.json", meta_json, "plot_observed_interaction_depth")) {
    return;
  }

  nlohmann::json calibration_json;
  if (!load_json_file(
          calibration_path,
          "calibration.json",
          calibration_json,
          "plot_observed_interaction_depth")) {
    return;
  }

  // Validate the fields needed to build the observed-start threshold.
  if (!meta_json.contains("gun_energy_GeV") || !meta_json["gun_energy_GeV"].is_number()) {
    std::cerr << "[plot_observed_interaction_depth] meta.json is missing gun_energy_GeV.\n";
    return;
  }
  if (!calibration_json.contains("muon_threshold_GeV") ||
      !calibration_json["muon_threshold_GeV"].is_number()) {
    std::cerr << "[plot_observed_interaction_depth] calibration.json is missing muon_threshold_GeV.\n";
    return;
  }

  const double gun_energy_GeV = meta_json["gun_energy_GeV"].get<double>();
  const double muon_threshold_GeV = calibration_json["muon_threshold_GeV"].get<double>();
  const double start_threshold_GeV = threshold_scale * muon_threshold_GeV;
  const double mc_rel_diff_limit = 0.1;  // Match the 10% beam-energy window used elsewhere.

  // Load the generated geometry and expand it into layer depths.
  nlohmann::json geometry_json;
  const std::string geometry_path = geometry_json_path(geometry_id);
  if (!load_json_file(
          geometry_path,
          "geometry.json",
          geometry_json,
          "plot_observed_interaction_depth")) {
    return;
  }

  std::vector<LayerDepthRow> layer_depth_rows;
  try {
    layer_depth_rows = build_layer_depth_rows(geometry_json);
  } catch (const std::exception& error) {
    std::cerr << "[plot_observed_interaction_depth] Failed to build HCAL layer depths: "
              << error.what() << ".\n";
    return;
  }
  if (layer_depth_rows.empty()) {
    std::cerr << "[plot_observed_interaction_depth] Geometry has no HCAL layers.\n";
    return;
  }

  // Open the raw event file after the static inputs are ready.
  podio::ROOTReader reader;
  try {
    reader.openFile(raw_path);
  } catch (const std::exception& error) {
    std::cerr << "[plot_observed_interaction_depth] Failed to open " << raw_path << ": "
              << error.what() << ".\n";
    return;
  }

  std::string event_category = "events";
  // Fall back to the first available category when "events" is not present.
  try {
    auto categories = reader.getAvailableCategories();
    if (!categories.empty() &&
        std::find(categories.begin(), categories.end(), "events") == categories.end()) {
      event_category = categories.front();
    }
  } catch (...) {
  }

  std::vector<long long> start_counts(layer_depth_rows.size(), 0);
  long long valid_event_count = 0;
  long long events_with_observed_start = 0;
  const std::size_t entry_count = reader.getEntries(event_category);

  // Reduce each selected event to one observed start layer.
  for (std::size_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    auto frame_data = reader.readEntry(event_category, entry_index);
    podio::Frame frame(std::move(frame_data));

    const edm4hep::MCParticleCollection* mc_collection = nullptr;
    try {
      mc_collection = &frame.get<edm4hep::MCParticleCollection>("MCParticles");
    } catch (...) {
    }
    if (!mc_collection) {
      continue;
    }

    // Keep only frames whose primary particle matches the requested beam sample.
    const std::optional<edm4hep::MCParticle> selected_candidate =
        select_primary_candidate(*mc_collection, expected_pdg);
    if (!selected_candidate) {
      continue;
    }

    const double mc_energy_GeV = energy_from_mc(*selected_candidate);
    if (mc_energy_GeV <= 0.0 || gun_energy_GeV <= 0.0) {
      continue;
    }
    const double relative_difference = std::abs(mc_energy_GeV - gun_energy_GeV) / gun_energy_GeV;
    if (relative_difference > mc_rel_diff_limit) {
      continue;
    }

    const edm4hep::SimCalorimeterHitCollection* sim_collection = nullptr;
    try {
      sim_collection = &frame.get<edm4hep::SimCalorimeterHitCollection>(hit_collection);
    } catch (...) {
    }
    if (!sim_collection) {
      continue;
    }

    valid_event_count++;
    std::vector<double> layer_energy_GeV(layer_depth_rows.size(), 0.0);

    // Sum visible energy per layer before finding the first threshold crossing.
    for (const auto& hit : *sim_collection) {
      const int layer_index = decode_hcal_layer(static_cast<std::uint64_t>(hit.getCellID()));
      if (layer_index < 0 || static_cast<std::size_t>(layer_index) >= layer_energy_GeV.size()) {
        std::cerr << "[plot_observed_interaction_depth] Layer index out of range: "
                  << layer_index << ".\n";
        return;
      }
      layer_energy_GeV[static_cast<std::size_t>(layer_index)] += hit.getEnergy();
    }

    const int start_layer = find_observed_start_layer(layer_energy_GeV, start_threshold_GeV);
    if (start_layer < 0) {
      continue;
    }

    start_counts[static_cast<std::size_t>(start_layer)]++;
    events_with_observed_start++;
  }

  if (valid_event_count <= 0) {
    std::cerr << "[plot_observed_interaction_depth] No valid events survived the selection.\n";
    return;
  }

  std::vector<ObservedLayerRow> observed_rows;
  try {
    observed_rows = build_observed_layer_rows(layer_depth_rows, start_counts, valid_event_count);
    write_layers_csv(out_csv_path, observed_rows);
    write_root_file(out_root_path, observed_rows);
  } catch (const std::exception& error) {
    std::cerr << "[plot_observed_interaction_depth] Failed to write outputs: "
              << error.what() << ".\n";
    return;
  }

  std::cout << "[plot_observed_interaction_depth] Wrote " << out_csv_path << ".\n";
  std::cout << "[plot_observed_interaction_depth] Wrote " << out_root_path << ".\n";
  std::cout << "[plot_observed_interaction_depth] valid_events=" << valid_event_count
            << " events_with_observed_start=" << events_with_observed_start
            << " start_threshold_GeV=" << start_threshold_GeV << ".\n";
}
