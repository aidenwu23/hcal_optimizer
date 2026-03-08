/*
Observed start-layer summary derived directly from raw EDM4hep calorimeter hits.
*/

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <optional>
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

struct StartLayerStats {
  long long valid_event_count = 0;
  long long start_in_seg1_count = 0;
  long long start_in_seg2_count = 0;
  long long start_in_seg3_count = 0;
  std::vector<int> start_layers;
};

struct SegmentBoundaries {
  int seg1_layers = 0;
  int seg2_layers = 0;
  int seg3_layers = 0;
  int total_layers = 0;
};

// The HCAL layer number is encoded in the calorimeter cell id, so decode it once here and keep
// the event loop focused on the observed start-layer logic.
int decode_hcal_layer(std::uint64_t cell_id) {
  return static_cast<int>((cell_id >> kLayerBitOffset) & kLayerMask);
}

// Resolve file locations relative to the repo root so the macro can move between the raw
// EDM4hep input, the generated geometry, and the analysis output tree.
std::string project_root_from_macro() {
  std::filesystem::path macro_path(__FILE__);
  return macro_path.parent_path().parent_path().parent_path().string();
}

std::string geometry_json_path(const std::string& geometry_id) {
  const std::filesystem::path project_root(project_root_from_macro());
  return (project_root / "geometries" / "generated" / geometry_id / "geometry.json").string();
}

bool read_required_int(const nlohmann::json& payload, const char* key, int& value) {
  if (!payload.contains(key) || !payload[key].is_number_integer()) {
    return false;
  }
  value = payload[key].get<int>();
  return true;
}

// Read the generated geometry description and recover the three HCAL segment boundaries used
// later to group observed start layers by segment.
bool load_segment_boundaries(const std::string& geometry_id, SegmentBoundaries& boundaries) {
  std::ifstream geometry_input(geometry_json_path(geometry_id));
  if (!geometry_input) {
    return false;
  }

  nlohmann::json geometry_json;
  geometry_input >> geometry_json;

  if (!read_required_int(geometry_json, "seg1_layers", boundaries.seg1_layers) ||
      !read_required_int(geometry_json, "seg2_layers", boundaries.seg2_layers) ||
      !read_required_int(geometry_json, "seg3_layers", boundaries.seg3_layers)) {
    return false;
  }

  boundaries.total_layers = boundaries.seg1_layers + boundaries.seg2_layers + boundaries.seg3_layers;
  return boundaries.seg1_layers > 0 && boundaries.seg2_layers > 0 && boundaries.seg3_layers > 0;
}

// Summarize the observed start-layer distribution with a median layer index.
double median_layer_index(std::vector<int> start_layers) {
  if (start_layers.empty()) {
    return -1.0;
  }
  std::sort(start_layers.begin(), start_layers.end());
  const std::size_t middle = start_layers.size() / 2;
  if (start_layers.size() % 2 == 1) {
    return static_cast<double>(start_layers[middle]);
  }
  return 0.5 * static_cast<double>(start_layers[middle - 1] + start_layers[middle]);
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

// Derive the processed run id from the raw EDM4hep file name so the macro can find matching
// metadata and write to the standard output location.
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

std::string default_meta_path_from_raw(const std::string& raw_path) {
  const std::filesystem::path raw_file(raw_path);
  const std::string geometry_id = raw_file.parent_path().filename().string();
  const std::string run_id = derive_run_id_from_raw_path(raw_path);
  const std::filesystem::path project_root(project_root_from_macro());
  return (project_root / "data" / "processed" / geometry_id / run_id / "meta.json").string();
}

// Keep the observed start-layer summary under data/geometry_analysis/<geometry_id>/<run_id>.
std::string default_output_path(const std::string& geometry_id, const std::string& run_id) {
  const std::filesystem::path project_root(project_root_from_macro());
  return (
      project_root /
      "data" /
      "geometry_analysis" /
      geometry_id /
      run_id /
      "start_layer_observed.json").string();
}

// Pick the MC particle that best represents the injected beam particle in this frame.
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

    // Prefer the requested PDG and a particle not created in simulation, but keep sensible
    // fallbacks when the truth record is incomplete.
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

}  // namespace

void start_layer_observed(const char* raw_path_cstr,
                          const char* meta_path_cstr = "",
                          const char* out_path_cstr = "",
                          double start_threshold_GeV = 1e-2,
                          int expected_pdg = 0,
                          const char* hit_collection_cstr = "HCal_Readout") {
  // Validate the main runtime inputs before opening files.
  if (!raw_path_cstr || std::string(raw_path_cstr).empty()) {
    std::cerr << "[start_layer_observed] raw EDM4hep path is required.\n";
    return;
  }

  // Resolve the metadata path, hit collection, and default output location for this run.
  const std::string raw_path(raw_path_cstr);
  const std::string meta_path =
      (meta_path_cstr && std::string(meta_path_cstr).size())
          ? std::string(meta_path_cstr)
          : default_meta_path_from_raw(raw_path);
  const std::string hit_collection =
      (hit_collection_cstr && std::string(hit_collection_cstr).size())
          ? std::string(hit_collection_cstr)
          : std::string("HCal_Readout");

  // Read the processed run metadata so the observed start-layer summary carries the geometry id
  // and nominal gun energy for this sample.
  std::ifstream meta_input(meta_path);
  if (!meta_input) {
    std::cerr << "[start_layer_observed] Failed to open meta.json at " << meta_path << ".\n";
    return;
  }

  nlohmann::json meta_json;
  try {
    meta_input >> meta_json;
  } catch (const std::exception& error) {
    std::cerr << "[start_layer_observed] Failed to parse meta.json: " << error.what() << ".\n";
    return;
  }

  if (!meta_json.contains("geometry_id") || !meta_json["geometry_id"].is_string()) {
    std::cerr << "[start_layer_observed] meta.json is missing geometry_id.\n";
    return;
  }
  if (!meta_json.contains("gun_energy_GeV") || !meta_json["gun_energy_GeV"].is_number()) {
    std::cerr << "[start_layer_observed] meta.json is missing gun_energy_GeV.\n";
    return;
  }

  const std::string geometry_id = meta_json["geometry_id"].get<std::string>();
  const std::string run_id = derive_run_id_from_raw_path(raw_path);
  const double gun_energy_GeV = meta_json["gun_energy_GeV"].get<double>();
  const double mc_rel_diff_limit = 0.1;
  const std::string start_definition = "first_active_layer_above_threshold";
  const std::string out_path =
      (out_path_cstr && std::string(out_path_cstr).size())
          ? std::string(out_path_cstr)
          : default_output_path(geometry_id, run_id);

  // Recover the HCAL segment boundaries before scanning events.
  SegmentBoundaries segment_boundaries;
  if (!load_segment_boundaries(geometry_id, segment_boundaries)) {
    std::cerr << "[start_layer_observed] Failed to resolve segment boundaries for geometry "
              << geometry_id << ".\n";
    return;
  }

  // Open the raw EDM4hep file and choose the category that carries the event frames.
  podio::ROOTReader reader;
  try {
    reader.openFile(raw_path);
  } catch (const std::exception& error) {
    std::cerr << "[start_layer_observed] Failed to open " << raw_path << ": "
              << error.what() << ".\n";
    return;
  }

  std::string category = "events";
  try {
    auto categories = reader.getAvailableCategories();
    if (!categories.empty() &&
        std::find(categories.begin(), categories.end(), "events") == categories.end()) {
      category = categories.front();
    }
  } catch (...) {
  }

  // Walk event by event, identify the primary particle, and then locate the first HCAL layer
  // whose summed simulated energy crosses the chosen start threshold.
  StartLayerStats stats;
  const std::size_t entry_count = reader.getEntries(category);
  for (std::size_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    auto frame_data = reader.readEntry(category, entry_index);
    podio::Frame frame(std::move(frame_data));

    // The observed start-layer summary only makes sense when the event has a usable primary truth particle.
    const edm4hep::MCParticleCollection* mc_collection = nullptr;
    try {
      mc_collection = &frame.get<edm4hep::MCParticleCollection>("MCParticles");
    } catch (...) {
    }
    if (!mc_collection) {
      continue;
    }

    const std::optional<edm4hep::MCParticle> selected_candidate =
        select_primary_candidate(*mc_collection, expected_pdg);
    if (!selected_candidate) {
      continue;
    }

    // Keep only events whose selected primary energy matches the nominal run energy closely enough.
    const double mc_energy_GeV = energy_from_mc(*selected_candidate);
    if (mc_energy_GeV <= 0.0 || gun_energy_GeV <= 0.0) {
      continue;
    }
    const double relative_difference = std::abs(mc_energy_GeV - gun_energy_GeV) / gun_energy_GeV;
    if (relative_difference > mc_rel_diff_limit) {
      continue;
    }

    stats.valid_event_count++;

    // Sum the simulated calorimeter energy layer by layer so the first active layer above
    // threshold can be identified.
    const edm4hep::SimCalorimeterHitCollection* sim_collection = nullptr;
    try {
      sim_collection = &frame.get<edm4hep::SimCalorimeterHitCollection>(hit_collection);
    } catch (...) {
    }
    if (!sim_collection) {
      continue;
    }

    std::vector<double> layer_sim_energy_GeV(
        static_cast<std::size_t>(segment_boundaries.total_layers),
        0.0);
    for (const auto& hit : *sim_collection) {
      const int layer_index = decode_hcal_layer(static_cast<std::uint64_t>(hit.getCellID()));
      if (layer_index < 0 || layer_index >= segment_boundaries.total_layers) {
        std::cerr << "[start_layer_observed] start layer index out of range: "
                  << layer_index << ".\n";
        return;
      }
      layer_sim_energy_GeV[static_cast<std::size_t>(layer_index)] += hit.getEnergy();
    }

    // The observed start layer is the first active HCAL layer whose summed simulated energy
    // exceeds the requested threshold.
    int start_layer = -1;
    for (int layer_index = 0; layer_index < segment_boundaries.total_layers; ++layer_index) {
      if (layer_sim_energy_GeV[static_cast<std::size_t>(layer_index)] > start_threshold_GeV) {
        start_layer = layer_index;
        break;
      }
    }

    if (start_layer < 0) {
      continue;
    }

    // Keep both the start-layer distribution itself and the segment counts used in the summary.
    stats.start_layers.push_back(start_layer);
    if (start_layer < segment_boundaries.seg1_layers) {
      stats.start_in_seg1_count++;
    } else if (start_layer < segment_boundaries.seg1_layers + segment_boundaries.seg2_layers) {
      stats.start_in_seg2_count++;
    } else {
      stats.start_in_seg3_count++;
    }
  }

  // Reduce the event-level counts to the compact summary used for geometry comparisons.
  nlohmann::json output;
  output["geometry_id"] = geometry_id;
  output["run_id"] = run_id;
  output["gun_energy_GeV"] = gun_energy_GeV;
  output["start_definition"] = start_definition;
  output["start_threshold_GeV"] = start_threshold_GeV;
  output["hit_collection"] = hit_collection;
  output["valid_event_count"] = stats.valid_event_count;
  if (!stats.start_layers.empty()) {
    output["start_layer_median"] = median_layer_index(stats.start_layers);
  } else {
    output["start_layer_median"] = nullptr;
  }

  if (stats.valid_event_count > 0) {
    const double valid_event_count = static_cast<double>(stats.valid_event_count);
    output["frac_seg1"] = static_cast<double>(stats.start_in_seg1_count) / valid_event_count;
    output["frac_seg2"] = static_cast<double>(stats.start_in_seg2_count) / valid_event_count;
    output["frac_seg3"] = static_cast<double>(stats.start_in_seg3_count) / valid_event_count;
  } else {
    output["frac_seg1"] = nullptr;
    output["frac_seg2"] = nullptr;
    output["frac_seg3"] = nullptr;
  }

  // Write the observed start-layer summary into the standard geometry-analysis output tree.
  const std::filesystem::path output_file_path(out_path);
  std::filesystem::create_directories(output_file_path.parent_path());
  std::ofstream output_file(out_path);
  if (!output_file) {
    std::cerr << "[start_layer_observed] Failed to open " << out_path << " for writing.\n";
    return;
  }
  output_file << output.dump(2) << '\n';
  std::cout << "[start_layer_observed] Wrote " << out_path << ".\n";
}
