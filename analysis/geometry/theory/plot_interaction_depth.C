/*
Write the geometry-only interaction-depth ROOT objects for one geometry.

Example:
root -l -b -q 'analysis/geometry/theory/plot_interaction_depth.C("81c3da7d","")'
*/

#include <TFile.h>
#include <TGraph.h>
#include <TH1D.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct LayerRow {
  int layer_index = -1;
  double depth_back_mm = 0.0;
  double delta_tau_layer = 0.0;
  double cumulative_tau = 0.0;
  double cumulative_probability = 0.0;
  double lambda_I_eff_layer_mm = 0.0;
};

// Trim CSV fields before converting them into numbers.
std::string trim_copy(std::string text) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
  text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
  return text;
}

// The Python analysis output is a simple CSV, so a lightweight comma split is enough here.
std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(trim_copy(field));
  }
  return fields;
}

std::string string_arg_or_default(const char* value, const std::string& fallback) {
  if (value && std::string(value).size()) {
    return std::string(value);
  }
  return fallback;
}

// Resolve paths relative to the repo root so the macro can find the geometry-analysis outputs.
std::string project_root_from_macro() {
  std::filesystem::path macro_path(__FILE__);
  return macro_path.parent_path().parent_path().parent_path().parent_path().string();
}

std::string layers_csv_path(const std::string& geometry_id) {
  const std::filesystem::path project_root(project_root_from_macro());
  return (project_root / "data" / "geometry_analysis" / geometry_id / "layers.csv").string();
}

std::string default_output_path(const std::string& geometry_id) {
  const std::filesystem::path project_root(project_root_from_macro());
  return (
      project_root /
      "data" /
      "geometry_analysis" /
      geometry_id /
      (geometry_id + "_interaction_depth.root")).string();
}

// Read the per-layer interaction curve written by interaction_depth.py.
bool read_layers_csv(const std::string& geometry_id, std::vector<LayerRow>& rows) {
  std::ifstream input(layers_csv_path(geometry_id));
  if (!input) {
    return false;
  }

  std::string header_line;
  if (!std::getline(input, header_line)) {
    return false;
  }

  const std::vector<std::string> header_fields = split_csv_line(header_line);
  std::map<std::string, std::size_t> column_index;
  for (std::size_t index = 0; index < header_fields.size(); ++index) {
    column_index[header_fields[index]] = index;
  }

  const char* required_columns[] = {
      "layer_index",
      "depth_back_mm",
      "delta_tau_layer",
      "cumulative_tau",
      "cumulative_probability",
      "lambda_I_eff_layer_mm",
  };
  for (const char* column_name : required_columns) {
    if (column_index.find(column_name) == column_index.end()) {
      return false;
    }
  }

  std::string line;
  // Read each layer row and convert the required CSV fields into numbers.
  while (std::getline(input, line)) {
    if (trim_copy(line).empty()) {
      continue;
    }

    const std::vector<std::string> fields = split_csv_line(line);
    for (const char* column_name : required_columns) {
      if (column_index[column_name] >= fields.size()) {
        return false;
      }
    }

    LayerRow row;
    try {
      row.layer_index = std::stoi(fields[column_index["layer_index"]]);
      row.depth_back_mm = std::stod(fields[column_index["depth_back_mm"]]);
      row.delta_tau_layer = std::stod(fields[column_index["delta_tau_layer"]]);
      row.cumulative_tau = std::stod(fields[column_index["cumulative_tau"]]);
      row.cumulative_probability = std::stod(fields[column_index["cumulative_probability"]]);
      row.lambda_I_eff_layer_mm = std::stod(fields[column_index["lambda_I_eff_layer_mm"]]);
      rows.push_back(row);
    } catch (const std::exception&) {
      return false;
    }
  }

  return !rows.empty();
}

}  // namespace

void plot_interaction_depth(const char* geometry_id_cstr,
                            const char* out_root_cstr = "") {
  if (!geometry_id_cstr || std::string(geometry_id_cstr).empty()) {
    std::cerr << "[plot_interaction_depth] geometry_id is required.\n";
    return;
  }

  const std::string geometry_id = trim_copy(std::string(geometry_id_cstr));
  const std::string out_root_path =
      string_arg_or_default(out_root_cstr, default_output_path(geometry_id));

  // Load the geometry-only interaction curve written by the Python analysis.
  std::vector<LayerRow> layer_rows;
  if (!read_layers_csv(geometry_id, layer_rows)) {
    std::cerr << "[plot_interaction_depth] failed to read layers for " << geometry_id << ".\n";
    return;
  }

  const std::filesystem::path output_path(out_root_path);
  std::filesystem::create_directories(output_path.parent_path());

  TFile output_file(out_root_path.c_str(), "RECREATE");
  if (output_file.IsZombie()) {
    std::cerr << "[plot_interaction_depth] failed to open " << out_root_path << " for writing.\n";
    return;
  }

  // Separate the depth, layer, and lambda views in the ROOT output.
  TDirectory* depth_directory = output_file.mkdir("depth");
  TDirectory* layer_directory = output_file.mkdir("layer");
  TDirectory* lambda_directory = output_file.mkdir("lambda");

  // Build the depth-based interaction plots.
  TGraph p_interact_vs_depth_mm(static_cast<int>(layer_rows.size()));
  p_interact_vs_depth_mm.SetName("p_interact_vs_depth_mm");
  p_interact_vs_depth_mm.SetTitle("Cumulative interaction probability;Depth [mm];P_{interact}");
  p_interact_vs_depth_mm.SetLineWidth(2);

  TGraph cumulative_tau_vs_depth_mm(static_cast<int>(layer_rows.size()));
  cumulative_tau_vs_depth_mm.SetName("cumulative_tau_vs_depth_mm");
  cumulative_tau_vs_depth_mm.SetTitle("Cumulative hadronic depth;Depth [mm];Cumulative #tau");
  cumulative_tau_vs_depth_mm.SetLineWidth(2);

  // Build the layer-index plots.
  TGraph p_interact_vs_layer(static_cast<int>(layer_rows.size()));
  p_interact_vs_layer.SetName("p_interact_vs_layer");
  p_interact_vs_layer.SetTitle("Cumulative interaction probability;Layer index;P_{interact}");
  p_interact_vs_layer.SetLineWidth(2);

  // Build the cumulative lambda_I plots.
  TGraph p_interact_vs_lambda(static_cast<int>(layer_rows.size()));
  p_interact_vs_lambda.SetName("p_interact_vs_lambda");
  p_interact_vs_lambda.SetTitle("Cumulative interaction probability;Cumulative #lambda_{I};P_{interact}");
  p_interact_vs_lambda.SetLineWidth(2);

  TGraph p_survive_vs_lambda(static_cast<int>(layer_rows.size()));
  p_survive_vs_lambda.SetName("p_survive_vs_lambda");
  p_survive_vs_lambda.SetTitle("Survival probability;Cumulative #lambda_{I};P_{survive}");
  p_survive_vs_lambda.SetLineWidth(2);

  // Build the per-layer histogram summaries.
  TH1D lambda_eff_by_layer(
      "lambda_eff_by_layer",
      "Layer effective interaction length;Layer index;#lambda_{I,eff} [mm]",
      static_cast<int>(layer_rows.size()),
      -0.5,
      static_cast<double>(layer_rows.size()) - 0.5);
  lambda_eff_by_layer.SetLineWidth(2);
  lambda_eff_by_layer.SetStats(false);
  lambda_eff_by_layer.SetDirectory(nullptr);

  // This histogram shows the hadronic optical-depth increment contributed by each built layer.
  TH1D delta_tau_layer_by_layer(
      "delta_tau_layer_by_layer",
      "Per-layer hadronic depth increment;Layer index;#Delta#tau_{layer}",
      static_cast<int>(layer_rows.size()),
      -0.5,
      static_cast<double>(layer_rows.size()) - 0.5);
  delta_tau_layer_by_layer.SetLineWidth(2);
  delta_tau_layer_by_layer.SetStats(false);
  delta_tau_layer_by_layer.SetDirectory(nullptr);

  // Fill every graph and histogram from the per-layer interaction table.
  for (std::size_t row_index = 0; row_index < layer_rows.size(); ++row_index) {
    const LayerRow& row = layer_rows[row_index];
    p_interact_vs_depth_mm.SetPoint(
        static_cast<int>(row_index),
        row.depth_back_mm,
        row.cumulative_probability);
    cumulative_tau_vs_depth_mm.SetPoint(
        static_cast<int>(row_index),
        row.depth_back_mm,
        row.cumulative_tau);
    p_interact_vs_layer.SetPoint(
        static_cast<int>(row_index),
        static_cast<double>(row.layer_index),
        row.cumulative_probability);
    p_interact_vs_lambda.SetPoint(
        static_cast<int>(row_index),
        row.cumulative_tau,
        row.cumulative_probability);
    p_survive_vs_lambda.SetPoint(
        static_cast<int>(row_index),
        row.cumulative_tau,
        1.0 - row.cumulative_probability);
    delta_tau_layer_by_layer.SetBinContent(static_cast<int>(row_index) + 1, row.delta_tau_layer);
    delta_tau_layer_by_layer.GetXaxis()->SetBinLabel(
        static_cast<int>(row_index) + 1,
        Form("%d", row.layer_index));
    lambda_eff_by_layer.SetBinContent(static_cast<int>(row_index) + 1, row.lambda_I_eff_layer_mm);
    lambda_eff_by_layer.GetXaxis()->SetBinLabel(
        static_cast<int>(row_index) + 1,
        Form("%d", row.layer_index));
  }

  depth_directory->cd();
  p_interact_vs_depth_mm.Write();
  cumulative_tau_vs_depth_mm.Write();

  layer_directory->cd();
  p_interact_vs_layer.Write();
  delta_tau_layer_by_layer.Write();
  lambda_eff_by_layer.Write();

  lambda_directory->cd();
  p_interact_vs_lambda.Write();
  p_survive_vs_lambda.Write();
  output_file.Close();

  std::cout << "[plot_interaction_depth] Wrote " << out_root_path << ".\n";
}
