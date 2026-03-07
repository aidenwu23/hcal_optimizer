/* Determine the metric threshold that matches a requested false-positive rate. */
// root -l -b -q 'simulation/calibration/calibrate_fpr.C("path/to/event.root", "path/to/output.json", "visible_E", 0.01)'

#include <TFile.h>
#include <TTree.h>
#include <TTreeFormula.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <limits>

void calibrate_fpr(const char* events_path_cstr, 
                   const char* out_json_cstr, 
                   const char* metric_expr_cstr, 
                   double target_fpr) {
  if (!events_path_cstr || std::string(events_path_cstr).empty()){
    std::cerr << "[calibrate_fpr] events.root path is required.\n";
    return;
  }
  if (!out_json_cstr || std::string(out_json_cstr).empty()){
    std::cerr << "[calibrate_fpr] Out json path is required.\n";
    return;
  }
  if (!metric_expr_cstr || std::string(metric_expr_cstr).empty()){
    std::cerr << "[calibrate_fpr] Metric expression is required.\n";
    return;
  }
  if (target_fpr <= 0.0 || target_fpr >= 1.0){
    std::cerr << "[calibrate_fpr] Target false positive rate should be between 0 and 1.\n";
    return;
  }

  const std::string events_path(events_path_cstr);
  const std::string out_json_path(out_json_cstr);
  const std::string metric_expr(metric_expr_cstr);

  TFile input(events_path.c_str(), "READ");
  if (input.IsZombie()){
    std::cerr << "[calibrate_fpr] Failed to open " << events_path << ".\n";
    return;
  }

  TTree* tree = nullptr;
  input.GetObject("events",tree);
  if (!tree){
    std::cerr << "[calibrate_fpr] events tree not found in " << events_path << ".\n";
    return;
  }

  TTreeFormula formula("metric", metric_expr.c_str(), tree);
  if (formula.GetNdim() <= 0) {
    std::cerr << "[calibrate_fpr] Failed to parse metric expression: " << metric_expr << "\n";
    return;
  }

  std::vector<double> values;
  values.reserve(static_cast<size_t>(tree->GetEntries()));

  const Long64_t n_entries = tree -> GetEntries();
  for (Long64_t i = 0; i < n_entries; ++i) {
    tree->GetEntry(i);
    const double x = formula.EvalInstance();
    if(!std::isfinite(x)){
      continue;
    }
    values.push_back(x);
  }

  const Long64_t n_values = static_cast<Long64_t>(values.size());
  if (n_values == 0){
    std::cerr << "[calibrate_fpr] No valid metric values found in the data" << "\n";
    return;
  }
  std::sort(values.begin(), values.end());
  const Long64_t cutoff_index = static_cast<Long64_t>(std::floor((1.0 - target_fpr) * static_cast<double>(n_values - 1)));
  const Long64_t lower_index = std::max<Long64_t>(0, std::min(n_values - 1, cutoff_index));
  const Long64_t upper_index = std::min(n_values - 1, lower_index + 1);
  const double threshold = 0.5 * (values[static_cast<size_t>(lower_index)] + values[static_cast<size_t>(upper_index)]);

  nlohmann::json output;
  output["muon_threshold_GeV"] = threshold;

  // Flush the run-level record to performance.json beside events.root.
  std::ofstream out(out_json_path);
  if (!out) {
    std::cerr << "[calibrate_fpr] Failed to open " << out_json_path
              << " for writing.\n";
    return;
  }
  out << output.dump(2) << '\n';
  std::cout << "[calibrate_fpr] Wrote " << out_json_path << "\n";
}
