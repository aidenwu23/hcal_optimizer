/*
Plot neutron detection efficiency versus kinetic energy for baseline and optimum samples.
Example: 

root -l -q 'docs/plots/efficiency_vs_kinetic_energy.C()'
*/

#include "ePIC_style.C"

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TTree.h"

namespace {

constexpr int kLayerCount = 10;
constexpr int kSegmentCount = 3;
constexpr double kNeutronMassGeV = 0.93956542052;

struct EfficiencyPoint
{
  double x = 0.0;
  double x_error = 0.0;
  double efficiency = 0.0;
  double efficiency_error = 0.0;
};

struct SampleResult
{
  std::string label;
  std::vector<EfficiencyPoint> points;
  TGraphErrors* graph = nullptr;
  TGraphErrors* graph_errors = nullptr;
};

int layer_to_segment(int layer_index)
{
  if (layer_index < 3)
    {
      return 0;
    }
  if (layer_index < 6)
    {
      return 1;
    }
  return 2;
}

bool event_detected(std::array<std::vector<float>*, kLayerCount>& layer_cell_E,
                    const std::array<double, kSegmentCount>& thresholds)
{
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index)
    {
      const int segment_index = layer_to_segment(layer_index);
      const auto* cell_energies = layer_cell_E[static_cast<std::size_t>(layer_index)];
      if (!cell_energies)
        {
          continue;
        }

      for (const float cell_energy : *cell_energies)
        {
          if (static_cast<double>(cell_energy) >= thresholds[static_cast<std::size_t>(segment_index)])
            {
              return true;
            }
        }
    }
  return false;
}

void style_graph(TGraphErrors* graph, int color, int marker_style)
{
  graph->SetLineColor(color);
  graph->SetLineWidth(2);
  graph->SetMarkerColor(color);
  graph->SetMarkerStyle(marker_style);
  graph->SetMarkerSize(1.3);
}

SampleResult scan_sample(const std::string& label,
                         const std::string& events_path,
                         const std::array<double, kSegmentCount>& thresholds,
                         const std::vector<double>& bin_edges,
                         int color,
                         int marker_style)
{
  SampleResult result;
  result.label = label;

  TFile input_file(events_path.c_str(), "READ");
  if (input_file.IsZombie())
    {
      Error("efficiency_vs_kinetic_energy", "Could not open %s", events_path.c_str());
      return result;
    }

  TTree* tree = nullptr;
  input_file.GetObject("events", tree);
  if (!tree)
    {
      Error("efficiency_vs_kinetic_energy", "Tree 'events' not found in %s", events_path.c_str());
      return result;
    }

  float mc_E = 0.0F;
  std::array<std::vector<float>*, kLayerCount> layer_cell_E {};
  tree->SetBranchAddress("mc_E", &mc_E);
  for (int layer_index = 0; layer_index < kLayerCount; ++layer_index)
    {
      const std::string branch_name = "layer_" + std::to_string(layer_index) + "_cell_E";
      if (tree->GetBranch(branch_name.c_str()) == nullptr)
        {
          Error("efficiency_vs_kinetic_energy", "Branch %s not found", branch_name.c_str());
          return result;
        }
      tree->SetBranchAddress(branch_name.c_str(), &layer_cell_E[static_cast<std::size_t>(layer_index)]);
    }

  const int bin_count = static_cast<int>(bin_edges.size()) - 1;
  std::vector<long long> valid_counts(static_cast<std::size_t>(bin_count), 0);
  std::vector<long long> detected_counts(static_cast<std::size_t>(bin_count), 0);

  const Long64_t entry_count = tree->GetEntries();
  for (Long64_t entry_index = 0; entry_index < entry_count; ++entry_index)
    {
      tree->GetEntry(entry_index);
      if (mc_E <= 0.0F)
        {
          continue;
        }

      const double kinetic_energy_GeV = static_cast<double>(mc_E) - kNeutronMassGeV;
      if (kinetic_energy_GeV < bin_edges.front() || kinetic_energy_GeV >= bin_edges.back())
        {
          continue;
        }

      int bin_index = -1;
      for (int index = 0; index < bin_count; ++index)
        {
          if (kinetic_energy_GeV >= bin_edges[static_cast<std::size_t>(index)] &&
              kinetic_energy_GeV < bin_edges[static_cast<std::size_t>(index + 1)])
            {
              bin_index = index;
              break;
            }
        }
      if (bin_index < 0)
        {
          continue;
        }

      valid_counts[static_cast<std::size_t>(bin_index)]++;
      if (event_detected(layer_cell_E, thresholds))
        {
          detected_counts[static_cast<std::size_t>(bin_index)]++;
        }
    }

  for (int index = 0; index < bin_count; ++index)
    {
      const long long valid_count = valid_counts[static_cast<std::size_t>(index)];
      if (valid_count <= 0)
        {
          continue;
        }

      const double detected_count = static_cast<double>(detected_counts[static_cast<std::size_t>(index)]);
      const double valid_count_double = static_cast<double>(valid_count);
      const double efficiency = detected_count / valid_count_double;
      const double error = std::sqrt(efficiency * (1.0 - efficiency) / valid_count_double);
      const double low_edge = bin_edges[static_cast<std::size_t>(index)];
      const double high_edge = bin_edges[static_cast<std::size_t>(index + 1)];

      result.points.push_back({
          0.5 * (low_edge + high_edge),
          0.5 * (high_edge - low_edge),
          efficiency,
          error,
      });
    }

  std::vector<double> x_values;
  std::vector<double> x_errors;
  std::vector<double> y_values;
  std::vector<double> y_errors;
  for (const EfficiencyPoint& point : result.points)
    {
      x_values.push_back(point.x);
      x_errors.push_back(point.x_error);
      y_values.push_back(point.efficiency);
      y_errors.push_back(point.efficiency_error);
    }

  std::vector<double> zero_errors(x_errors.size(), 0.0);

  result.graph = new TGraphErrors(
      result.points.size(), x_values.data(), y_values.data(), zero_errors.data(), zero_errors.data());
  result.graph->SetName(("graph_efficiency_vs_kinetic_energy_" + label).c_str());
  result.graph->SetTitle(";Neutron kinetic energy [GeV];Neutron detection efficiency");
  style_graph(result.graph, color, marker_style);

  result.graph_errors = new TGraphErrors(
      result.points.size(), x_values.data(), y_values.data(), zero_errors.data(), y_errors.data());
  result.graph_errors->SetName(("graph_efficiency_vs_kinetic_energy_errors_" + label).c_str());
  result.graph_errors->SetTitle(";Neutron kinetic energy [GeV];Neutron detection efficiency");
  style_graph(result.graph_errors, color, marker_style);
  result.graph_errors->SetLineWidth(1);

  input_file.Close();
  return result;
}

}  // namespace

void efficiency_vs_kinetic_energy(
    const char* baseline_events_path = "docs/plots/data/baseline_20k.root",
    const char* optimum_events_path = "docs/plots/data/optimum_20k.root",
    const char* output_path = "docs/plots/efficiency_vs_kinetic_energy.root")
{
  gROOT->ProcessLine("set_ePIC_style()");

  const std::array<double, kSegmentCount> baseline_thresholds {{
      0.00030921430275800485,
      0.00031005075640097585,
      0.000309692606906191,
  }};
  const std::array<double, kSegmentCount> optimum_thresholds {{
      0.00039665340244560064,
      0.00036116950517068734,
      0.000371856516514473,
  }};

  const std::vector<double> bin_edges {{
      0.005, 0.008, 0.013, 0.021, 0.034, 0.055, 0.089, 0.144,
      0.233, 0.377, 0.611, 0.989, 1.60,
  }};

  SampleResult baseline = scan_sample(
      "baseline", baseline_events_path, baseline_thresholds, bin_edges, kBlue + 1, 20);
  SampleResult optimum = scan_sample(
      "optimum", optimum_events_path, optimum_thresholds, bin_edges, kRed + 1, 21);

  TCanvas* canvas = new TCanvas("canvas_efficiency_vs_kinetic_energy", "", 800, 600);
  canvas->cd();
  canvas->SetLogx();
  baseline.graph->Draw("AP");
  baseline.graph->GetXaxis()->SetLimits(0.005, 1.7);
  baseline.graph->SetMinimum(0.0);
  baseline.graph->SetMaximum(1.0);
  optimum.graph->Draw("P same");

  TCanvas* canvas_errors = new TCanvas("canvas_efficiency_vs_kinetic_energy_errors", "", 800, 600);
  canvas_errors->cd();
  canvas_errors->SetLogx();
  baseline.graph_errors->Draw("APZ");
  baseline.graph_errors->GetXaxis()->SetLimits(0.005, 1.7);
  baseline.graph_errors->SetMinimum(0.0);
  baseline.graph_errors->SetMaximum(1.0);
  optimum.graph_errors->Draw("PZ same");

  canvas->cd();

  TLegend* legend = new TLegend(0.56, 0.72, 0.88, 0.86);
  legend->SetTextSize(0.032);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->AddEntry(baseline.graph, "Baseline", "P");
  legend->AddEntry(optimum.graph, "Optimum", "P");
  legend->Draw();

  TLatex text_epic;
  text_epic.SetTextSize(0.05);
  text_epic.SetTextFont(62);
  text_epic.DrawLatexNDC(.15,.88,"Efficiency across neutron energies");

  TLatex text_com;
  text_com.SetTextSize(0.038);
  text_com.SetTextAlign(13);
  text_com.DrawLatexNDC(.15,.83,"Baseline vs optimum");

  TFile output_file(output_path, "RECREATE");
  baseline.graph->Write();
  baseline.graph_errors->Write();
  optimum.graph->Write();
  optimum.graph_errors->Write();
  canvas->Write();
  canvas_errors->Write();
  output_file.Close();

  std::cout << "[efficiency_vs_kinetic_energy] Wrote " << output_path << ".\n";
}
