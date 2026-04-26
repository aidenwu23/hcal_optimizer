/*
Scan baseline and optimum hit-activity metrics as a function of the MIP threshold fraction.
Example: 
root -l -q 'docs/plots/threshold_scan.C()'
*/

#include "ePIC_style.C"

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TTree.h"

namespace {

constexpr int kLayerCount = 10;
constexpr int kSegmentCount = 3;

struct ScanPoint
{
  double alpha = 0.0;
  double efficiency = 0.0;
  double tiles_mean = 0.0;
  double layers_mean = 0.0;
};

struct ScanResult
{
  std::string label;
  std::vector<ScanPoint> points;
  TGraph* efficiency_graph = nullptr;
  TGraph* tiles_graph = nullptr;
  TGraph* layers_graph = nullptr;
  TCanvas* canvas = nullptr;
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

std::array<double, kSegmentCount> segment_mpvs(double alpha,
                                               const std::array<double, kSegmentCount>& thresholds)
{
  std::array<double, kSegmentCount> mpvs {};
  for (int index = 0; index < kSegmentCount; ++index)
    {
      mpvs[static_cast<std::size_t>(index)] = thresholds[static_cast<std::size_t>(index)] / alpha;
    }
  return mpvs;
}

// Count detected events and multiplicities at one MIP-fraction threshold.
ScanPoint scan_alpha(TTree& tree,
                     std::array<std::vector<float>*, kLayerCount>& layer_cell_E,
                     float& mc_E,
                     const std::array<double, kSegmentCount>& mpvs,
                     double alpha)
{
  long long valid_count = 0;
  long long detected_count = 0;
  double tile_sum = 0.0;
  double layer_sum = 0.0;

  const Long64_t entry_count = tree.GetEntries();
  for (Long64_t entry_index = 0; entry_index < entry_count; ++entry_index)
    {
      tree.GetEntry(entry_index);
      if (mc_E <= 0.0F)
        {
          continue;
        }

      valid_count++;

      int fired_cell_count = 0;
      int fired_layer_count = 0;
      for (int layer_index = 0; layer_index < kLayerCount; ++layer_index)
        {
          const int segment_index = layer_to_segment(layer_index);
          const double threshold = alpha * mpvs[static_cast<std::size_t>(segment_index)];
          const auto* cell_energies = layer_cell_E[static_cast<std::size_t>(layer_index)];
          if (!cell_energies)
            {
              continue;
            }

          bool layer_has_fired_cell = false;
          for (const float cell_energy : *cell_energies)
            {
              if (static_cast<double>(cell_energy) >= threshold)
                {
                  fired_cell_count++;
                  layer_has_fired_cell = true;
                }
            }
          if (layer_has_fired_cell)
            {
              fired_layer_count++;
            }
        }

      tile_sum += static_cast<double>(fired_cell_count);
      layer_sum += static_cast<double>(fired_layer_count);
      if (fired_cell_count > 0)
        {
          detected_count++;
        }
    }

  ScanPoint point;
  point.alpha = alpha;
  if (valid_count > 0)
    {
      point.efficiency = static_cast<double>(detected_count) / static_cast<double>(valid_count);
      point.tiles_mean = tile_sum / static_cast<double>(valid_count);
      point.layers_mean = layer_sum / static_cast<double>(valid_count);
    }
  return point;
}

void style_graph(TGraph* graph, int color, int marker_style)
{
  graph->SetLineColor(color);
  graph->SetLineWidth(2);
  graph->SetMarkerColor(color);
  graph->SetMarkerStyle(marker_style);
  graph->SetMarkerSize(0.55);
}

TGraph* make_graph(const std::vector<ScanPoint>& points,
                   const std::string& name,
                   const std::string& y_title,
                   double ScanPoint::*value_member,
                   int color,
                   int marker_style)
{
  std::vector<double> alphas;
  std::vector<double> values;
  alphas.reserve(points.size());
  values.reserve(points.size());
  for (const ScanPoint& point : points)
    {
      alphas.push_back(point.alpha);
      values.push_back(point.*value_member);
    }

  TGraph* graph = new TGraph(points.size(), alphas.data(), values.data());
  graph->SetName(name.c_str());
  graph->SetTitle((";MIP threshold fraction;" + y_title).c_str());
  style_graph(graph, color, marker_style);
  return graph;
}

ScanResult scan_file(const std::string& label,
                     const std::string& events_path,
                     const std::array<double, kSegmentCount>& mpvs,
                     double alpha_min,
                     double alpha_max,
                     int alpha_count,
                     int color,
                     int marker_style)
{
  ScanResult result;
  result.label = label;

  TFile input_file(events_path.c_str(), "READ");
  if (input_file.IsZombie())
    {
      Error("threshold_scan", "Could not open %s", events_path.c_str());
      return result;
    }

  TTree* tree = nullptr;
  input_file.GetObject("events", tree);
  if (!tree)
    {
      Error("threshold_scan", "Tree 'events' not found in %s", events_path.c_str());
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
          Error("threshold_scan", "Branch %s not found", branch_name.c_str());
          return result;
        }
      tree->SetBranchAddress(branch_name.c_str(), &layer_cell_E[static_cast<std::size_t>(layer_index)]);
    }

  result.points.reserve(static_cast<std::size_t>(alpha_count));
  for (int index = 0; index < alpha_count; ++index)
    {
      const double fraction = alpha_count > 1 ? static_cast<double>(index) / static_cast<double>(alpha_count - 1) : 0.0;
      const double alpha = alpha_min + fraction * (alpha_max - alpha_min);
      result.points.push_back(scan_alpha(*tree, layer_cell_E, mc_E, mpvs, alpha));
    }

  result.efficiency_graph = make_graph(
      result.points, ("graph_efficiency_" + label).c_str(), "Neutron detection efficiency",
      &ScanPoint::efficiency, color, marker_style);
  result.tiles_graph = make_graph(
      result.points, ("graph_tiles_" + label).c_str(), "Average fired tiles",
      &ScanPoint::tiles_mean, color, marker_style);
  result.layers_graph = make_graph(
      result.points, ("graph_layers_" + label).c_str(), "Average fired layers",
      &ScanPoint::layers_mean, color, marker_style);

  result.canvas = new TCanvas(("canvas_threshold_scan_" + label).c_str(), "", 800, 600);
  result.canvas->cd();
  result.efficiency_graph->Draw("APL");
  result.efficiency_graph->GetXaxis()->SetLimits(0.0, 1.0);
  result.efficiency_graph->SetMinimum(0.4);
  result.efficiency_graph->SetMaximum(1.0);

  TLatex text_epic;
  text_epic.SetTextSize(0.05);
  text_epic.SetTextFont(62);
  text_epic.DrawLatexNDC(.15,.88,"Efficiency across thresholds");

  TLatex text_com;
  text_com.SetTextSize(0.038);
  text_com.SetTextAlign(13);
  text_com.DrawLatexNDC(.15,.83,(label + " comparison").c_str());

  input_file.Close();
  return result;
}

}  // namespace

void threshold_scan(const char* baseline_events_path = "docs/plots/data/baseline_20k.root",
                    const char* optimum_events_path = "docs/plots/data/optimum_20k.root",
                    const char* output_path = "docs/plots/threshold_scan.root",
                    double alpha_min = 0.0,
                    double alpha_max = 1.0,
                    int alpha_count = 51)
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

  const double reference_alpha = 0.5;
  const auto baseline_mpvs = segment_mpvs(reference_alpha, baseline_thresholds);
  const auto optimum_mpvs = segment_mpvs(reference_alpha, optimum_thresholds);

  ScanResult baseline = scan_file(
      "baseline", baseline_events_path, baseline_mpvs, alpha_min, alpha_max, alpha_count, kBlue + 1, 20);
  ScanResult optimum = scan_file(
      "optimum", optimum_events_path, optimum_mpvs, alpha_min, alpha_max, alpha_count, kRed + 1, 21);

  TCanvas* overlay_canvas = new TCanvas("canvas_threshold_scan_overlay", "", 800, 600);
  overlay_canvas->cd();
  baseline.efficiency_graph->Draw("APL");
  baseline.efficiency_graph->GetXaxis()->SetLimits(0.0, 1.0);
  baseline.efficiency_graph->SetMinimum(0.4);
  baseline.efficiency_graph->SetMaximum(1.0);
  optimum.efficiency_graph->Draw("PL same");

  TLegend* legend = new TLegend(0.56, 0.72, 0.88, 0.86);
  legend->SetTextSize(0.032);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->AddEntry(baseline.efficiency_graph, "Baseline", "PL");
  legend->AddEntry(optimum.efficiency_graph, "Optimum", "PL");
  legend->Draw();

  TLatex text_epic;
  text_epic.SetTextSize(0.05);
  text_epic.SetTextFont(62);
  text_epic.DrawLatexNDC(.15,.88,"Efficiency across thresholds");

  TLatex text_com;
  text_com.SetTextSize(0.038);
  text_com.SetTextAlign(13);
  text_com.DrawLatexNDC(.15,.83,"Baseline vs optimum");

  TFile output_file(output_path, "RECREATE");
  baseline.efficiency_graph->Write();
  baseline.tiles_graph->Write();
  baseline.layers_graph->Write();
  baseline.canvas->Write();
  optimum.efficiency_graph->Write();
  optimum.tiles_graph->Write();
  optimum.layers_graph->Write();
  optimum.canvas->Write();
  overlay_canvas->Write();
  output_file.Close();

  std::cout << "[threshold_scan] Wrote " << output_path << ".\n";
}
