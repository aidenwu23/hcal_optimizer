/*
  Plot the sorted neutron detection efficiency for the initial 150-geometry scan.
  Example: root -l -q 'docs/plots/sorted_eff.C(true)'
*/

#include "ePIC_style.C"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TLine.h"
#include "TMarker.h"


struct EfficiencyPoint
{
  std::string geometry_id;
  double efficiency;
};


//=====================
void sorted_eff(bool save_plot = false,
                bool add_epic_logo = false)
{
  // Load the initial 150-point compact scan.
  std::ifstream training_input("docs/plots/data/training_compact_0.csv");
  std::string line;
  std::vector<EfficiencyPoint> points;

  if (!training_input.is_open())
    {
      Error("sorted_eff", "Could not open docs/plots/data/training_compact_0.csv");
      return;
    }

  std::getline(training_input, line);

  while (std::getline(training_input, line))
    {
      if (line.empty())
        {
          continue;
        }

      std::stringstream line_stream(line);
      std::string value;
      std::vector<std::string> columns;

      while (std::getline(line_stream, value, ','))
        {
          columns.push_back(value);
        }

      if (columns.size() <= 12 || columns[12].empty())
        {
          continue;
        }

      points.push_back({columns[0], std::stod(columns[12])});
    }

  // Load the repeated-run baseline reference efficiency.
  std::ifstream baseline_input("csv_data/results/baseline_compact.csv");
  double baseline_efficiency = 0.0;

  if (!baseline_input.is_open())
    {
      Error("sorted_eff", "Could not open csv_data/results/baseline_compact.csv");
      return;
    }

  std::getline(baseline_input, line);
  std::getline(baseline_input, line);

  {
    std::stringstream line_stream(line);
    std::string value;
    std::vector<std::string> columns;

    while (std::getline(line_stream, value, ','))
      {
        columns.push_back(value);
      }

    if (columns.size() <= 12 || columns[12].empty())
      {
        Error("sorted_eff", "Baseline CSV is missing neutron_efficiency");
        return;
      }

    baseline_efficiency = std::stod(columns[12]);
  }

  std::sort(points.begin(), points.end(),
            [](const EfficiencyPoint& left, const EfficiencyPoint& right)
            {
              return left.efficiency < right.efficiency;
            });

  gROOT->ProcessLine("set_ePIC_style()");

  const int n_points = points.size();
  std::vector<double> ranks(n_points);
  std::vector<double> efficiencies(n_points);

  int confirmed_optimum_rank = -1;
  int best_low_stat_rank = n_points;

  for (int i = 0; i < n_points; ++i)
    {
      ranks[i] = i + 1;
      efficiencies[i] = points[i].efficiency;

      if (points[i].geometry_id == "1fa19adc")
        {
          confirmed_optimum_rank = i + 1;
        }
    }

  TGraph* graph = new TGraph(n_points, ranks.data(), efficiencies.data());
  graph->SetName("graph_sorted_efficiency");
  graph->SetTitle(";Geometry rank in initial 150-point scan;Neutron detection efficiency");
  graph->SetMarkerStyle(20);
  graph->SetMarkerSize(0.8);
  graph->SetMarkerColor(kBlack);

  TCanvas* canvas = new TCanvas("canvas_sorted_efficiency", "", 800, 600);
  canvas->cd();

  graph->Draw("AP");
  graph->GetXaxis()->SetLimits(1, n_points);
  graph->SetMinimum(0.508);
  graph->SetMaximum(0.552);

  TLine* baseline_line = new TLine(1, baseline_efficiency, n_points, baseline_efficiency);
  baseline_line->SetLineColor(kBlue + 1);
  baseline_line->SetLineWidth(2);
  baseline_line->SetLineStyle(2);
  baseline_line->Draw();

  TMarker* confirmed_optimum_marker = nullptr;

  if (confirmed_optimum_rank > 0)
    {
      confirmed_optimum_marker = new TMarker(confirmed_optimum_rank, points[confirmed_optimum_rank - 1].efficiency, 29);
      confirmed_optimum_marker->SetMarkerColor(kRed + 1);
      confirmed_optimum_marker->SetMarkerSize(1.6);
      confirmed_optimum_marker->Draw();
    }

  TMarker* best_low_stat_marker = new TMarker(best_low_stat_rank, points.back().efficiency, 33);
  best_low_stat_marker->SetMarkerColor(kGreen + 2);
  best_low_stat_marker->SetMarkerSize(1.6);
  best_low_stat_marker->Draw();

  TLegend* legend = new TLegend(0.56, 0.70, 0.90, 0.88);
  legend->AddEntry(graph, "Initial scan", "P");
  legend->AddEntry(baseline_line, "Baseline", "L");
  if (confirmed_optimum_marker != nullptr)
    {
      legend->AddEntry(confirmed_optimum_marker, "Confirmed optimum", "P");
    }
  legend->AddEntry(best_low_stat_marker, "Top scan point", "P");
  legend->Draw();

  TLatex text_epic;
  text_epic.SetTextSize(0.05);
  text_epic.SetTextFont(62);
  text_epic.DrawLatexNDC(.15,.88,"Detector Performance");

  TLatex text_com;
  text_com.SetTextSize(0.038);
  text_com.SetTextAlign(13);
  text_com.DrawLatexNDC(.15,.85,"Initial 150-geometry Latin-hypercube scan");
  text_com.DrawLatexNDC(.15,.80,"Baseline shown for comparison");

  if(add_epic_logo)
    {
      TImage *logo = TImage::Open("docs/plots/EPIC-logo_black_small.png");
      TPad *pad2 = new TPad("pad2", "Pad 2", 0.8, 0.8, 0.93, 0.93);
      pad2->Draw();
      pad2->cd();
      logo->Draw();
    }

  if(save_plot)
    {
      TFile* output = new TFile("docs/plots/sorted_eff.root", "RECREATE");
      canvas->Write();
      output->Close();
    }
}
