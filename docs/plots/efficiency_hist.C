/*
  Plot the neutron detection efficiency histogram for the initial 150-geometry scan.
  Example: root -l -q 'docs/plots/efficiency_hist.C(true)'
*/

#include "ePIC_style.C"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "TFile.h"


//=====================
void efficiency_hist(bool save_plot = false,
                     bool add_epic_logo = false)
{
  // Load the initial 150-point compact scan efficiencies.
  std::ifstream input("docs/plots/data/training_compact_0.csv");
  std::string line;
  std::vector<double> efficiencies;

  if (!input.is_open())
    {
      Error("efficiency_hist", "Could not open docs/plots/data/training_compact_0.csv");
      return;
    }

  std::getline(input, line);

  while (std::getline(input, line))
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

      efficiencies.push_back(std::stod(columns[12]));
    }

  // Apply the common ePIC plot style and fill the histogram.
  gROOT->ProcessLine("set_ePIC_style()");

  const int nBins = 10;
  const double xMin = 0.508;
  const double xMax = 0.552;

  TH1F* hist_efficiency = new TH1F("hist_efficiency", "", nBins, xMin, xMax);

  for (double efficiency : efficiencies)
    {
      hist_efficiency->Fill(efficiency);
    }

  TCanvas *canvas = new TCanvas("canvas", "", 800, 600);
  canvas->cd();

  hist_efficiency->SetTitle(";Neutron detection efficiency;Number of geometries");
  hist_efficiency->SetLineColor(kBlack);
  hist_efficiency->SetFillColor(kAzure - 9);
  hist_efficiency->SetMaximum(hist_efficiency->GetMaximum() * 1.2);
  hist_efficiency->Draw("hist");

  TLatex Text_com;
  Text_com.SetTextAlign(13);
  Text_com.DrawLatexNDC(.15,.84,"Neutron efficiency distribution for the");
  Text_com.DrawLatexNDC(.15,.79,"initial 150-geometry scan");

  TLatex Text_ePIC;
  Text_ePIC.SetTextSize(0.05);
  Text_ePIC.SetTextFont(62);
  Text_ePIC.DrawLatexNDC(.15,.88,"Detector Performance");

  if(add_epic_logo)
    {
      TImage *logo = TImage::Open("EPIC-logo_black_small.png");
      TPad *pad2 = new TPad("pad2", "Pad 2", 0.8, 0.8, 0.93, 0.93);
      pad2->Draw();
      pad2->cd();
      logo->Draw();
    }

  if(save_plot)
    {
      TFile* output = new TFile("docs/plots/efficiency_hist.root", "RECREATE");
      hist_efficiency->Write();
      canvas->Write();
      output->Close();
    }
}
