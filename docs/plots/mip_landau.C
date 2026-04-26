/*
  Plot the baseline muon MIP calibration spectrum used for threshold setting.
  Example: 
  root -l -q 'docs/plots/mip_landau.C(true)'
*/

#include "ePIC_style.C"

#include <cmath>
#include <string>

#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TH1D.h"
#include "TImage.h"
#include "TLegend.h"
#include "TLine.h"
#include "TPad.h"
#include "TLatex.h"

namespace {

constexpr double kLandauMpvShift = 0.22278298;

// Return the MPV used by the calibration script from the saved Landau fit.
double fitted_mpv(const TF1& fit)
{
  const double mu = fit.GetParameter(1);
  const double sigma = fit.GetParameter(2);
  return mu - kLandauMpvShift * sigma;
}

}  // namespace

void mip_landau(bool save_plot = false,
                bool add_epic_logo = false,
                const char* input_path = "data/calib/landau_plots.root")
{
  gROOT->ProcessLine("set_ePIC_style()");

  TFile input(input_path, "READ");
  if (input.IsZombie())
    {
      Error("mip_landau", "Could not open %s", input_path);
      return;
    }

  auto* histogram = dynamic_cast<TH1D*>(input.Get("h_seg1_mip"));
  auto* fit = dynamic_cast<TF1*>(input.Get("landau_fit_seg1"));
  if (!histogram || !fit)
    {
      Error("mip_landau", "Missing h_seg1_mip or landau_fit_seg1 in %s", input_path);
      return;
    }

  histogram->SetDirectory(nullptr);

  const int bin_count = histogram->GetNbinsX();
  const double x_min_mev = histogram->GetXaxis()->GetXmin() * 1000.0;
  const double x_max_mev = histogram->GetXaxis()->GetXmax() * 1000.0;
  auto* histogram_mev = new TH1D("h_seg1_mip_mev", "", bin_count, x_min_mev, x_max_mev);
  histogram_mev->SetDirectory(nullptr);
  for (int bin = 1; bin <= bin_count; ++bin)
    {
      histogram_mev->SetBinContent(bin, histogram->GetBinContent(bin));
      histogram_mev->SetBinError(bin, histogram->GetBinError(bin));
    }

  auto* landau_fit = new TF1("landau_fit_seg1_draw", "landau", fit->GetXmin() * 1000.0, fit->GetXmax() * 1000.0);
  landau_fit->SetParameters(fit->GetParameter(0), fit->GetParameter(1) * 1000.0, fit->GetParameter(2) * 1000.0);
  input.Close();

  histogram_mev->SetTitle(";Layer energy [MeV];Layer count");
  histogram_mev->SetLineColor(kBlue + 1);
  histogram_mev->SetLineWidth(2);
  histogram_mev->SetStats(false);
  histogram_mev->GetXaxis()->SetRangeUser(0.0, 5.0);
  histogram_mev->SetMinimum(0.8);
  histogram_mev->SetMaximum(1.0e4);

  landau_fit->SetLineColor(kRed + 1);
  landau_fit->SetLineWidth(3);

  const double mpv = fitted_mpv(*landau_fit);

  TCanvas* canvas = new TCanvas("canvas_mip_landau", "", 900, 650);
  canvas->SetLogy();
  canvas->cd();

  histogram_mev->Draw("hist");
  landau_fit->Draw("same");

  TLine* mpv_line = new TLine(mpv, 0.82, mpv, 1.9e3);
  mpv_line->SetLineColor(kBlack);
  mpv_line->SetLineStyle(2);
  mpv_line->SetLineWidth(2);
  mpv_line->Draw("same");

  TLegend* legend = new TLegend(0.56, 0.68, 0.88, 0.86);
  legend->SetTextSize(0.032);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->AddEntry(histogram_mev, "Muon layer deposits", "l");
  legend->AddEntry(landau_fit, "Landau fit", "l");
  legend->AddEntry(mpv_line, "Fitted MPV", "l");
  legend->Draw();

  TLatex text;
  text.SetTextAlign(13);
  text.SetTextSize(0.042);
  text.DrawLatexNDC(0.16, 0.88, "Baseline segment 1 MIP calibration");
  text.SetTextSize(0.035);
  text.DrawLatexNDC(0.16, 0.82, Form("MPV = %.3g MeV", mpv));

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
      TFile* output = new TFile("docs/plots/mip_landau.root", "RECREATE");
      histogram_mev->Write();
      landau_fit->Write();
      canvas->Write();
      output->Close();
    }
}
