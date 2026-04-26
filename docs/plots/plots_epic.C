#include "ePIC_style.C"


//=====================
void plots_epic(bool save_plot = true,
		bool add_epic_logo = false)
{
  // ===== Set the style =====
  gROOT->ProcessLine("set_ePIC_style()");

  // ===== Histogram configuration =====
  const int nBins = 40;
  const double xMin = 10;
  const double xMax = 150;

  TH1F* hist_data = new TH1F("hist_data", "", nBins, xMin, xMax);
  TH1F* hist_mc   = new TH1F("hist_mc",   "", nBins, xMin, xMax);

  // ===== Independent RNGs for statistical difference =====
  TRandom3 rand_data(0);         // Data
  TRandom3 rand_mc(12345);       // MC

  // ===== Fill histograms =====
  for (int i = 0; i < 1e5; ++i)
    {
      hist_data->Fill(rand_data.Landau(40, 10));
      hist_mc->Fill(rand_mc.Landau(41.0, 10)); // Slight shift
    }

  // ===== Draw histograms =====
   
   TCanvas *canvas = new TCanvas("canvas", "", 800, 600);
   canvas -> SetTitle("Plot leading ejet");
   canvas -> cd();
   canvas -> SetLogy();

   hist_data -> Sumw2();
   hist_data -> SetTitle(";p_{T} (GeV/c);N_{evts}");
   hist_data -> SetLineColor(kBlack);
   hist_data -> SetMaximum(1.e5);
   hist_data -> SetMinimum(1.e2);
   hist_data -> Draw("");

   hist_mc -> SetLineColor(kRed);
   hist_mc -> Draw("hist same");
    
   TLegend *legend = new TLegend(0.65, 0.72, 0.9, 0.9);
   legend -> AddEntry(hist_data,"Data", "P");
   legend -> AddEntry(hist_mc,"Simulation", "L");
   legend -> Draw();

   TLatex Text_com;
   Text_com.SetTextAlign(13);  //align at top
   Text_com.DrawLatexNDC(.15,.85,"e+p, #sqrt{s} = 140 GeV");
   Text_com.DrawLatexNDC(.15,.8,"L_{proj} = 10 fb^{-1}");


   TLatex Text_ePIC;
   Text_ePIC.SetTextSize(0.05);
   Text_ePIC.SetTextFont(62);
   Text_ePIC.DrawLatexNDC(.15,.88,"ePIC Performance");  // performance plot
   //Text_ePIC.DrawLatexNDC(.15,.88,"ePIC Internal");  // for internal use only
   //Text_ePIC.DrawLatexNDC(.15,.88,"ePIC Preliminary"); // preliminary released version 
   //Text_ePIC.DrawLatexNDC(.15,.88,"ePIC Work in Progress"); // work in progress to be shown outside
   //Text_ePIC.DrawLatexNDC(.15,.88,"ePIC"); // final published version

   // Add dates: needed for performance plots
   TLatex Text_date;
   Text_date.SetTextSize(0.035);
   Text_date.SetTextFont(52);
   Text_date.DrawLatexNDC(.65,.96,"Simu campaign: MM/YYYY");  // performance plot

   if(add_epic_logo)
     {
       // ===== Add ePIC logo to the figure if desired ======
       TImage *logo = TImage::Open("EPIC-logo_black_small.png");
       TPad *pad2 = new TPad("pad2", "Pad 2", 0.8, 0.8, 0.93, 0.93); // Create a new pad and then draw the image in it
       pad2->Draw();
       pad2->cd(); // Enter the new pad
       logo->Draw();
     }

   if(save_plot)
     {
       canvas -> SaveAs("example_plot.pdf");
     }
}
