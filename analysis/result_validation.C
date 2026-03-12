'''
Possibly try this: 
root -l 'plot_eff_vs_threshold.C("threshold_vs_efficiency.csv","eff_vs_threshold.png")'
'''

#include <TCanvas.h>
#include <TGraph.h>
#include <TAxis.h>
#include <TStyle.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void plot_eff_vs_threshold(const char* csv_file = "threshold_vs_efficiency.csv",
                           const char* output_file = "eff_vs_threshold.png") {
    std::ifstream fin(csv_file);
    if (!fin.is_open()) {
        std::cerr << "Error: could not open " << csv_file << std::endl;
        return;
    }

    std::vector<double> x_vals;
    std::vector<double> y_vals;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string x_str, y_str;

        if (!std::getline(ss, x_str, ',')) continue;
        if (!std::getline(ss, y_str, ',')) continue;

        // Skip header row if present
        if (x_str == "muon_threshold" || y_str == "efficiency") continue;

        try {
            double x = std::stod(x_str);
            double y = std::stod(y_str);
            x_vals.push_back(x);
            y_vals.push_back(y);
        } catch (...) {
            std::cerr << "Skipping malformed line: " << line << std::endl;
        }
    }

    fin.close();

    if (x_vals.empty()) {
        std::cerr << "No valid data found in " << csv_file << std::endl;
        return;
    }

    TGraph* g = new TGraph(x_vals.size(), x_vals.data(), y_vals.data());
    gStyle->SetOptStat(0);

    g->SetTitle("Efficiency vs Muon Threshold;Muon Threshold (GeV);Detection Efficiency");
    g->SetMarkerStyle(20);
    g->SetMarkerSize(1.2);
    g->SetLineWidth(2);

    TCanvas* c = new TCanvas("c", "Efficiency vs Muon Threshold", 900, 700);
    g->Draw("ALP");

    c->SaveAs(output_file);
    std::cout << "Saved plot to " << output_file << std::endl;
}