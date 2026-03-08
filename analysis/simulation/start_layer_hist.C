/*
Histogram the start_layer values stored in a processed events.root file.

The histogram includes one extra bin for start_layer = -1 so you can see how
many events never crossed the start threshold in any active HCAL layer.

root -l -b -q 'analysis/simulation/start_layer_hist.C("data/processed/9d9b1d6b/run7fe93f121f/events.root","start_layer_hist.root")'
*/

#include <TH1D.h>
#include <TFile.h>
#include <TTree.h>

#include <iostream>
#include <string>

void start_layer_hist(const char* events_path_cstr,
                      const char* out_root_cstr = "start_layer_hist.root") {
  if (!events_path_cstr || std::string(events_path_cstr).empty()) {
    std::cerr << "[start_layer_hist] events.root path is required.\n";
    return;
  }

  const std::string events_path(events_path_cstr);
  const std::string out_root =
      (out_root_cstr && std::string(out_root_cstr).size())
          ? std::string(out_root_cstr)
          : std::string("start_layer_hist.root");

  TFile input_file(events_path.c_str(), "READ");
  if (input_file.IsZombie()) {
    std::cerr << "[start_layer_hist] Failed to open " << events_path << ".\n";
    return;
  }

  TTree* tree = nullptr;
  input_file.GetObject("events", tree);
  if (!tree) {
    std::cerr << "[start_layer_hist] Tree 'events' not found in " << events_path << ".\n";
    return;
  }
  if (tree->GetBranch("start_layer") == nullptr) {
    std::cerr << "[start_layer_hist] Branch start_layer is required in " << events_path << ".\n";
    return;
  }

  int start_layer = -1;
  tree->SetBranchAddress("start_layer", &start_layer);

  TH1D histogram(
      "h_start_layer",
      "Start-layer distribution;start_layer;Event count",
      11,
      -1.5,
      9.5);
  histogram.SetDirectory(nullptr);
  histogram.SetStats(false);

  const Long64_t entry_count = tree->GetEntries();
  for (Long64_t event_index = 0; event_index < entry_count; ++event_index) {
    tree->GetEntry(event_index);
    histogram.Fill(static_cast<double>(start_layer));
  }

  TFile output_file(out_root.c_str(), "RECREATE");
  if (output_file.IsZombie()) {
    std::cerr << "[start_layer_hist] Failed to open " << out_root
              << " for writing.\n";
    return;
  }
  histogram.Write();
  output_file.Close();

  std::cout << "[start_layer_hist] Wrote " << out_root << "\n";
  std::cout << "[start_layer_hist] Events=" << entry_count << "\n";
}
