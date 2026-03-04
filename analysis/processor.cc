// mini_detector/analysis/processor.cc (streamlined)
// Purpose: Minimal, ML-focused analysis for longitudinal HCAL with
//          energy-resolution-first features. Includes auto z-range discovery
//          and a compact branch set tailored for downstream modeling.
//
// How it works (high level):
// 1. Open an EDM4hep file and work out which MC truth and calorimeter hit
//    collections actually contain data.
// 2. For each event, pick the leading primary MC particle to define the truth
//    energy and direction that resolution studies compare against.
// 3. Sum HCAL SimCalorimeterHits into configurable longitudinal layers while
//    deriving a few coarse shower-shape observables.
// 4. Write one event per row into a ROOT TTree ready for resolution fits or ML regressors.
//
// Feature overview:
// - Two-zone calibration branches emit rec_E_2z/R_2z only when --twozone is set.
// - Shower-shape observables stay disabled unless --shape is provided.
// - Z-range probes run when --auto_z is active or no explicit zmin/zmax is given.
// - Layer binning clamps indices explicitly and reports edge usage in debug mode.
//
// CLI:
//   process <input.edm4hep.root> [--out out.root]
//            [--mc MCParticles] [--simhits HCal_Readout]
//            [--nlayers N] [--zmin_cm Zmin] [--zmax_cm Zmax]
//            [--sampling f]
//            [--twozone] [--Lsplit L] [--sf_front f1] [--sf_rear f2]
//            [--shape] [--Nfirst N] [--Mlast M]
//            [--auto_z] [--debug]

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <fstream>
#include <optional>
#include <type_traits>
#include <cctype>

// ROOT I/O
#include "TFile.h"
#include "TTree.h"

// ---------- Podio compatibility ----------
// Podio API selection keeps the processor portable across sites.
#if __has_include(<podio/Frame.h>)
  #define PODIO_API_FRAME 1
  #include <podio/Frame.h>
  #if __has_include(<podio/podioVersion.h>)
    #include <podio/podioVersion.h>
  #else
    #define podio_VERSION_MAJOR 1
    #define podio_VERSION_MINOR 0
  #endif
  #if (podio_VERSION_MAJOR == 0 && podio_VERSION_MINOR < 99) && __has_include(<podio/ROOTFrameReader.h>)
    #include <podio/ROOTFrameReader.h>
    namespace podio_compat { using ROOTReader = podio::ROOTFrameReader; }
  #elif __has_include(<podio/ROOTReader.h>)
    #include <podio/ROOTReader.h>
    namespace podio_compat { using ROOTReader = podio::ROOTReader; }
  #else
    #define PODIO_API_MISSING 1
  #endif
#else
  #define PODIO_API_MISSING 1
#endif

#ifdef PODIO_API_MISSING
int main(int, char**) {
  std::cerr << "[processor] No usable Podio headers found.\n";
  return 2;
}
#endif

// ---------- EDM4hep ----------
#include "edm4hep/MCParticleCollection.h"
#include "edm4hep/SimCalorimeterHitCollection.h"

// ============================================================================
// CLI Helpers
// ============================================================================
static bool hasArg(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) if (key == argv[i]) return true; return false;
}
static std::string getArg(int argc, char** argv, const std::string& key, const std::string& def="") {
  for (int i = 1; i < argc - 1; ++i) if (key == argv[i]) return std::string(argv[i+1]); return def;
}
static double getArgD(int argc, char** argv, const std::string& key, double def) {
  const auto s = getArg(argc, argv, key, ""); return s.empty() ? def : std::stod(s);
}
static int getArgI(int argc, char** argv, const std::string& key, int def) {
  const auto s = getArg(argc, argv, key, ""); return s.empty() ? def : std::stoi(s);
}
static std::vector<int> getArgIntList(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) {
      std::vector<int> values;
      int j = i + 1;
      while (j < argc) {
        const std::string token = argv[j];
        if (token.empty()) { ++j; continue; }
        if (token[0] == '-' && (token.size() == 1 || !std::isdigit(static_cast<unsigned char>(token[1])))) {
          break;
        }
        try {
          values.push_back(std::stoi(token));
        } catch (...) {
          break;
        }
        ++j;
      }
      return values;
    }
  }
  return {};
}

static std::string json_escape(const std::string& in) {
  std::string out; out.reserve(in.size() + (in.size() / 2));
  for (char c : in) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

// ============================================================================
// Math / Kinematics Helpers
// ============================================================================
inline double pt(double px, double py) { return std::sqrt(px*px + py*py); }

// eta safeguard: saturate for pT~0 (pencil beams)
inline double eta_from_p(double px, double py, double pz) {
  const double pT = pt(px, py);
  const double p  = std::sqrt(px*px + py*py + pz*pz);
  if (p <= 0.0) return 0.0;
  if (pT < 1e-9) return (pz >= 0 ? 10.0 : -10.0);
  const double theta = std::atan2(pT, pz);
  return -std::log(std::tan(0.5 * theta));
}
inline double phi_from_p(double px, double py) { return std::atan2(py, px); }

inline double energy_from_mc(const edm4hep::MCParticle& p) {
  double E = p.getEnergy();
  if (E > 0) return E;
  const auto& q = p.getMomentum();
  const double p2 = q.x*q.x + q.y*q.y + q.z*q.z;
  const double m  = p.getMass();
  return std::sqrt(std::max(0.0, p2 + m*m));
}

// ============================================================================
// Layer Mapping by z
// ============================================================================
struct LayerBinner {
  double zmin_mm = 3000.0;
  double zmax_mm = 5000.0;
  int    nLayers = 40;
  int index_from_z_mm(double z_mm) const {
    if (nLayers <= 0 || zmax_mm <= zmin_mm) return -1;
    const double f = (z_mm - zmin_mm) / (zmax_mm - zmin_mm);
    int idx = static_cast<int>(std::floor(f * nLayers));
    if (idx < 0) idx = 0;
    if (idx >= nLayers) idx = nLayers - 1;
    return idx;
  }
};

// ============================================================================
// Collection Helpers
// ============================================================================

#if PODIO_API_FRAME
template<typename T>
static bool tryGetParam(const podio::Frame& frame, const std::string& key, T& out) {
  auto opt = frame.getParameter<T>(key);
  if (opt) { out = *opt; return true; }
  return false;
}
#endif

#if PODIO_API_FRAME
template<typename CollT>
const CollT* tryGetFrameByName(const podio::Frame& frame, const std::vector<std::string>& names, std::string& chosen) {
  for (const auto& n : names) {
    try {
      const auto& ref = frame.get<CollT>(n);
      if (!ref.empty()) { chosen = n; return &ref; }
    } catch (...) {}
  }
  return nullptr;
}

template<typename CollT>
const CollT* findByType(const podio::Frame& frame, std::string& chosen) {
  const auto names = frame.getAvailableCollections();
  for (const auto& n : names) {
    try {
      const auto& ref = frame.get<CollT>(n);
      if (!ref.empty()) { chosen = n; return &ref; }
    } catch (...) {}
  }
  return nullptr;
}
#endif

int main(int argc, char** argv) {
  if (argc < 2 || hasArg(argc, argv, "--help")) {
  std::cout <<
      "Usage: process <input.edm4hep.root> [--out out.root]\n"
      "                [--mc MCParticles] [--simhits HCal_Readout]\n"
      "                [--nlayers N] [--zmin_cm Zmin] [--zmax_cm Zmax]\n"
      "                [--sampling f]\n"
      "                [--twozone] [--Lsplit L] [--sf_front f1] [--sf_rear f2]\n"
      "                [--shape] [--Nfirst N] [--Mlast M]\n"
      "                [--auto_z] [--debug]\n"
      "                [--geom-id ID] [--run-id N]\n"
      "                [--collections-out path]\n";
    return 0;
  }

  const bool debug     = hasArg(argc, argv, "--debug");
  const std::string inFile   = argv[1];
  const std::string outFile  = getArg(argc, argv, "--out", "events.root");
  const std::string mcName   = getArg(argc, argv, "--mc", "MCParticles");
  const std::string simHint  = getArg(argc, argv, "--simhits", "");
  const int expectedPDG      = getArgI(argc, argv, "--expected-pdg", 0);
  const int    nLayers       = getArgI(argc, argv, "--nlayers", 40);
  const double zmin_cm_in    = getArgD(argc, argv, "--zmin_cm", 300.0);
  const double zmax_cm_in    = getArgD(argc, argv, "--zmax_cm", 500.0);
  const bool   userSetZmin   = hasArg(argc, argv, "--zmin_cm");
  const bool   userSetZmax   = hasArg(argc, argv, "--zmax_cm");
  const bool   autoZ_flag    = hasArg(argc, argv, "--auto_z") || !(userSetZmin || userSetZmax);
  const double sampling_f    = getArgD(argc, argv, "--sampling", 0.03);
  const std::string geom_id  = getArg(argc, argv, "--geom-id", "");
  const int run_id           = getArgI(argc, argv, "--run-id", 0);
  const std::string collections_out = getArg(argc, argv, "--collections-out", "");

  const bool   twozone       = hasArg(argc, argv, "--twozone");
  const int    Lsplit        = getArgI(argc, argv, "--Lsplit", std::max(1, nLayers/2));
  const double sf_front      = getArgD(argc, argv, "--sf_front", sampling_f);
  const double sf_rear       = getArgD(argc, argv, "--sf_rear",  sampling_f);

  const bool   shape         = hasArg(argc, argv, "--shape");
  const int    N_first       = getArgI(argc, argv, "--Nfirst", 6);
  const int    M_last        = getArgI(argc, argv, "--Mlast", 3);
  const std::vector<int> seg_layers_cli = getArgIntList(argc, argv, "--seg-layers");

  // World energy: prefer the world parameter; otherwise set it from the summed calorimeter energy.

  // CLI toggles above align the reader with the chosen filenames, geometry granularity,
  // and calibration strategy without recompiling.

  // Keep order: user-specified name first, then common names
  std::vector<std::string> mcCandidates = {
    mcName,
    "GeneratedParticles",
    "MCParticles",
    "GenParticles",
    "MCParticlesSkimmed",
    "MCParticlesInit",
    "SimPrimaryParticles",
    "PrimaryParticles"
  };
  {
    std::vector<std::string> unique; unique.reserve(mcCandidates.size());
    for (auto& n : mcCandidates) if (std::find(unique.begin(), unique.end(), n) == unique.end()) unique.push_back(n);
    mcCandidates.swap(unique);
  }

  std::vector<std::string> simCandidates = simHint.empty()
    ? std::vector<std::string>{ "HCal_Readout", "HCalHits", "HCALHits", "LFHCALHits",
                                "HadCalorimeterHits", "HADCalHits" }
    : std::vector<std::string>{ simHint };

  // Candidate calorimeter collection names used by this dataset family.

  // ---- Prepare layerer (auto z-range if requested) ----
  LayerBinner layerer; layerer.nLayers = nLayers;
  layerer.zmin_mm = zmin_cm_in * 10.0;
  layerer.zmax_mm = zmax_cm_in * 10.0;

#if PODIO_API_FRAME
  if (autoZ_flag) {
    if (debug) std::cerr << "[process] Auto z-range probe...\n";
    podio_compat::ROOTReader probe;
    try { probe.openFile(inFile); } catch (std::exception& e) { std::cerr << "ERROR opening " << inFile << ": " << e.what() << std::endl; return 2; }

    std::string category = "events";
    try {
      auto cats = probe.getAvailableCategories();
      if (!cats.empty()) {
        if (std::find(cats.begin(), cats.end(), "events") != cats.end()) category = "events"; else category = cats.front();
      }
    } catch (...) {}

    const size_t nEntriesP = probe.getEntries(category);
    double zmin_mm = std::numeric_limits<double>::infinity();
    double zmax_mm = -std::numeric_limits<double>::infinity();
    size_t nHitsSeen = 0;

    for (size_t iev = 0; iev < nEntriesP; ++iev) {
      auto fdat = probe.readEntry(category, iev);
      podio::Frame frame(std::move(fdat));
      std::string simChosen;
      const auto* simCol = tryGetFrameByName<edm4hep::SimCalorimeterHitCollection>(frame, simCandidates, simChosen);
      if (!simCol) simCol = findByType<edm4hep::SimCalorimeterHitCollection>(frame, simChosen);
      if (!simCol || simCol->empty()) continue;
      for (const auto& hit : *simCol) {
        const double z = hit.getPosition().z;
        zmin_mm = std::min(zmin_mm, z);
        zmax_mm = std::max(zmax_mm, z);
        ++nHitsSeen;
      }
    }
    if (nHitsSeen > 0 && zmax_mm > zmin_mm) {
      // Add a tiny margin so later hits do not clip right at the boundary.
      const double dz = (zmax_mm - zmin_mm) * 0.01 + 1.0; // 1% or 1 mm
      layerer.zmin_mm = zmin_mm - dz;
      layerer.zmax_mm = zmax_mm + dz;
      if (debug) std::cerr << "[process] Auto z-range: [" 
                           << layerer.zmin_mm << ", " << layerer.zmax_mm << "] mm from " << nHitsSeen << " hits\n";
    } else if (debug) {
      std::cerr << "[process] Auto z-range failed (no hits); using CLI defaults.\n";
    }
  }
#endif


  // ---- Output TFile & TTree ----
  TFile* fout = TFile::Open(outFile.c_str(), "RECREATE");
  if (!fout || fout->IsZombie()) {
    std::cerr << "ERROR: Cannot open output file: " << outFile << std::endl;
    return 2;
  }

  float  t_mc_E=0, t_mc_eta=0, t_mc_phi=0;
  float  t_sim_E=0; int t_nsim=0;
  
  // World holds the containment denominator.
  float  t_simE_world_GeV = -1.0f;

  // ML targets/predictions and features (minimal by default)
  float  t_rec_E=0, t_R=0;
  // Optional 2-zone
  float  t_rec_E_2z=0, t_R_2z=0;

  // Optional shower-shape
  int    t_shower_max=-1, t_L20=-1, t_L50=-1, t_L80=-1;
  float  t_depth_mean=0, t_contain_firstN=0, t_leak_lastM=0, t_t_first_min=0;
  std::vector<float> t_layerE, t_layerTmin; // branched only if --shape

  // Segment-level energy summaries (always available when segmentation supplied)
  std::vector<int>   t_seg_layers;
  std::vector<float> t_segE_sim;
  std::vector<float> t_segE_rec;

  // Provenance/meta
  std::string t_geom_id = geom_id;
  int         t_run_id  = run_id;
  int         t_event_id = -1;
  std::string t_input_file = inFile;
  std::string t_category;
  std::string t_mc_collection;
  std::string t_sim_collection;
  int         t_nlayers_used = nLayers;
  float       t_zmin_mm_used = (float)(zmin_cm_in * 10.0);
  float       t_zmax_mm_used = (float)(zmax_cm_in * 10.0);
  int         t_mc_pdg = 0;

  TTree* events = new TTree("events", "per-event ML dataset (minimal by default)");
  events->Branch("mc_E",   &t_mc_E);
  events->Branch("mc_eta", &t_mc_eta);
  events->Branch("mc_phi", &t_mc_phi);
  events->Branch("mc_pdg", &t_mc_pdg);
  events->Branch("sim_E",  &t_sim_E);
  events->Branch("nsim",   &t_nsim);
  events->Branch("simE_world_GeV",        &t_simE_world_GeV);

  events->Branch("rec_E", &t_rec_E);
  events->Branch("R",    &t_R);

  if (twozone) {
    events->Branch("rec_E_2z", &t_rec_E_2z);
    events->Branch("R_2z",    &t_R_2z);
  }
  if (shape) {
    events->Branch("shower_max",     &t_shower_max);
    events->Branch("L20",            &t_L20);
    events->Branch("L50",            &t_L50);
    events->Branch("L80",            &t_L80);
    events->Branch("depth_mean",     &t_depth_mean);
    events->Branch("contain_firstN", &t_contain_firstN);
    events->Branch("leak_lastM",     &t_leak_lastM);
    events->Branch("t_first_min",    &t_t_first_min);
    events->Branch("layerE",         &t_layerE);
    events->Branch("layerTmin",      &t_layerTmin);
  }
  events->Branch("seg_layers",      &t_seg_layers);
  events->Branch("segE_sim_GeV",    &t_segE_sim);
  events->Branch("segE_rec_GeV",    &t_segE_rec);

  // Always branch provenance/meta
  events->Branch("geom_id",        &t_geom_id);
  events->Branch("run_id",         &t_run_id);
  events->Branch("event_id",       &t_event_id);
  events->Branch("input_file",     &t_input_file);
  events->Branch("category",       &t_category);
  events->Branch("mc_collection",  &t_mc_collection);
  events->Branch("sim_collection", &t_sim_collection);
  events->Branch("nlayers_used",   &t_nlayers_used);
  events->Branch("zmin_mm_used",   &t_zmin_mm_used);
  events->Branch("zmax_mm_used",   &t_zmax_mm_used);

  // After potential auto-z probing, update the recorded z-range to the actual layerer range
  t_zmin_mm_used = (float)layerer.zmin_mm;
  t_zmax_mm_used = (float)layerer.zmax_mm;

  size_t nEvents=0, nWithMC=0, nWithSim=0;
  size_t nEdge0=0, nEdgeLast=0, nLayerBinned=0; // debug stats

  // ---- Segment bookkeeping ----
  std::vector<int> seg_layers = seg_layers_cli;
  seg_layers.erase(std::remove(seg_layers.begin(), seg_layers.end(), 0), seg_layers.end());
  const bool haveSegments = !seg_layers.empty();
  t_seg_layers = seg_layers;
  const size_t nSegments = seg_layers.size();
  std::vector<int> layer_to_segment;
  if (haveSegments) {
    layer_to_segment.assign(nLayers, -1);
    int cursor = 0;
    for (size_t s = 0; s < seg_layers.size(); ++s) {
      const int count = std::max(0, seg_layers[s]);
      for (int L = 0; L < count && cursor < nLayers; ++L, ++cursor) {
        layer_to_segment[cursor] = static_cast<int>(s);
      }
    }
    if (cursor < nLayers && !seg_layers.empty()) {
      const int last_seg = static_cast<int>(seg_layers.size()) - 1;
      for (int L = cursor; L < nLayers; ++L) {
        layer_to_segment[L] = last_seg;
      }
    }
    if (cursor > nLayers && debug) {
      std::cerr << "[process] Warning: seg layer counts exceed nLayers; truncating remainder.\n";
    }
  }

  // ---- Input & loop ----
  podio_compat::ROOTReader reader;
  try { reader.openFile(inFile); }
  catch (std::exception& e) { std::cerr << "ERROR opening " << inFile << ": " << e.what() << std::endl; return 2; }

  std::string category = "events";
  if constexpr (std::is_same<podio_compat::ROOTReader, podio::ROOTReader>::value) {
    try {
      auto cats = reader.getAvailableCategories();
      if (!cats.empty()) {
        if (std::find(cats.begin(), cats.end(), "events") != cats.end()) category = "events"; else category = cats.front();
      }
      if (debug) {
        std::cerr << "[process] Categories: "; for (auto& c : cats) std::cerr << c << " "; std::cerr << " | chosen=" << category << "\n";
      }
    } catch (...) {}
  }

  const size_t nEntries = reader.getEntries(category);
  t_category = category;
  for (size_t iev = 0; iev < nEntries; ++iev) {
    auto frame_data = reader.readEntry(category, iev);
    podio::Frame frame(std::move(frame_data));
    t_event_id = static_cast<int>(iev);

    t_simE_world_GeV        = -1.0f;

    // podio::Frame returns std::optional<T>; unwrap safely
    if (auto w = frame.getParameter<double>("simE_world_GeV")) {
      t_simE_world_GeV = static_cast<float>(*w);
    }
    // World parameter controls containment bookkeeping.

    // Print a one-time inventory of frame collections in debug mode so naming issues surface quickly.
    if (debug && iev==0) {
      auto names = frame.getAvailableCollections();
      std::cerr << "[process] Frame[0] collections (" << names.size() << "): ";
      for (auto& n : names) std::cerr << n << " ";
      std::cerr << "\n";
    }
    ++nEvents;
    t_mc_pdg = 0;

    // ---- MC truth  ----
    bool haveMC=false; double mcE=0, mcEta=0, mcPhi=0;

    const edm4hep::MCParticleCollection* mcCol=nullptr;
    std::string mcChosen;
    mcCol = tryGetFrameByName<edm4hep::MCParticleCollection>(frame, mcCandidates, mcChosen);
    if (!mcCol) mcCol = findByType<edm4hep::MCParticleCollection>(frame, mcChosen);

    if (mcCol) {
      if (debug && nEvents==1) std::cerr << "[process] Using MC collection: " << mcChosen << " (size=" << mcCol->size() << ")\n";
      t_mc_collection = mcChosen;
      // Rank candidates: primary status > expected PDG matches > highest-energy particle.
      // We try a few friendly heuristics so the neutron is found even if the MC tree
      // does not mark it as a clean parentless primary.
      const int expectedAbsPDG = (expectedPDG != 0 ? std::abs(expectedPDG) : 0);
      std::optional<edm4hep::MCParticle> best_expected_status;
      std::optional<edm4hep::MCParticle> best_expected_notSim;
      std::optional<edm4hep::MCParticle> best_expected_near;
      std::optional<edm4hep::MCParticle> best_expected_energy;
      std::optional<edm4hep::MCParticle> best_status_any;
      std::optional<edm4hep::MCParticle> best_notSim_any;
      std::optional<edm4hep::MCParticle> best_near_any;
      std::optional<edm4hep::MCParticle> best_energy_any;
      double best_expected_status_E = -1.0;
      double best_expected_notSim_E = -1.0;
      double best_expected_near_dist = std::numeric_limits<double>::max();
      double best_expected_near_E = -1.0;
      double best_expected_energy_E = -1.0;
      double best_status_any_E = -1.0;
      double best_notSim_any_E = -1.0;
      double best_near_any_dist = std::numeric_limits<double>::max();
      double best_near_any_E = -1.0;
      double best_energy_any_E = -1.0;
      for (const auto& p : *mcCol) {
        // Energy is the main quantity we keep comparing, so cache it once per particle.
        const double Ecalc = energy_from_mc(p);
        const bool status_primary = (p.getGeneratorStatus() > 0);
        const bool not_from_sim   = !p.isCreatedInSimulation();
        const auto& vtx = p.getVertex();
        const double vtx_dist2 = vtx.x * vtx.x + vtx.y * vtx.y + vtx.z * vtx.z;
        const int pdg = p.getPDG();

        if (status_primary && Ecalc > best_status_any_E) {
          // Generator status > 0 usually tags the gun particle or close friends.
          best_status_any = p;
          best_status_any_E = Ecalc;
        }
        if (not_from_sim && Ecalc > best_notSim_any_E) {
          // This filters out secondaries born inside the detector material.
          best_notSim_any = p;
          best_notSim_any_E = Ecalc;
        }
        if (vtx_dist2 < best_near_any_dist ||
            (std::abs(vtx_dist2 - best_near_any_dist) < 1e-9 && Ecalc > best_near_any_E)) {
          // As a tiebreaker, stay close to the gun origin.
          best_near_any = p;
          best_near_any_dist = vtx_dist2;
          best_near_any_E = Ecalc;
        }
        if (Ecalc > best_energy_any_E) {
          best_energy_any = p;
          best_energy_any_E = Ecalc;
        }

        if (expectedAbsPDG > 0 && pdg != 0 && std::abs(pdg) == expectedAbsPDG) {
          if (status_primary && Ecalc > best_expected_status_E) {
            best_expected_status = p;
            best_expected_status_E = Ecalc;
          }
          if (not_from_sim && Ecalc > best_expected_notSim_E) {
            best_expected_notSim = p;
            best_expected_notSim_E = Ecalc;
          }
          if (vtx_dist2 < best_expected_near_dist ||
              (std::abs(vtx_dist2 - best_expected_near_dist) < 1e-9 && Ecalc > best_expected_near_E)) {
            best_expected_near = p;
            best_expected_near_dist = vtx_dist2;
            best_expected_near_E = Ecalc;
          }
          if (Ecalc > best_expected_energy_E) {
            best_expected_energy = p;
            best_expected_energy_E = Ecalc;
          }
        }
      }

      // Apply the priority ladder for neutrons first, then reuse the same checks for any particle.
      std::optional<edm4hep::MCParticle> pick;
      if (best_expected_status) pick = best_expected_status;
      else if (best_expected_notSim) pick = best_expected_notSim;
      else if (best_expected_near) pick = best_expected_near;
      else if (best_expected_energy) pick = best_expected_energy;
      else if (best_status_any) pick = best_status_any;
      else if (best_notSim_any) pick = best_notSim_any;
      else if (best_near_any) pick = best_near_any;
      else if (best_energy_any) pick = best_energy_any;

      if (pick) {
        // Record the chosen particle so downstream code sees the right truth energy and direction.
        const auto& chosen = *pick;
        const auto& mom = chosen.getMomentum();
        mcE   = energy_from_mc(chosen);
        mcEta = eta_from_p(mom.x, mom.y, mom.z);
        mcPhi = phi_from_p(mom.x, mom.y);
        t_mc_pdg = chosen.getPDG();
      } else if (debug && expectedPDG != 0) {
        std::cerr << "[process] Warning: expected PDG " << expectedPDG
                  << " not found in MCParticles for event " << (nEvents - 1) << std::endl;
      }
    }
    haveMC = (mcE > 0); if (haveMC) ++nWithMC;

    // ---- SimCalorimeterHits ----
    t_nsim = 0; t_sim_E = 0.f;

    if (!shape) {
      t_layerE.clear();
      t_layerTmin.clear();
    }
    t_segE_sim.assign(nSegments, 0.f);
    t_segE_rec.assign(nSegments, 0.f);
    std::vector<float> layerE_local;
    std::vector<float> layerTmin_local;

    std::string simChosen;
    const auto* simCol = tryGetFrameByName<edm4hep::SimCalorimeterHitCollection>(frame, simCandidates, simChosen);
    if (!simCol) simCol = findByType<edm4hep::SimCalorimeterHitCollection>(frame, simChosen);
    if (simCol && debug && nEvents==1) std::cerr << "[process] Using Sim collection: " << simChosen << " (size=" << simCol->size() << ")\n";
    if (simCol) t_sim_collection = simChosen;

    if (simCol) {
      t_nsim = (int)simCol->size();
      if (t_nsim > 0) ++nWithSim; // count only when non-empty
      const bool needLayerVectors = shape || twozone;
      const bool needLayerIndex   = needLayerVectors || haveSegments;
      if (needLayerVectors) {
        layerE_local.assign(nLayers, 0.f);
        layerTmin_local.assign(nLayers, +1e9f);
      }
      for (const auto& hit : *simCol) {
        const double e    = hit.getEnergy();           // GeV
        t_sim_E += (float)e;
        if (needLayerIndex) {
          double tmin_ns = 1e9;
          if (needLayerVectors) {
            for (const auto& c : hit.getContributions()) {
              tmin_ns = std::min(tmin_ns, (double)c.getTime());
            }
          }
          const double z_mm = hit.getPosition().z;
          const int L = layerer.index_from_z_mm(z_mm);
          if (L >= 0 && L < nLayers) {
            if (needLayerVectors) {
              layerE_local[L]     += (float)e;
              layerTmin_local[L]   = std::min(layerTmin_local[L], (float)tmin_ns);
              if (L==0) { ++nEdge0; ++nLayerBinned; }
              else if (L==nLayers-1) { ++nEdgeLast; ++nLayerBinned; }
              else { ++nLayerBinned; }
            }
            if (!layer_to_segment.empty()) {
              const int seg_idx = layer_to_segment[L];
              if (seg_idx >= 0 && seg_idx < (int)nSegments) {
                t_segE_sim[static_cast<size_t>(seg_idx)] += (float)e;
              }
            }
          }
        }
      }
      if (haveSegments) {
        for (size_t s = 0; s < nSegments; ++s) {
          t_segE_rec[s] = (sampling_f > 0 ? t_segE_sim[s] / (float)sampling_f : 0.f);
        }
      }
      if (needLayerVectors) {
        t_layerE = layerE_local;
        t_layerTmin = layerTmin_local;
      }
    }

    // If the frame does not provide a world sum, the next step fills it from the summed calorimeter energy.

    // If frame lacks world, set it to the summed HCal energy.
    if (t_simE_world_GeV < 0.f) {
      t_simE_world_GeV = t_sim_E;
    }

    // ---- Reconstructed energies (sim-based only) ----
    // Treat the summed SimCalorimeterHit energy as if it were raw charge and
    // scale by an approximate sampling fraction to mimic a simple calibration.
    t_rec_E = (sampling_f > 0 ? t_sim_E / (float)sampling_f : 0.f);
    t_R    = (mcE>0 ? t_rec_E / (float)mcE : 0.f);

    if (twozone && !t_layerE.empty()) {
      // Split the calorimeter stack in two to test piecewise calibrations, e.g.
      // different sampling fractions in the front and rear absorber sections.
      double E_front = 0, E_rear = 0;
      for (int L=0; L<std::min(Lsplit,nLayers); ++L) E_front += t_layerE[L];
      for (int L=Lsplit; L<nLayers; ++L)            E_rear  += t_layerE[L];
      t_rec_E_2z = (sf_front>0 ? (float)(E_front/sf_front) : 0.f)
                + (sf_rear >0 ? (float)(E_rear /sf_rear ) : 0.f);
      t_R_2z    = (mcE>0 ? t_rec_E_2z / (float)mcE : 0.f);
    }

    // ---- Optional shower-shape features ----
    // These observables describe how energy is distributed through the depth of
    // the calorimeter and extend the simple sampling-fraction scaling above.
    if (shape && !t_layerE.empty()) {
      int    argmaxL=0; double eTot=0;
      for (int L=0; L<nLayers; ++L) { if (t_layerE[L] > t_layerE[argmaxL]) argmaxL = L; eTot += t_layerE[L]; }
      t_shower_max = argmaxL;

      double depth_num=0; for (int L=0; L<nLayers; ++L) depth_num += L * t_layerE[L];
      t_depth_mean = (eTot>0 ? (float)(depth_num / eTot) : 0.f);

      t_L20 = t_L50 = t_L80 = -1;
      if (eTot > 0) {
        double cum=0;
        for (int L=0; L<nLayers; ++L) {
          cum += t_layerE[L]; const double f = cum / eTot;
          if (t_L20<0 && f>=0.20) t_L20=L;
          if (t_L50<0 && f>=0.50) t_L50=L;
          if (t_L80<0 && f>=0.80) t_L80=L;
        }
      }

      double eFirst=0, eLast=0;
      for (int L=0; L<std::min(N_first, nLayers); ++L) eFirst += t_layerE[L];
      for (int k=0; k<std::min(M_last, nLayers); ++k) eLast  += t_layerE[nLayers-1-k];
      t_contain_firstN = (eTot>0 ? (float)(eFirst/eTot) : 0.f);
      t_leak_lastM     = (eTot>0 ? (float)(eLast /eTot) : 0.f);

      float tmin_all = 1e9; for (int L=0; L<nLayers; ++L) tmin_all = std::min(tmin_all, t_layerTmin[L]);
      t_t_first_min = (tmin_all<1e8 ? tmin_all : 0.f);
    }

    // ---- Write event ----
    t_mc_E   = (float)mcE; t_mc_eta = (float)mcEta; t_mc_phi = (float)mcPhi;
    events->Fill();
  } // loop

  // ---- Save & close ----
  fout->cd(); events->Write(); fout->Close();

  if (!collections_out.empty()) {
    std::ofstream out(collections_out);
    if (out.good()) {
      out << "{\n";
      out << "  \"mc_collection\": \"" << json_escape(t_mc_collection) << "\",\n";
      out << "  \"sim_collection\": \"" << json_escape(t_sim_collection) << "\"\n";
      out << "}\n";
    } else {
      std::cerr << "[process] Warning: unable to write collections file " << collections_out << "\n";
    }
  }

  std::cout << "[process] Done. Events: " << nEvents
            << " | with MC: "  << nWithMC
            << " | with Sim: " << nWithSim;
  if (nLayerBinned>0) {
    const double f0 = (double)nEdge0 / (double)nLayerBinned;
    const double fL = (double)nEdgeLast / (double)nLayerBinned;
    std::cout << " | layer-edge fraction (L=0/L=last): " << f0 << "/" << fL;
  }
  std::cout << "\nOutput: " << outFile << std::endl;
  return 0;
}
