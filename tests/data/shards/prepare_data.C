// This ROOT script copies a tiny portion of a data ROOT file into a set of
// small files to be used for testing. Requires a full installation of ROOT.
// Run as
// root -b -q "prepare_data.C(\"path/to/donor/file.root\")"

#include <string>

#include <TFile.h>
#include <TTree.h>


void prepare_data(std::string const &source_file_path) {
  TFile source_file{source_file_path.c_str()};
  auto source_tree = source_file.Get<TTree>("Jets");
  int entry = 0;

  for (int ifile = 0; ifile < 5; ++ifile) {
    TFile output_file{
      (std::to_string(ifile + 1) + ".root").c_str(), "recreate"};
    TTree *output_tree = source_tree->CloneTree(0);
    output_tree->SetDirectory(&output_file);

    for (int i = 0; i < 10; ++i) {
      source_tree->GetEntry(entry);
      ++entry;
      output_tree->Fill();
    }

    output_file.Write();
    output_file.Close();
  }

  source_file.Close();
}
