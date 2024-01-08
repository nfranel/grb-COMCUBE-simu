/* 
 * find_detector.cxx
 *
 * This program uses the MEGALib software to obtain the detector of interation 
 * using the position of an event and the geometry used for the simulation
 *
 */

// Standard
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <csignal>
#include <cstdlib>
using namespace std;

// ROOT
#include <TROOT.h>
#include <TEnv.h>
#include <TSystem.h>
#include <TApplication.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>

// MEGAlib
#include "MGlobal.h"
#include "MInterfaceGeomega.h"

////////////////////////////////////////////////////////////////////////////////


//! A standalone program based on MEGAlib and ROOT
class PosFinder
{
public:
  //! Default constructor and destructor
  PosFinder();
  ~PosFinder();
  
  //! Parse the command line
  bool ParseCommandLine(int argc, char** argv);
  //! Analyzes what ever needs to be analyzed...
  bool Analyze(int argc, char** argv);
  //! Interrupt the analysis
  void Interrupt() { m_Interrupt = true; }
  //! Setter and Getter
  void SetGeometry(MString geometry) {m_GeometryFileName = geometry;}
  void SetPosition(MVector position) {m_PosVector = position;}
  MString GetGeometry() {return m_GeometryFileName;}
  MVector GetPosition() {return m_PosVector;}

private:
  //! True, if the analysis needs to be interrupted
  bool m_Interrupt;
  //! Other attributes that will be needed (geometry, x, y, z)
  MString m_GeometryFileName;
  string m_dat_file;
  string m_save_file;
  MVector m_PosVector;
  MInterfaceGeomega m_Interface;
  MDGeometryQuest* m_Geometry;
};


////////////////////////////////////////////////////////////////////////////////


//! Default constructor : Initialize the interuption as false
PosFinder::PosFinder() : m_Interrupt(false)
{
  gStyle->SetPalette(1, 0);
}


////////////////////////////////////////////////////////////////////////////////


//! Default destructor
PosFinder::~PosFinder()
{
  // Intentionally left blank
}


////////////////////////////////////////////////////////////////////////////////


//! Parse the command line
bool PosFinder::ParseCommandLine(int argc, char** argv)
{
  ostringstream Usage;
  Usage<<endl;
  Usage<<"  Usage: PosFinder <options>"<<endl;
  Usage<<"    General options:"<<endl;
  Usage<<"         -g:   geometry file name"<<endl;
  Usage<<"         -p:   position of 1 event"<<endl;
  Usage<<"         -f:   name of the file containing the events position"<<endl;
  Usage<<"         -h:   print this help"<<endl;
  Usage<<endl;

  string Option;

  // Check for help
  for (int i = 1; i < argc; i++) {
    Option = argv[i];
    if (Option == "-h" || Option == "--help" || Option == "?" || Option == "-?") {
      cout<<Usage.str()<<endl;
      return false;
    }
  }

  // Now parse the command line options:
  for (int i = 1; i < argc; i++) {
    Option = argv[i];
    // First check if each option has sufficient arguments:
    // Single argument
    if (Option == "-g") {
      if (!((argc > i+1) && 
            (argv[i+1][0] != '-' || isalpha(argv[i+1][1]) == 0))){
        cout<<"Error: Option "<<argv[i][1]<<" needs a second argument!"<<endl;
        cout<<Usage.str()<<endl;
        return false;
      }
    }
    else if (Option == "-f") {
      if (!((argc > i+1) &&
            (argv[i+1][0] != '-' || isalpha(argv[i+1][1]) == 0))){
        cout<<"Error: Option "<<argv[i][1]<<" needs a second argument!"<<endl;
        cout<<Usage.str()<<endl;
        return false;
      }
    }
    // Multiple arguments
    else if (Option == "-p") {
      if (!((argc > i+3) && 
            (argv[i+1][0] != '-' || isalpha(argv[i+1][1]) == 0) && 
            (argv[i+2][0] != '-' || isalpha(argv[i+2][1]) == 0) &&
            (argv[i+3][0] != '-' || isalpha(argv[i+3][1]) == 0))){
        cout<<"Error: Option "<<argv[i][1]<<" needs three arguments!"<<endl;
        cout<<Usage.str()<<endl;
        return false;
      }
    }

    // Then fulfill the options:
    if (Option == "-g") {
      m_GeometryFileName = argv[++i];
      cout << "Accepting Geometry file name: " << m_GeometryFileName << endl;
    }
    else if (Option == "-p") {
      m_PosVector = MVector(stod(argv[i+1]), stod(argv[i+2]), stod(argv[i+3]));
      cout<<"Saving the position vector: "<<m_PosVector<<endl;
      i+=3;
    }
    else if (Option == "-f") {
      m_dat_file = argv[i+1];
      m_dat_file += ".txt";
      cout << "Nom du fichier d'extraction : " << m_dat_file.c_str() << "\n" << endl;
      m_save_file = argv[i+1];
      m_save_file += "save.txt";
      cout << "Nom du fichier de sauvegarde : " << m_save_file.c_str() << "\n" << endl;
      i+=3;
    }
    else {
      cout<<"Error: Unknown option \""<<Option<<"\"!"<<endl;
      cout<<Usage.str()<<endl;
      return false;
    }
  }

  return true;
}


////////////////////////////////////////////////////////////////////////////////

// OLD VERSION
// //! Do whatever analysis is necessary
// bool PosFinder::Analyze(int argc, char** argv)
// {
//   if (m_Interrupt == true) return false;
//   // Load geometry:
//   m_Geometry = new MDGeometryQuest();
//   if (m_Geometry->ScanSetupFile(m_GeometryFileName) == true) {
//     cout<<"Geometry "<<m_Geometry->GetName()<<" loaded!"<<endl;
//     //m_Geometry->ActivateNoising(false);
//     //m_Geometry->SetGlobalFailureRate(0.0);
//   } else {
//     cout<<"Loading of geometry "<<m_Geometry->GetName()<<" failed!!"<<endl;
//     return false;
//   }
//   // First a goody: Check for overlaps:
//   vector<MDVolume*> OverlappingVolumes;
//   m_Geometry->GetWorldVolume()->FindOverlaps(m_PosVector, OverlappingVolumes);
//   cout<<endl;
//   if (OverlappingVolumes.size() == 0) {
//     cout<<"Outside worldvolume "<<m_PosVector<<" cm:"<<endl;
//   }
//   else if (OverlappingVolumes.size() == 1) {
//     cout<<"Details for position "<<m_PosVector<<" cm (no overlaps found) :"<<endl;
//     MDVolumeSequence Vol = m_Geometry->GetVolumeSequence(m_PosVector);
//     // Next line gives out all the information about the location, works for all location, even out of a sensitive volume
//     cout<<Vol.ToString()<<endl;
//     // Next line enable the extraction of the precise location, but it does not work if the location given is not in a sensitive volume !
//     cout<<"  TEST  :  "<<Vol.GetVolumeAt(1)->GetName()<<"/"<<Vol.GetVolumeAt(2)->GetName()<<endl;
//   }
//   else {
//     cout<<"Following volumes overlap at position "<<m_PosVector<<" cm:"<<endl;
//     for (unsigned int i = 0; i < OverlappingVolumes.size(); ++i) {
//       cout<<OverlappingVolumes[i]->GetName()<<endl;
//     }
//   }
//   return true;
// }

//! Do whatever analysis is necessary
bool PosFinder::Analyze(int argc, char** argv)
{
  // Variable declaration
  string line;
  double xpos, ypos, zpos;

  // Looking for any interuption
  if (m_Interrupt == true) return false;

  // Load geometry:
  m_Geometry = new MDGeometryQuest();
  if (m_Geometry->ScanSetupFile(m_GeometryFileName) == true) {
    cout<<"Geometry "<<m_Geometry->GetName()<<" loaded!"<<endl;
    //m_Geometry->ActivateNoising(false);
    //m_Geometry->SetGlobalFailureRate(0.0);
  } else {
    cout<<"Loading of geometry "<<m_Geometry->GetName()<<" failed!!"<<endl;
    return false;
  }

  // Opening the file containing positions
  std::ifstream file_stream(m_dat_file.c_str(), std::ios::binary);
  if (!file_stream) {
    cerr<<"Impossible d'ouvrir le fichier de positions"<<endl;
  }
  // Opening the file where the interaction detector will be saved
    std::ofstream save_stream(m_save_file.c_str());
  if (!save_stream) {
    cerr<<"Impossible d'ouvrir le fichier de sauvegarde"<<endl;
  }

  // Reading the file, making the analysis for each line and saving the result
  while (getline(file_stream, line)) {
    cout << "Line : " << line << endl;
    // Utiliser un stringstream pour extraire les variables
    std::istringstream iss(line);

    // Extraire les variables de la ligne
    if (iss >> xpos >> ypos >> zpos) {
        // Faire quelque chose avec les variables, par exemple les afficher
        std::cout << "xpos : " << xpos << std::endl;
        std::cout << "ypos : " << ypos << std::endl;
        std::cout << "zpos : " << zpos << std::endl;
    } else {
        std::cerr << "Erreur lors de l'extraction des variables depuis la ligne." << std::endl;
    }
    m_PosVector = MVector(xpos, ypos, zpos);

    // First a goody: Check for overlaps:
    vector<MDVolume*> OverlappingVolumes;
    m_Geometry->GetWorldVolume()->FindOverlaps(m_PosVector, OverlappingVolumes);
    cout<<endl;
    if (OverlappingVolumes.size() == 0) {
      save_stream << "Outside" << " " << "Outside" << endl;
      cout<<"Outside worldvolume "<<m_PosVector<<" cm:"<<endl;
    }
    else if (OverlappingVolumes.size() == 1) {
      cout<<"Details for position "<<m_PosVector<<" cm (no overlaps found) :"<<endl;
      MDVolumeSequence Vol = m_Geometry->GetVolumeSequence(m_PosVector);
      // Next line gives out all the information about the location, works for all location, even out of a sensitive volume
      cout<<Vol.ToString()<<endl;
      // Next line enable the extraction of the precise location, but it does not work if the location given is not in a sensitive volume !
      cout<<"  TEST  :  "<<Vol.GetVolumeAt(1)->GetName()<<"/"<<Vol.GetVolumeAt(2)->GetName()<<endl;
      cout<<"Outside worldvolume "<<m_PosVector<<" cm:"<<endl;
      save_stream << Vol.GetVolumeAt(1)->GetName() << " " << Vol.GetVolumeAt(2)->GetName() << endl;
    }
    else {
      cout<<"Following volumes overlap at position "<<m_PosVector<<" cm:"<<endl;
      for (unsigned int i = 0; i < OverlappingVolumes.size(); ++i) {
        cout<<OverlappingVolumes[i]->GetName()<<endl;
      save_stream << "Overlap" << " " << "Overlap" << endl;
      }
    }
  }
//       cout<<"Saving the position vector: "<<m_PosVector<<endl;

  // Fermeture des fichiers
  file_stream.close();
  save_stream.close();
  return true;
}


////////////////////////////////////////////////////////////////////////////////


PosFinder* g_Prg = 0;
int g_NInterruptCatches = 1;


////////////////////////////////////////////////////////////////////////////////


//! Called when an interrupt signal is flagged
//! All catched signals lead to a well defined exit of the program
void CatchSignal(int a)
{
  if (g_Prg != 0 && g_NInterruptCatches-- > 0) {
    cout<<"Catched signal Ctrl-C (ID="<<a<<"):"<<endl;
    g_Prg->Interrupt();
  } else {
    abort();
  }
}


////////////////////////////////////////////////////////////////////////////////


//! Main program
int main(int argc, char** argv)
{
  // Catch a user interupt for graceful shutdown
  // signal(SIGINT, CatchSignal);

  // Initialize global MEGALIB variables, especially mgui, etc.
  //MGlobal::Initialize("PosFinder", "a program to find the detector in which an event occured");

  //TApplication PosFinderApp("PosFinderApp", 0, 0);

  g_Prg = new PosFinder();
  // The following lines are used to call the methods and determine if there was an error
  if (g_Prg->ParseCommandLine(argc, argv) == false) {
    cerr<<"Error during parsing of command line!"<<endl;
    return -1;
  } 
  if (g_Prg->Analyze(argc, argv) == false) {
    cerr<<"Error during analysis!"<<endl;
    return -2;
  }

  //PosFinderApp.Run();

  cout<<"Program exited normally!"<<endl;

  return 0;
}


////////////////////////////////////////////////////////////////////////////////
