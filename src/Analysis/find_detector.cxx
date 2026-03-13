/*
 * ================================================================
 * Author      : Nathan Franel
 * Version     : 1.0
 * Created     : 2024-01-09
 * Description  :  find_detector.cxx
 * This program uses the MEGALib software to obtain the detector of interation 
 * using the position of an event and the geometry used for the simulation
 * ================================================================
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
  bool m_UseFile;
  bool m_UsePos;
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
PosFinder::PosFinder() : m_Interrupt(false), m_UseFile(false), m_UsePos(false)
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
  Usage<<"    Warning : Either -p x y z or -f filename must be used"<<endl;
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
      if (m_UseFile) {
        cout << "Error: -p and -f cannot be used together!" << endl;
        return false;
      }
      m_UsePos = true;
      m_PosVector = MVector(stod(argv[i+1]), stod(argv[i+2]), stod(argv[i+3]));
      cout<<"Saving the position vector: "<<m_PosVector<<endl;
      i+=3;
    }
    else if (Option == "-f") {
      if (m_UsePos) {
        cout << "Error: -p and -f cannot be used together!" << endl;
        return false;
      }
      m_UseFile = true;
      m_dat_file = argv[i+1];
      m_dat_file += ".txt";
      cout << "Name of the extraction file : " << m_dat_file.c_str() << "\n" << endl;
      m_save_file = argv[i+1];
      m_save_file += "save.txt";
      cout << "Name of the saving file : " << m_save_file.c_str() << "\n" << endl;
      i+=1;
    }
    else {
      cout<<"Error: Unknown option \""<<Option<<"\"!"<<endl;
      cout<<Usage.str()<<endl;
      return false;
    }
  }
  if (!m_UseFile && !m_UsePos) {
    cout << "Error: either -p or -f must be specified!" << endl;
    cout << Usage.str() << endl;
    return false;
  }
  return true;
}


////////////////////////////////////////////////////////////////////////////////

//! Retrieve the detector where the interaction happened
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
  } else {
    cout<<"Loading of geometry "<<m_Geometry->GetName()<<" failed!!"<<endl;
    return false;
  }

  // Analysing according to the two possibilities -p or -f
  if (m_UseFile) {
    // Opening the file containing positions
    std::ifstream file_stream(m_dat_file.c_str(), std::ios::binary);
    if (!file_stream) {
      cerr<<"Impossible to open position file"<<endl;
      return false;
    }
    // Opening the file where the interaction detector will be saved
      std::ofstream save_stream(m_save_file.c_str());
    if (!save_stream) {
      cerr<<"Impossible to open saving file"<<endl;
      return false;
    }

    // Reading the file, making the analysis for each line and saving the result
    while (getline(file_stream, line)) {
      cout << "Line : " << line << endl;
      // Using a streamstring to extract the positions from the line
      std::istringstream iss(line);

      // Extraction of the position from the line
      if (iss >> xpos >> ypos >> zpos) {
          // Displaying the positions if needed for testing
          // std::cout << "xpos : " << xpos << std::endl;
          // std::cout << "ypos : " << ypos << std::endl;
          // std::cout << "zpos : " << zpos << std::endl;
      }
      else {
          std::cerr << "Error during varible extraction from the line." << std::endl;
          save_stream << "Error" << " " << "InvalidPosition" << endl;
          continue;
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
        // cout<<Vol.ToString()<<endl;
        // Next line enable the extraction of the precise location, but it does not work if the location given is not in a sensitive volume !
        // cout<<"  TEST  :  "<<Vol.GetVolumeAt(1)->GetName()<<"/"<<Vol.GetVolumeAt(2)->GetName()<<endl;
        // cout<<"Outside worldvolume "<<m_PosVector<<" cm:"<<endl;
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
    // Closing the files
    file_stream.close();
    save_stream.close();
  } else {
    // First a goody: Check for overlaps:
    vector<MDVolume*> OverlappingVolumes;
    m_Geometry->GetWorldVolume()->FindOverlaps(m_PosVector, OverlappingVolumes);
    cout<<endl;
    if (OverlappingVolumes.size() == 0) {
      cout <<"Outside worldvolume "<<m_PosVector<<" cm:"<<endl;
    }
    else if (OverlappingVolumes.size() == 1) {
      cout<<"Details for position "<<m_PosVector<<" cm (no overlaps found) :"<<endl;
      MDVolumeSequence Vol = m_Geometry->GetVolumeSequence(m_PosVector);
      // Next line gives out all the information about the location, works for all location, even out of a sensitive volume
      // cout<<Vol.ToString()<<endl;
      // Next line enable the extraction of the precise location, but it does not work if the location given is not in a sensitive volume !
      // cout<<"  TEST  :  "<<Vol.GetVolumeAt(1)->GetName()<<"/"<<Vol.GetVolumeAt(2)->GetName()<<endl;
      // cout<<"Outside worldvolume "<<m_PosVector<<" cm:"<<endl;
      cout << Vol.GetVolumeAt(1)->GetName() << " " << Vol.GetVolumeAt(2)->GetName() << endl;
    }
    else {
      cout<<"Following volumes overlap at position "<<m_PosVector<<" cm:"<<endl;
      for (unsigned int i = 0; i < OverlappingVolumes.size(); ++i) {
        cout << "Overlap : " << OverlappingVolumes[i]->GetName()<<endl;
      }
    }
  }
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
//   Catch a user interupt for graceful shutdown
  signal(SIGINT, CatchSignal);

  g_Prg = new PosFinder();
  // The following lines are used to call the methods and determine if there was an error
  if (g_Prg->ParseCommandLine(argc, argv) == false) {
    cerr<<"Error during parsing of command line!"<<endl;
    return -1;
  }
  // Execution of the analysis
  if (g_Prg->Analyze(argc, argv) == false) {
    cerr<<"Error during analysis!"<<endl;
    return -2;
  }

  cout<<"Program exited normally!"<<endl;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
