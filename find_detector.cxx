/* 
 * find_detector.cxx
 *
 * This program uses the MEGALib software to obtain the detector of interation 
 * using the position of an event and the geometry used for the simulation
 *
 */

// Standard
#include <iostream>
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
  //! Analyze what eveer needs to be analyzed...
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
  Usage<<"         -p:   position of the event"<<endl;
  // Usage<<"         -f:   file name"<<endl;
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
      cout<<"Accepting Geometry file name: "<<m_GeometryFileName<<endl;
    } else if (Option == "-p") {
      m_PosVector = MVector(stod(argv[i+1]), stod(argv[i+2]), stod(argv[i+3]));
      cout<<"Saving the position vector: "<<m_PosVector<<endl;
      i+=3;
    } else {
      cout<<"Error: Unknown option \""<<Option<<"\"!"<<endl;
      cout<<Usage.str()<<endl;
      return false;
    }
  }

  return true;
}


////////////////////////////////////////////////////////////////////////////////


//! Do whatever analysis is necessary
bool PosFinder::Analyze(int argc, char** argv)
{
  if (m_Interrupt == true) return false;
  //m_Interface;
  /*cout<<"test de la classe : "<<GetGeometry()<<endl;
  char* short_argv[3];
  for (int i = 0; i < 3; ++i) {
    short_argv[i] = argv[i];
    cout<<" TEST : "<<short_argv[i]<<endl;
  }*/
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
  // First a goody: Check for overlaps:
  vector<MDVolume*> OverlappingVolumes;
  m_Geometry->GetWorldVolume()->FindOverlaps(m_PosVector, OverlappingVolumes);
  cout<<endl;
  if (OverlappingVolumes.size() == 0) {
    cout<<"Outside worldvolume "<<m_PosVector<<" cm:"<<endl;
  } else if (OverlappingVolumes.size() == 1) {
    cout<<"Details for position "<<m_PosVector<<" cm (no overlaps found) :"<<endl;
    MDVolumeSequence Vol = m_Geometry->GetVolumeSequence(m_PosVector);
    //cout<<Vol.ToString()<<endl;
    cout<<"  TEST  :  "<<Vol.GetVolumeAt(1)->GetName()<<"/"<<Vol.GetVolumeAt(2)->GetName()<<endl;
  } else {
    cout<<"Following volumes overlap at position "<<m_PosVector<<" cm:"<<endl;
    for (unsigned int i = 0; i < OverlappingVolumes.size(); ++i) {
      cout<<OverlappingVolumes[i]->GetName()<<endl;
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
