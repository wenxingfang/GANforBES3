#include "GaudiKernel/MsgStream.h"
#include "GaudiKernel/AlgFactory.h"
#include "GaudiKernel/ISvcLocator.h"
#include "GaudiKernel/SmartDataPtr.h"
#include "GaudiKernel/IDataProviderSvc.h"
#include "GaudiKernel/PropertyMgr.h"
#include "GaudiKernel/Bootstrap.h"
#include "EventModel/EventModel.h"
#include "EventModel/Event.h"
#include "EvtRecEvent/EvtRecEvent.h"
#include "EvtRecEvent/EvtRecTrack.h"
#include "EventModel/EventHeader.h"
#include "DstEvent/TofHitStatus.h"
#include "McTruth/McParticle.h"
#include "McTruth/DecayMode.h"
#include "McTruth/MdcMcHit.h"
#include "McTruth/TofMcHit.h"
#include "McTruth/EmcMcHit.h"
#include "McTruth/TofMcHit.h"
#include "McTruth/EmcMcHit.h"
#include "McTruth/MucMcHit.h"
#include "McTruth/McEvent.h"
#include "TMath.h"
#include "GaudiKernel/INTupleSvc.h"
#include "GaudiKernel/NTuple.h"
#include "GaudiKernel/Bootstrap.h"
#include "GaudiKernel/IHistogramSvc.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Vector/TwoVector.h"
#include "CLHEP/Geometry/Point3D.h"
#include "EmcRecGeoSvc/EmcRecBarrelGeo.h"
#include "EmcRecGeoSvc/EmcRecGeoSvc.h"
using CLHEP::Hep3Vector;
using CLHEP::Hep2Vector;
using CLHEP::HepLorentzVector;
#ifndef ENABLE_BACKWARDS_COMPATIBILITY
   typedef HepGeom::Point3D<double> HepPoint3D;
#endif
#include "Luminosity/Luminosity.h"
#include "VertexFit/IVertexDbSvc.h"
#include "VertexFit/KinematicFit.h"
#include "VertexFit/KalmanKinematicFit.h"
#include "VertexFit/VertexFit.h"
#include "VertexFit/Helix.h"
#include "ParticleID/ParticleID.h"
#include <vector>
#ifndef PI 
#define PI acos(-1)
#endif
#ifndef DEBUG 
#define DEBUG false
#endif

//const double twopi = 6.2831853;
//const double pi = 3.1415927;
const double mpi = 0.13957;
const double mka = 0.493677;
const double velc = 299.792458;   // tof path unit in mm
typedef std::vector<int> Vint;
typedef std::vector<HepLorentzVector> Vp4;
int Ntot=0,N0=0,N1=0,N2=0,N3=0,N4=0,N5=0,N6=0;
int is_track[40];
int has_mdcinfo[300]={0},is_mdcext[300]={0};

double getPhi  (const double x, const double y);
double getTheta(const double x, const double y, const double z);
/////////////////////////////////////////////////////////////////////////////

Luminosity::Luminosity(const std::string& name, ISvcLocator* pSvcLocator) : Algorithm(name, pSvcLocator) {
	//Declare the properties  
	
	declareProperty("CUT",CUT=0);
	declareProperty("Ecms",Ecms=4.19);
	declareProperty("Vr0cut", m_vr0cut=1.0);
	declareProperty("Vz0cut", m_vz0cut=10.0);
	declareProperty("EnergyThreshold", m_energyThreshold=0.04);
	declareProperty("EnergyThresholdbarrel", m_energyThresholdbarrel=0.025);
	declareProperty("EnergyThresholdendcap", m_energyThresholdendcap=0.05);
	declareProperty("Tracktheta", m_costheta=0.93);
	declareProperty("GammaPhiCut", m_gammaPhiCut=20.0);
	declareProperty("GammaThetaCut", m_gammaThetaCut=20.0);
	declareProperty("GammaAngleCut", m_gammaAngleCut=20.0);
	declareProperty("Mode", m_mode = 0);	// 0->Bhabha, 1->Digamma
	declareProperty("Test1C", m_test1C = 1);
	declareProperty("CheckDedx", m_checkDedx = 1);
	declareProperty("CheckTof",  m_checkTof = 1);

}

/////////////////////////////////////////////////////////////////////////////

StatusCode Luminosity::initialize(){
	MsgStream log(msgSvc(), name());
		
	log << MSG::INFO << "in initialize()" << endmsg;
	
	StatusCode status;
	
	m_tuple1 = ntupleSvc()->book ("FILE1/Bhabha", CLID_ColumnWiseTuple, "N-Tuple");

	status = m_tuple1->addItem ("run",           m_run );
	status = m_tuple1->addItem ("evt",           m_evt );
        // wxfang 
	status = m_tuple1->addItem ("isBB",          m_isBB );
	status = m_tuple1->addItem("index"                 ,    m_indexHit   , 0, 150);
	status = m_tuple1->addItem("ep_mdc_mom"            ,    m_ep_mdc_mom          );
	status = m_tuple1->addItem("ep_mdc_theta"          ,    m_ep_mdc_theta        );
	status = m_tuple1->addItem("ep_mdc_phi"            ,    m_ep_mdc_phi          );
	status = m_tuple1->addItem("em_mdc_mom"            ,    m_em_mdc_mom          );
	status = m_tuple1->addItem("em_mdc_theta"          ,    m_em_mdc_theta        );
	status = m_tuple1->addItem("em_mdc_phi"            ,    m_em_mdc_phi          );
        status = m_tuple1->addItem("ep_thetaModule"        ,    m_ep_thetaModule      );
        status = m_tuple1->addItem("ep_e3x3"               ,    m_ep_e3x3             );
        status = m_tuple1->addItem("ep_e5x5"               ,    m_ep_e5x5             );
        status = m_tuple1->addItem("ep_etof"               ,    m_ep_etof             );
        status = m_tuple1->addItem("ep_etof2x1"            ,    m_ep_etof2x1          );
        status = m_tuple1->addItem("ep_etof2x3"            ,    m_ep_etof2x3          );
        status = m_tuple1->addItem("ep_cluster2ndMoment"   ,    m_ep_cluster2ndMoment );
        status = m_tuple1->addItem("ep_cluster_e"          ,    m_ep_cluster_e        );
        status = m_tuple1->addItem("ep_secondMoment"       ,    m_ep_secondMoment     );
        status = m_tuple1->addItem("ep_latMoment"          ,    m_ep_latMoment        );
        status = m_tuple1->addItem("ep_a20Moment"          ,    m_ep_a20Moment        );
        status = m_tuple1->addItem("ep_a42Moment"          ,    m_ep_a42Moment        );
        status = m_tuple1->addItem("em_thetaModule"        ,    m_em_thetaModule      );
        status = m_tuple1->addItem("em_e3x3"               ,    m_em_e3x3             );
        status = m_tuple1->addItem("em_e5x5"               ,    m_em_e5x5             );
        status = m_tuple1->addItem("em_etof"               ,    m_em_etof             );
        status = m_tuple1->addItem("em_etof2x1"            ,    m_em_etof2x1          );
        status = m_tuple1->addItem("em_etof2x3"            ,    m_em_etof2x3          );
        status = m_tuple1->addItem("em_cluster2ndMoment"   ,    m_em_cluster2ndMoment );
        status = m_tuple1->addItem("em_cluster_e"          ,    m_em_cluster_e        );
        status = m_tuple1->addItem("em_secondMoment"       ,    m_em_secondMoment     );
        status = m_tuple1->addItem("em_latMoment"          ,    m_em_latMoment        );
        status = m_tuple1->addItem("em_a20Moment"          ,    m_em_a20Moment        );
        status = m_tuple1->addItem("em_a42Moment"          ,    m_em_a42Moment        );

	status = m_tuple1->addItem("ep_ext_x"              ,    m_ep_ext_x            );
	status = m_tuple1->addItem("ep_ext_y"              ,    m_ep_ext_y            );
	status = m_tuple1->addItem("ep_ext_z"              ,    m_ep_ext_z            );
	status = m_tuple1->addItem("em_ext_x"              ,    m_em_ext_x            );
	status = m_tuple1->addItem("em_ext_y"              ,    m_em_ext_y            );
	status = m_tuple1->addItem("em_ext_z"              ,    m_em_ext_z            );
	status = m_tuple1->addItem("ep_ext_Px"             ,    m_ep_ext_Px           );
	status = m_tuple1->addItem("ep_ext_Py"             ,    m_ep_ext_Py           );
	status = m_tuple1->addItem("ep_ext_Pz"             ,    m_ep_ext_Pz           );
	status = m_tuple1->addItem("em_ext_Px"             ,    m_em_ext_Px           );
	status = m_tuple1->addItem("em_ext_Py"             ,    m_em_ext_Py           );
	status = m_tuple1->addItem("em_ext_Pz"             ,    m_em_ext_Pz           );
	status = m_tuple1->addItem("ep_emc_shower_energy"  ,    m_ep_emc_shower_energy);
	status = m_tuple1->addItem("ep_emc_shower_x"       ,    m_ep_emc_shower_x     );
	status = m_tuple1->addItem("ep_emc_shower_y"       ,    m_ep_emc_shower_y     );
	status = m_tuple1->addItem("ep_emc_shower_z"       ,    m_ep_emc_shower_z     );
	status = m_tuple1->addItem("ep_M_dtheta"           ,    m_ep_M_dtheta         );
	status = m_tuple1->addItem("ep_M_dphi"             ,    m_ep_M_dphi           );
	status = m_tuple1->addItem("ep_P_dz"               ,    m_ep_P_dz             );
	status = m_tuple1->addItem("ep_P_dphi"             ,    m_ep_P_dphi           );
	status = m_tuple1->addIndexedItem("ep_emc_hit_energy" , m_indexHit,    m_ep_cell_E);
	status = m_tuple1->addItem("em_emc_shower_energy"  ,    m_em_emc_shower_energy);
	status = m_tuple1->addItem("em_emc_shower_x"       ,    m_em_emc_shower_x     );
	status = m_tuple1->addItem("em_emc_shower_y"       ,    m_em_emc_shower_y     );
	status = m_tuple1->addItem("em_emc_shower_z"       ,    m_em_emc_shower_z     );
	status = m_tuple1->addItem("em_M_dtheta"           ,    m_em_M_dtheta         );
	status = m_tuple1->addItem("em_M_dphi"             ,    m_em_M_dphi           );
	status = m_tuple1->addItem("em_P_dz"               ,    m_em_P_dz             );
	status = m_tuple1->addItem("em_P_dphi"             ,    m_em_P_dphi           );
	status = m_tuple1->addIndexedItem("em_emc_hit_energy" , m_indexHit,    m_em_cell_E);

	status = m_tuple1->addItem("ep_track_N"            ,    m_ep_track_N          );
	status = m_tuple1->addItem("ep_shower_N"           ,    m_ep_shower_N         );
	status = m_tuple1->addItem("ep_shower_N_v1"        ,    m_ep_shower_N_v1      );
	status = m_tuple1->addItem("ep_cluster_N"          ,    m_ep_cluster_N        );
	status = m_tuple1->addItem("ep_trkext_N"           ,    m_ep_trkext_N         );
	status = m_tuple1->addItem("em_track_N"            ,    m_em_track_N          );
	status = m_tuple1->addItem("em_shower_N"           ,    m_em_shower_N         );
	status = m_tuple1->addItem("em_shower_N_v1"        ,    m_em_shower_N_v1      );
	status = m_tuple1->addItem("em_cluster_N"          ,    m_em_cluster_N        );
	status = m_tuple1->addItem("em_trkext_N"           ,    m_em_trkext_N         );
	status = m_tuple1->addItem("E_shower_total"        ,    m_E_shower_total      );


	status = m_tuple1->addItem("ep_mc_mom"             ,    m_ep_mc_mom           );
	status = m_tuple1->addItem("ep_mc_theta"           ,    m_ep_mc_theta         );
	status = m_tuple1->addItem("ep_mc_phi"             ,    m_ep_mc_phi           );
	status = m_tuple1->addItem("em_mc_mom"             ,    m_em_mc_mom           );
	status = m_tuple1->addItem("em_mc_theta"           ,    m_em_mc_theta         );
	status = m_tuple1->addItem("em_mc_phi"             ,    m_em_mc_phi           );
	
	status = m_tuple1->addItem("SumHitE"               ,    m_SumHitE           );
	status = m_tuple1->addItem("dumy"                  ,    m_dumy              );
  	//--------end of book--------
  	log << MSG::INFO << "successfully return from initialize()" <<endmsg;
  	return StatusCode::SUCCESS;

}
          
///////////////////////////////////////////////////////////////////////////////////////////
StatusCode Luminosity::execute() {
        Ntot++;
        if(Ntot%1000==0)std::cout<<"for event "<<Ntot<<std::endl;
	MsgStream log(msgSvc(), name());
	//log << MSG::INFO << "in execute()" << endreq;
        init();
      
	SmartDataPtr<Event::EventHeader> eventHeader(eventSvc(),"/Event/EventHeader");
	m_run = eventHeader->runNumber();
	m_evt = eventHeader->eventNumber();
//************************ save some MC info **********************************************************************
	if(true)
        {
            SmartDataPtr<Event::McParticleCol> mcParticleCol(eventSvc(), "/Event/MC/McParticleCol");
	    if (!mcParticleCol){
	 	std::cout << "Could not retrieve McParticelCol" << std::endl;
	  	return StatusCode::FAILURE;
		}
	    else
            {
	        Event::McParticleCol::iterator iter_mc = mcParticleCol->begin();
	        for (; iter_mc != mcParticleCol->end(); iter_mc++){
	        	int pdgid = (*iter_mc)->particleProperty();
		        double px    = (*iter_mc)->initialFourMomentum().x();
		        double py    = (*iter_mc)->initialFourMomentum().y();
		        double pz    = (*iter_mc)->initialFourMomentum().z();
		        double p     = sqrt(px*px+py*py+pz*pz);
		        double e     = (*iter_mc)->initialFourMomentum().e();
		        double theta = (*iter_mc)->initialFourMomentum().theta();
		        double phi   = (*iter_mc)->initialFourMomentum().phi();
                        /*std::cout<<"mc pdg="<< pdgid <<
                        "px="<< px<<
                        "py="<< py<<
                        "pz="<< pz<<
                        "p=" << p <<
                        "e=" << e <<
                        "theta="<< theta<<
                        "phi"<< phi<< std::endl;
                         */
                         if(pdgid==-11)
                         {
                         m_ep_mc_mom    = p    ;
                         m_ep_mc_theta  = theta;
                         m_ep_mc_phi    = phi  ;
                         }
                         else if(pdgid==11)
                         {
                         m_em_mc_mom    = p    ;
                         m_em_mc_theta  = theta;
                         m_em_mc_phi    = phi  ;
                         }
                }
            }

        }

//**********************************************************************************************
	SmartDataPtr<EvtRecEvent> evtRecEvent(eventSvc(), EventModel::EvtRec::EvtRecEvent);
	/* log << MSG::DEBUG <<"ncharg, nneu, tottks = "
	    << evtRecEvent->totalCharged() << " , "
	    << evtRecEvent->totalNeutral() << " , "
	    << evtRecEvent->totalTracks() <<endreq;
        */
	SmartDataPtr<EvtRecTrackCol> evtRecTrkCol(eventSvc(),  EventModel::EvtRec::EvtRecTrackCol);

	
        IEmcRecGeoSvc* iGeoSvc;
        ISvcLocator* svcLocator = Gaudi::svcLocator();
        StatusCode sc = svcLocator->service("EmcRecGeoSvc",iGeoSvc);
        if(sc!=StatusCode::SUCCESS)  cout<<"Error: Can't get EmcRecGeoSvc"<<endl;
        m_indexHit=121;//11*11 

        SmartDataPtr<RecEmcHitCol> emcHitCol(eventSvc(),"/Event/Recon/RecEmcHitCol");
        if (!emcHitCol)
        {   
            log << MSG::FATAL << "Could not find EMC emcHitCol" << endreq;
            return( StatusCode::FAILURE);
        }

        SmartDataPtr<RecEmcShowerCol> emcShowerCol(eventSvc(),"/Event/Recon/RecEmcShowerCol");
        if (!emcShowerCol)
        {   
            log << MSG::FATAL << "Could not find EMC emcShowerCol" << endreq;
            return( StatusCode::FAILURE);
        }
        
        m_ep_shower_N_v1 = emcShowerCol->size();
        m_em_shower_N_v1 = emcShowerCol->size();
        getShowerInfo(emcShowerCol, m_E_shower_total);
//**********************************************************************************************

                int index_ep = -1 ;
                int index_em = -1 ;
                float ep_max = 0 ;
                float em_max = 0 ;
		for(int i = 0; i < evtRecEvent->totalCharged(); i++){
			EvtRecTrackIterator itTrk=evtRecTrkCol->begin() + i;
			if(!(*itTrk)->isMdcTrackValid()) continue;
			RecMdcTrack *mdcTrk = (*itTrk)->mdcTrack();
                        if(mdcTrk->charge() >0){m_ep_track_N++;}
                        else                   {m_em_track_N++;}

			if(!(*itTrk)->isEmcShowerValid()) continue;
			RecEmcShower *emcTrk = (*itTrk)->emcShower();
                        if(mdcTrk->charge() >0){m_ep_shower_N++;}
                        else                   {m_em_shower_N++;}
                        if(emcTrk->energy() < m_energyThreshold) continue;

                        RecEmcCluster *emcCls = emcTrk->getCluster();
                        if(emcCls==0)continue; 
                        if(mdcTrk->charge() >0){m_ep_cluster_N++;}
                        else                   {m_em_cluster_N++;}

			if(!(*itTrk)->isExtTrackValid()) continue;
                        if(mdcTrk->charge() >0){m_ep_trkext_N++;}
                        else                   {m_em_trkext_N++;}
                        
                        if     (mdcTrk->charge() >0 && mdcTrk->p() > ep_max){index_ep = i; ep_max = mdcTrk->p();}
                        else if(mdcTrk->charge() <0 && mdcTrk->p() > em_max){index_em = i; em_max = mdcTrk->p();}
                        }
                 if( index_ep !=-1 && index_em !=-1 ){
     
                     EvtRecTrackIterator itTrk_ep=evtRecTrkCol->begin() + index_ep;
                     EvtRecTrackIterator itTrk_em=evtRecTrkCol->begin() + index_em;
		     RecMdcTrack *mdcTrk_ep = (*itTrk_ep)->mdcTrack();
		     RecMdcTrack *mdcTrk_em = (*itTrk_em)->mdcTrack();
		     RecEmcShower *emcTrk_ep = (*itTrk_ep)->emcShower();
		     RecEmcShower *emcTrk_em = (*itTrk_em)->emcShower();
                     RecEmcID ep_id = Identifier(emcTrk_ep->cellId());
                     RecEmcID em_id = Identifier(emcTrk_em->cellId());
                     m_ep_thetaModule = EmcID::theta_module(ep_id);
                     m_em_thetaModule = EmcID::theta_module(em_id);
                     RecEmcCluster *emcCls_ep = emcTrk_ep->getCluster();
                     RecEmcCluster *emcCls_em = emcTrk_em->getCluster();
                     RecExtTrack *extTrk_ep = (*itTrk_ep)->extTrack(); 
                     RecExtTrack *extTrk_em = (*itTrk_em)->extTrack(); 


                     if((*itTrk_ep)->isTofTrackValid()){
                       SmartRefVector<RecTofTrack> recTofTrackVec=(*itTrk_ep)->tofTrack();
                       if(!recTofTrackVec.empty()) m_ep_etof=recTofTrackVec[0]->energy();
                       if(m_ep_etof>100.)m_ep_etof=0;
                     }
                     if((*itTrk_em)->isTofTrackValid()){
                       SmartRefVector<RecTofTrack> recTofTrackVec=(*itTrk_em)->tofTrack();
                       if(!recTofTrackVec.empty()) m_em_etof=recTofTrackVec[0]->energy();
                       if(m_em_etof>100.)m_em_etof=0;
                     }

                     ////////// some shower variables //////////////////////
                     m_ep_e3x3            =emcTrk_ep->e3x3();
                     m_ep_e5x5            =emcTrk_ep->e5x5();
                     m_ep_etof2x1         =emcTrk_ep->getETof2x1();
                     m_ep_etof2x3         =emcTrk_ep->getETof2x3();
                     m_ep_cluster2ndMoment=emcTrk_ep->getCluster()->getSecondMoment();
                     m_ep_cluster_e       =emcTrk_ep->getCluster()->getEnergy();
                     m_ep_secondMoment    =emcTrk_ep->secondMoment();
                     m_ep_latMoment       =emcTrk_ep->latMoment();
                     m_ep_a20Moment       =emcTrk_ep->a20Moment();
                     m_ep_a42Moment       =emcTrk_ep->a42Moment();
                     m_em_e3x3            =emcTrk_em->e3x3();
                     m_em_e5x5            =emcTrk_em->e5x5();
                     m_em_etof2x1         =emcTrk_em->getETof2x1();
                     m_em_etof2x3         =emcTrk_em->getETof2x3();
                     m_em_cluster2ndMoment=emcTrk_em->getCluster()->getSecondMoment();
                     m_em_cluster_e       =emcTrk_em->getCluster()->getEnergy();
                     m_em_secondMoment    =emcTrk_em->secondMoment();
                     m_em_latMoment       =emcTrk_em->latMoment();
                     m_em_a20Moment       =emcTrk_em->a20Moment();
                     m_em_a42Moment       =emcTrk_em->a42Moment();

                     //std::cout<<"em trk x="<<(extTrk_em->emcPosition()).x()<<",y="<<(extTrk_em->emcPosition()).y()<<",z="<<(extTrk_em->emcPosition()).z()<<std::endl; 
                     /*
                     m_ep_ext_x = (extTrk_ep->emcPosition()).x();
                     m_ep_ext_y = (extTrk_ep->emcPosition()).y();
                     m_ep_ext_z = (extTrk_ep->emcPosition()).z();
                     m_em_ext_x = (extTrk_em->emcPosition()).x();
                     m_em_ext_y = (extTrk_em->emcPosition()).y();
                     m_em_ext_z = (extTrk_em->emcPosition()).z();
                     m_ep_ext_Px = (extTrk_ep->emcMomentum()).x();
                     m_ep_ext_Py = (extTrk_ep->emcMomentum()).y();
                     m_ep_ext_Pz = (extTrk_ep->emcMomentum()).z();
                     m_em_ext_Px = (extTrk_em->emcMomentum()).x();
                     m_em_ext_Py = (extTrk_em->emcMomentum()).y();
                     m_em_ext_Pz = (extTrk_em->emcMomentum()).z();
                     */
                     // e, mu, pi, k, proton for 0, 1, 2, 3, 4
                     m_ep_ext_x = (extTrk_ep->emcPosition(0)).x();
                     m_ep_ext_y = (extTrk_ep->emcPosition(0)).y();
                     m_ep_ext_z = (extTrk_ep->emcPosition(0)).z();
                     m_em_ext_x = (extTrk_em->emcPosition(0)).x();
                     m_em_ext_y = (extTrk_em->emcPosition(0)).y();
                     m_em_ext_z = (extTrk_em->emcPosition(0)).z();
                     m_ep_ext_Px =(extTrk_ep->emcMomentum(0)).x();
                     m_ep_ext_Py =(extTrk_ep->emcMomentum(0)).y();
                     m_ep_ext_Pz =(extTrk_ep->emcMomentum(0)).z();
                     m_em_ext_Px =(extTrk_em->emcMomentum(0)).x();
                     m_em_ext_Py =(extTrk_em->emcMomentum(0)).y();
                     m_em_ext_Pz =(extTrk_em->emcMomentum(0)).z();
                     //int get_ep_info = getInfo(iGeoSvc, m_ep_ext_x, m_ep_ext_y, m_ep_ext_z, m_ep_ext_Px, m_ep_ext_Py, m_ep_ext_Pz, emcCls_ep, m_ep_M_dtheta, m_ep_M_dphi, m_ep_P_dz, m_ep_P_dphi, m_ep_cell_E);
                     int get_ep_info = getInfo(iGeoSvc, m_ep_ext_x, m_ep_ext_y, m_ep_ext_z, m_ep_ext_Px, m_ep_ext_Py, m_ep_ext_Pz, emcHitCol, m_ep_M_dtheta, m_ep_M_dphi, m_ep_P_dz, m_ep_P_dphi, m_ep_cell_E, m_SumHitE);
                     int get_em_info = getInfo(iGeoSvc, m_em_ext_x, m_em_ext_y, m_em_ext_z, m_em_ext_Px, m_em_ext_Py, m_em_ext_Pz, emcHitCol, m_em_M_dtheta, m_em_M_dphi, m_em_P_dz, m_em_P_dphi, m_em_cell_E, m_dumy);
                     if(DEBUG)std::cout<<"ep info:"<<"M_dtheta="<<m_ep_M_dtheta<<",M_dphi="<<m_ep_M_dphi<<",P_dz="<<m_ep_P_dz<<",P_dphi="<<m_ep_P_dphi<<std::endl;
                     if(DEBUG)std::cout<<"em info:"<<"M_dtheta="<<m_em_M_dtheta<<",M_dphi="<<m_em_M_dphi<<",P_dz="<<m_em_P_dz<<",P_dphi="<<m_em_P_dphi<<std::endl;
                     if(get_ep_info==0 || get_em_info==0) return StatusCode::SUCCESS;
                     m_ep_mdc_mom       =mdcTrk_ep->p(); 
                     m_ep_mdc_theta     =mdcTrk_ep->theta(); 
                     m_ep_mdc_phi       =mdcTrk_ep->phi(); 
                     m_ep_emc_shower_energy = emcTrk_ep->energy(); 
                     m_ep_emc_shower_x = emcTrk_ep->x();
                     m_ep_emc_shower_y = emcTrk_ep->y();
                     m_ep_emc_shower_z = emcTrk_ep->z();
                     float theta_barrel_max = 0.81673 ;// this is the max theta of cell f center in barrel, if for max theta of xtal it is 0.82584
                     float cell_frontCenter_z = 136.14 ;// cm, for barrel
                     //int m_ep_isBarrel = fabs(cos(sqrt(m_ep_emc_shower_x*m_ep_emc_shower_x + m_ep_emc_shower_y*m_ep_emc_shower_y)/m_ep_emc_shower_z)) < 0.83 ? 1 : 0;
                     //int m_ep_isBarrel = fabs(m_ep_ext_z/sqrt(m_ep_ext_x*m_ep_ext_x + m_ep_ext_y*m_ep_ext_y + m_ep_ext_z*m_ep_ext_z)) < theta_barrel_max ? 1 : 0;
                     int m_ep_isBarrel = fabs(m_ep_ext_z) < cell_frontCenter_z ? 1 : 0;

                     /*
		     m_ep_indexHit = emcTrk_ep->numHits();
                     int i_tmp = 0;
                     for(RecEmcHitMap::const_iterator it = emcCls_ep->Begin(); it != emcCls_ep->End(); it++){
                         const std::pair<const Identifier, RecEmcHit> tmp_HitMap = *it;
                         //int  tmp_HitID = tmp_HitMap.first;
                         RecEmcHit tmp_EmcHit = tmp_HitMap.second;
                         RecEmcID tmp_cellID = tmp_EmcHit.getCellId();
                         
                         HepPoint3D xtal0,xtal1,xtal2,xtal3;
                         xtal0=iGeoSvc->GetCrystalPoint(tmp_cellID,0);
                         xtal1=iGeoSvc->GetCrystalPoint(tmp_cellID,1);
                         xtal2=iGeoSvc->GetCrystalPoint(tmp_cellID,2);
                         xtal3=iGeoSvc->GetCrystalPoint(tmp_cellID,3);
                         unsigned int npart,ntheta,nphi;
                         npart  = EmcID::barrel_ec(tmp_cellID);
                         ntheta = EmcID::theta_module(tmp_cellID);
                         nphi   = EmcID::phi_module(tmp_cellID);
                         //std::cout<<"----------: "<<std::endl;
                         //std::cout<<"npart :"<<npart<<std::endl;
                         //std::cout<<"ntheta:"<<ntheta<<std::endl;
                         //std::cout<<"nphi  :"<<nphi<<std::endl;
                         //if(nphi==0 || nphi==120)std::cout<<"nphi  :"<<nphi<<std::endl;
                         //if(ntheta==0 || ntheta>=44)std::cout<<"ntheta  :"<<ntheta<<std::endl;
                         //std::cout<<"xtal0 "<<xtal0.theta()<<"phi "<<xtal0.phi()*180/3.14159265<<std::endl;
                         //std::cout<<"xtal1 "<<xtal1.theta()<<"phi "<<xtal1.phi()*180/3.14159265<<std::endl;
                         //std::cout<<"xtal2 "<<xtal2.theta()<<"phi "<<xtal2.phi()*180/3.14159265<<std::endl;
                         //std::cout<<"xtal3 "<<xtal3.theta()<<"phi "<<xtal3.phi()*180/3.14159265<<std::endl;
                         float tmp_HitE = tmp_EmcHit.getEnergy();
                         HepPoint3D cellCenter = tmp_EmcHit.getCenter();
                         HepPoint3D cellFCenter =tmp_EmcHit.getFrontCenter();
                         m_ep_emc_hit_energy[i_tmp] = tmp_HitE;
                         m_ep_emc_hit_x     [i_tmp] = cellCenter.x();
                         m_ep_emc_hit_y     [i_tmp] = cellCenter.y();
                         m_ep_emc_hit_z     [i_tmp] = cellCenter.z();
                         m_ep_emc_hit_Fx    [i_tmp] = cellFCenter.x();
                         m_ep_emc_hit_Fy    [i_tmp] = cellFCenter.y();
                         m_ep_emc_hit_Fz    [i_tmp] = cellFCenter.z();
                         i_tmp ++ ;
                         }
                     */
		     
                     m_em_mdc_mom       =mdcTrk_em->p(); 
                     m_em_mdc_theta     =mdcTrk_em->theta(); 
                     m_em_mdc_phi       =mdcTrk_em->phi(); 
                     m_em_emc_shower_energy = emcTrk_em->energy(); 
                     m_em_emc_shower_x = emcTrk_em->x();
                     m_em_emc_shower_y = emcTrk_em->y();
                     m_em_emc_shower_z = emcTrk_em->z();
                     //int m_em_isBarrel = fabs(cos(sqrt(m_em_emc_shower_x*m_em_emc_shower_x + m_em_emc_shower_y*m_em_emc_shower_y)/m_em_emc_shower_z)) < 0.83 ? 1 : 0;
                     //int m_em_isBarrel = fabs(m_em_ext_z/sqrt(m_em_ext_x*m_em_ext_x + m_em_ext_y*m_em_ext_y + m_em_ext_z*m_em_ext_z)) < theta_barrel_max ? 1 : 0;
                     int m_em_isBarrel = fabs(m_em_ext_z) < cell_frontCenter_z ? 1 : 0;
                     /*
		     m_em_indexHit = emcTrk_em->numHits();
                     i_tmp = 0;
                     for(RecEmcHitMap::const_iterator it = emcCls_em->Begin(); it != emcCls_em->End(); it++){
                         const std::pair<const Identifier, RecEmcHit> tmp_HitMap = *it;
                         RecEmcHit tmp_EmcHit = tmp_HitMap.second;
                         float tmp_HitE = tmp_EmcHit.getEnergy();
                         HepPoint3D cellCenter = tmp_EmcHit.getCenter();
                         HepPoint3D cellFCenter =tmp_EmcHit.getFrontCenter();
                         m_em_emc_hit_energy[i_tmp] = tmp_HitE;
                         m_em_emc_hit_x     [i_tmp] = cellCenter.x();
                         m_em_emc_hit_y     [i_tmp] = cellCenter.y();
                         m_em_emc_hit_z     [i_tmp] = cellCenter.z();
                         m_em_emc_hit_Fx    [i_tmp] = cellFCenter.x();
                         m_em_emc_hit_Fy    [i_tmp] = cellFCenter.y();
                         m_em_emc_hit_Fz    [i_tmp] = cellFCenter.z();
                         i_tmp ++;
                      }
                      */
                if ( (m_ep_isBarrel + m_em_isBarrel) ==2){ m_isBB=1; N2++; }
                else if ( (m_ep_isBarrel + m_em_isBarrel) ==1)N3++;	
                else if ( (m_ep_isBarrel + m_em_isBarrel) ==0)N4++;	
                if( (m_ep_isBarrel + m_em_isBarrel) !=2 ) return StatusCode::SUCCESS;
		N1++;
	    	m_tuple1 -> write();
		}//if 
                else
                {
                    N5++;
                }
    	        return StatusCode::SUCCESS;
	
}

//////////////////////////////////////////////////////////////////////////////////////////////

int Luminosity::getShowerInfo(RecEmcShowerCol* Col, NTuple::Item<double>& E_shower_total) {
    for(RecEmcShowerCol::iterator it=Col->begin();it!= Col->end();it++)
    {
        E_shower_total = E_shower_total + (*it)->energy();
    }
    return 1;
}


//int Luminosity::getInfo(const IEmcRecGeoSvc* iGeoSvc, const double& x, const double& y, const double& z, const double& px, const double& py, const double& pz, const RecEmcCluster* clus, NTuple::Item<double>& M_dtheta, NTuple::Item<double>& M_dphi, NTuple::Item<double>& P_dz, NTuple::Item<double>& P_dphi, NTuple::Array<double>& cell_E) {
int Luminosity::getInfo(const IEmcRecGeoSvc* iGeoSvc, const double& x, const double& y, const double& z, const double& px, const double& py, const double& pz, RecEmcHitCol* EmcHitCol, NTuple::Item<double>& M_dtheta, NTuple::Item<double>& M_dphi, NTuple::Item<double>& P_dz, NTuple::Item<double>& P_dphi, NTuple::Array<double>& cell_E, NTuple::Item<double>& SumHitE) {
    float dist_min = 999;
    int center_it=-1;
    int counter = 0;    
    //RecEmcHitMap::const_iterator cen_it; 
    RecEmcHitCol::iterator cen_it;
    unsigned int npart,ntheta,nphi;
    HepPoint3D xtal0,xtal1,xtal2,xtal3;
    //for(RecEmcHitMap::const_iterator it = clus->Begin(); it != clus->End(); it++)
    for(RecEmcHitCol::iterator it=EmcHitCol->begin();it!= EmcHitCol->end();it++)
    {
        SumHitE = SumHitE + (*it)->getEnergy();
        RecEmcID id = (*it)->getCellId();
        RecEmcHit aHit;
        aHit.CellId(id);
        aHit.Energy((*it)->getEnergy());
        aHit.Time(0);
        HepPoint3D cellFCenter = aHit.getFrontCenter();
        float dist = sqrt( pow(cellFCenter.x()-x,2) + pow(cellFCenter.y()-y,2) + pow(cellFCenter.z()-z,2));
        if(dist < dist_min)
        {
            dist_min = dist;
            center_it = counter;       
            cen_it = it;
        } 
        counter ++;      
    }
    if(dist_min <= 2.5*sqrt(2) && center_it !=-1)
    //if(dist_min <= 100 && center_it !=-1)
    {
        if(DEBUG)std::cout<<"find center"<<std::endl;
        //RecEmcHitMap::const_iterator it = cen_it;
        RecEmcHitCol::iterator it = cen_it;
        //const std::pair<const Identifier, RecEmcHit> tmp_HitMap = *it;
        RecEmcID id = (*it)->getCellId();
        RecEmcHit tmp_EmcHit;
        tmp_EmcHit.CellId(id);
        tmp_EmcHit.Energy((*it)->getEnergy());
        tmp_EmcHit.Time(0);
        //RecEmcHit tmp_EmcHit   = tmp_HitMap.second;
        RecEmcID tmp_cellID    = tmp_EmcHit.getCellId();
        HepPoint3D cellFCenter = tmp_EmcHit.getFrontCenter();
        HepPoint3D cellCenter  = tmp_EmcHit.getCenter();
        HepPoint3D cellDirection = cellCenter - cellFCenter ;

        double M_phi = getPhi(px, py);
        double M_theta = getTheta(px, py, pz);
        double cellDir_phi   = getPhi(cellDirection.x(), cellDirection.y());
        double cellDir_theta = getTheta(cellDirection.x(), cellDirection.y(), cellDirection.z());
        M_dphi   = M_phi - cellDir_phi ;
        if(M_dphi< -180) M_dphi = M_dphi + 360;
        else if(M_dphi > 180) M_dphi = M_dphi - 360;
        M_dtheta = M_theta - cellDir_theta ;
        P_dz = z - cellFCenter.z();
        P_dphi = getPhi(x, y) - getPhi(cellFCenter.x(),cellFCenter.y());
        if(P_dphi< -180) P_dphi = P_dphi + 360;
        else if(P_dphi > 180) P_dphi = P_dphi - 360;
        //xtal0=iGeoSvc->GetCrystalPoint(tmp_cellID,0);
        //xtal1=iGeoSvc->GetCrystalPoint(tmp_cellID,1);
        //xtal2=iGeoSvc->GetCrystalPoint(tmp_cellID,2);
        //xtal3=iGeoSvc->GetCrystalPoint(tmp_cellID,3);


        npart  = EmcID::barrel_ec(tmp_cellID);
        ntheta = EmcID::theta_module(tmp_cellID);// within 0-43
        nphi   = EmcID::phi_module(tmp_cellID);  //within 0-119
        int id_theta[121]={-1};
        int id_phi  [121]={-1};
        for(int i=0; i<11; i++)
        {
            unsigned int itheta = (i-5)+ntheta ;
            for(int j=0; j<11; j++)
            {
                unsigned iphi = (5-j)+nphi ;
                if(iphi> 119) iphi = iphi - 120 ;
                else if(iphi< 0) iphi = iphi + 120 ;
                id_theta[i*11+j]=itheta   ;
                id_phi  [i*11+j]=iphi    ;
            }
        }
        for(int i=0; i<121; i++)
        {
            cell_E[i] = 0;
        }
        //for(RecEmcHitMap::const_iterator it = clus->Begin(); it != clus->End(); it++)
        for(RecEmcHitCol::iterator it=EmcHitCol->begin();it!= EmcHitCol->end();it++)
        {
            RecEmcID id = (*it)->getCellId();
            RecEmcHit tmp_Hit;
            tmp_Hit.CellId(id);
            tmp_Hit.Energy((*it)->getEnergy());
            tmp_Hit.Time(0);
            //const std::pair<const Identifier, RecEmcHit> tmp = *it;
            //RecEmcHit tmp_Hit   = tmp.second;
            RecEmcID tmp_ID    = tmp_Hit.getCellId();
            double   tmp_E     = tmp_Hit.getEnergy();
            unsigned int ipart  = EmcID::barrel_ec(tmp_ID);
            unsigned int itheta = EmcID::theta_module(tmp_ID);// within 0-43 from positive Z to negative Z
            unsigned int iphi   = EmcID::phi_module(tmp_ID);  //within 0-119
            /*
            if(ipart==1 && (itheta==0|| itheta==43))
            //if(ipart==0 && (itheta==0|| itheta==5))
            {
                HepPoint3D cellFCenter = tmp_Hit.getFrontCenter();
                double cell_theta = getTheta(cellFCenter.x(), cellFCenter.y(), cellFCenter.z());
                xtal0=iGeoSvc->GetCrystalPoint(tmp_ID,0);
                xtal1=iGeoSvc->GetCrystalPoint(tmp_ID,1);
                xtal2=iGeoSvc->GetCrystalPoint(tmp_ID,2);
                xtal3=iGeoSvc->GetCrystalPoint(tmp_ID,3);
                double cell_theta0 = getTheta(xtal0.x(), xtal0.y(), xtal0.z());
                double cell_theta1 = getTheta(xtal1.x(), xtal1.y(), xtal1.z());
                double cell_theta2 = getTheta(xtal2.x(), xtal2.y(), xtal2.z());
                double cell_theta3 = getTheta(xtal3.x(), xtal3.y(), xtal3.z());
                std::cout<<"cell theta="<<cell_theta<<", theta0="<<cell_theta0<<", theta1="<<cell_theta1<<", theta2="<<cell_theta2<<", theta3="<<cell_theta3<<std::endl; 
                std::cout<<"cell z="<<cellFCenter.z()<<",xtal0.z="<<xtal0.z()<<", xtal1.z="<<xtal1.z()<<", xtal2.z="<<xtal2.z()<<", xtal3.z="<<xtal3.z()<<std::endl; 
            }
            */ 
            if(ipart !=1)continue;
            for(int i=0; i<121; i++)
            {
                if(id_theta[i]==itheta && id_phi[i]==iphi)
                {
                    cell_E[i] = tmp_E;
                    break;
                }
            }
        }
        
    }
    else return 0;
    return 1;
}

int Luminosity::init() {



	 m_ep_mdc_mom           = 0 ;
	 m_ep_mdc_theta         = 0 ;
	 m_ep_mdc_phi           = 0 ;
	 m_em_mdc_mom           = 0 ;
	 m_em_mdc_theta         = 0 ;
	 m_em_mdc_phi           = 0 ;
         m_ep_thetaModule       = 0 ;
         m_ep_e3x3              = 0 ;
         m_ep_e5x5              = 0 ;
         m_ep_etof              = 0 ;
         m_ep_etof2x1           = 0 ;
         m_ep_etof2x3           = 0 ;
         m_ep_cluster2ndMoment  = 0 ;
         m_ep_cluster_e         = 0 ;
         m_ep_secondMoment      = 0 ;
         m_ep_latMoment         = 0 ;
         m_ep_a20Moment         = 0 ;
         m_ep_a42Moment         = 0 ;
         m_em_thetaModule       = 0 ;
         m_em_e3x3              = 0 ;
         m_em_e5x5              = 0 ;
         m_em_etof2x1           = 0 ;
         m_em_etof              = 0 ;
         m_em_etof2x3           = 0 ;
         m_em_cluster2ndMoment  = 0 ;
         m_em_cluster_e         = 0 ;
         m_em_secondMoment      = 0 ;
         m_em_latMoment         = 0 ;
         m_em_a20Moment         = 0 ;
         m_em_a42Moment         = 0 ;

	 m_ep_ext_x             = 0 ;
	 m_ep_ext_y             = 0 ;
	 m_ep_ext_z             = 0 ;
	 m_em_ext_x             = 0 ;
	 m_em_ext_y             = 0 ;
	 m_em_ext_z             = 0 ;
	 m_ep_ext_Px            = 0 ;
	 m_ep_ext_Py            = 0 ;
	 m_ep_ext_Pz            = 0 ;
	 m_em_ext_Px            = 0 ;
	 m_em_ext_Py            = 0 ;
	 m_em_ext_Pz            = 0 ;
	 m_ep_emc_shower_energy = 0 ;
	 m_ep_emc_shower_x      = 0 ;
	 m_ep_emc_shower_y      = 0 ;
	 m_ep_emc_shower_z      = 0 ;
	 m_ep_M_dtheta          = 0 ;
	 m_ep_M_dphi            = 0 ;
	 m_ep_P_dz              = 0 ;
	 m_ep_P_dphi            = 0 ;
	 m_em_emc_shower_energy = 0 ;
	 m_em_emc_shower_x      = 0 ;
	 m_em_emc_shower_y      = 0 ;
	 m_em_emc_shower_z      = 0 ;
	 m_em_M_dtheta          = 0 ;
	 m_em_M_dphi            = 0 ;
	 m_em_P_dz              = 0 ;
	 m_em_P_dphi            = 0 ;

         m_ep_track_N           = 0 ;
         m_ep_shower_N          = 0 ;
         m_ep_shower_N_v1       = 0 ;
         m_ep_cluster_N         = 0 ;
         m_ep_trkext_N          = 0 ;
         m_em_track_N           = 0 ;
         m_em_shower_N          = 0 ;
         m_em_shower_N_v1       = 0 ;
         m_em_cluster_N         = 0 ;
         m_em_trkext_N          = 0 ;
         m_E_shower_total       = 0 ;       

	 m_ep_mc_mom            = 0 ;
	 m_ep_mc_theta          = 0 ;
	 m_ep_mc_phi            = 0 ;
	 m_em_mc_mom            = 0 ;
	 m_em_mc_theta          = 0 ;
	 m_em_mc_phi            = 0 ;
         
	 m_SumHitE              = 0 ;
	 m_dumy                 = 0 ;

     //m_ep_cell_E  ={0};
     //m_em_cell_E  ={0};
     

}
//////////////////////////////////////////////////////////////////////////////////////////////
StatusCode Luminosity::finalize() {
	MsgStream log(msgSvc(), name());
	log << MSG::INFO << "in finalize()" << endmsg;
	cout<<"Total:	         	"<<Ntot<<endl;
	cout<<"Events passing cut:	"<<N1<<endl;
	cout<<"N BB:                  	"<<N2<<endl;
	cout<<"N BE:                  	"<<N3<<endl;
	cout<<"N EE:                  	"<<N4<<endl;
	cout<<"N not (e+ && e-):       	"<<N5<<endl;
	return StatusCode::SUCCESS;
}


double getPhi(const double x, const double y)
{
    if     (x==0 && y>0) return 90;
    else if(x==0 && y<0) return 270;
    else if(x==0 && y==0) return 0;
    double phi = atan(y/x)*180/PI;
    if                 (x<0) phi = phi + 180;
    else if     (x>0 && y<0) phi = phi + 360;
    return phi;
}


double getTheta(const double x, const double y, const double z)
{
    double pre = sqrt(x*x + y*y);
    double theta = z != 0 ? atan(pre/z)*180/PI : 90;
    if(theta<0) theta = 180 + theta;  
    return theta; 
}
