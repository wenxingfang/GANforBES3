#ifndef Physics_Analysis_Luminosity_H
#define Physics_Analysis_Luminosity_H

#include "GaudiKernel/AlgFactory.h"
#include "GaudiKernel/Algorithm.h"
#include "GaudiKernel/NTuple.h"
#include "GaudiKernel/SmartDataPtr.h"
#include "CLHEP/Geometry/Point3D.h"
#include "EmcRecGeoSvc/EmcRecBarrelGeo.h"
#include "EmcRecGeoSvc/EmcRecGeoSvc.h"
#include "EvtRecEvent/EvtRecTrack.h"
//#include "VertexFit/ReadBeamParFromDb.h"

class Luminosity : public Algorithm {

public:
	Luminosity(const std::string& name, ISvcLocator* pSvcLocator);
	StatusCode initialize();
	StatusCode execute();
	StatusCode finalize();
        //int getInfo(const IEmcRecGeoSvc* iGeoSvc, const double& x, const double& y, const double& z, const double& px, const double& py, const double& pz, const RecEmcCluster* clus, NTuple::Item<double>& M_dtheta, NTuple::Item<double>& M_dphi, NTuple::Item<double>& P_dz, NTuple::Item<double>& P_dphi, NTuple::Array<double>& cell_E) ;
        //int getInfo(const IEmcRecGeoSvc* iGeoSvc, const double& x, const double& y, const double& z, const double& px, const double& py, const double& pz, const SmartDataPtr<RecEmcHitCol>& EmcHitCol, NTuple::Item<double>& M_dtheta, NTuple::Item<double>& M_dphi, NTuple::Item<double>& P_dz, NTuple::Item<double>& P_dphi, NTuple::Array<double>& cell_E) ;
        //int getInfo(const IEmcRecGeoSvc* iGeoSvc, const double& x, const double& y, const double& z, const double& px, const double& py, const double& pz, RecEmcHitCol* EmcHitCol, NTuple::Item<double>& M_dtheta, NTuple::Item<double>& M_dphi, NTuple::Item<double>& P_dz, NTuple::Item<double>& P_dphi, NTuple::Array<double>& cell_E) ;
        int getShowerInfo(RecEmcShowerCol* Col, NTuple::Item<double>& E_shower_total) ;
        int getInfo(const IEmcRecGeoSvc* iGeoSvc, const double& x, const double& y, const double& z, const double& px, const double& py, const double& pz, RecEmcHitCol* EmcHitCol, NTuple::Item<double>& M_dtheta, NTuple::Item<double>& M_dphi, NTuple::Item<double>& P_dz, NTuple::Item<double>& P_dphi, NTuple::Array<double>& cell_E, NTuple::Item<double>& SumHitE) ;
        int init();

private:
	bool CUT;
	double Ecms;
	double m_vr0cut;
	double m_vz0cut;
	double m_energyThreshold;
	double m_gammaPhiCut;
	double m_gammaThetaCut;
	double m_gammaAngleCut;
	double m_energyThresholdbarrel;
	double m_energyThresholdendcap;
	double m_costheta;
	int m_mode;
	int m_test1C;
	int m_checkDedx;
	int m_checkTof;
	// define Ntuples here

	NTuple::Tuple*  m_tuple1;      // charged track vertex

        NTuple::Item<int> m_run          ;
        NTuple::Item<int> m_evt          ;
        // wx 
        NTuple::Item<int> m_isBB                 ;
        NTuple::Item<int> m_indexHit          ;
        NTuple::Item<double> m_ep_ext_x          ; 
        NTuple::Item<double> m_ep_ext_y          ; 
        NTuple::Item<double> m_ep_ext_z          ; 
        NTuple::Item<double> m_em_ext_x          ; 
        NTuple::Item<double> m_em_ext_y          ; 
        NTuple::Item<double> m_em_ext_z          ; 

        NTuple::Item<double> m_ep_ext_Px         ; 
        NTuple::Item<double> m_ep_ext_Py         ; 
        NTuple::Item<double> m_ep_ext_Pz         ; 
        NTuple::Item<double> m_em_ext_Px         ; 
        NTuple::Item<double> m_em_ext_Py         ; 
        NTuple::Item<double> m_em_ext_Pz         ; 

        NTuple::Item<double> m_ep_mdc_mom           ;
        NTuple::Item<double> m_ep_mdc_theta         ;
        NTuple::Item<double> m_ep_mdc_phi           ;
        NTuple::Item<double> m_em_mdc_mom           ;
        NTuple::Item<double> m_em_mdc_theta         ;
        NTuple::Item<double> m_em_mdc_phi           ;

        NTuple::Item<double> m_ep_emc_shower_energy ;
        NTuple::Item<double> m_ep_emc_shower_x      ;
        NTuple::Item<double> m_ep_emc_shower_y      ;
        NTuple::Item<double> m_ep_emc_shower_z      ;

        NTuple::Array<double> m_ep_cell_E   ;
        NTuple::Item<double> m_ep_M_dtheta      ;
        NTuple::Item<double> m_ep_M_dphi        ;
        NTuple::Item<double> m_ep_P_dz          ;
        NTuple::Item<double> m_ep_P_dphi        ;

        NTuple::Item<double> m_em_emc_shower_energy ;
        NTuple::Item<double> m_em_emc_shower_x      ;
        NTuple::Item<double> m_em_emc_shower_y      ;
        NTuple::Item<double> m_em_emc_shower_z      ;

        NTuple::Array<double> m_em_cell_E   ;
        NTuple::Item<double> m_em_M_dtheta      ;
        NTuple::Item<double> m_em_M_dphi        ;
        NTuple::Item<double> m_em_P_dz          ;
        NTuple::Item<double> m_em_P_dphi        ;


        NTuple::Item<double> m_ep_thetaModule     ;
        NTuple::Item<double> m_ep_e3x3            ;
        NTuple::Item<double> m_ep_e5x5            ;
        NTuple::Item<double> m_ep_etof            ;
        NTuple::Item<double> m_ep_etof2x1         ;
        NTuple::Item<double> m_ep_etof2x3         ;
        NTuple::Item<double> m_ep_cluster2ndMoment;
        NTuple::Item<double> m_ep_cluster_e       ;
        NTuple::Item<double> m_ep_secondMoment    ;
        NTuple::Item<double> m_ep_latMoment       ;
        NTuple::Item<double> m_ep_a20Moment       ;
        NTuple::Item<double> m_ep_a42Moment       ;
        NTuple::Item<double> m_em_thetaModule     ;
        NTuple::Item<double> m_em_e3x3            ;
        NTuple::Item<double> m_em_e5x5            ;
        NTuple::Item<double> m_em_etof            ;
        NTuple::Item<double> m_em_etof2x1         ;
        NTuple::Item<double> m_em_etof2x3         ;
        NTuple::Item<double> m_em_cluster2ndMoment;
        NTuple::Item<double> m_em_cluster_e       ;
        NTuple::Item<double> m_em_secondMoment    ;
        NTuple::Item<double> m_em_latMoment       ;
        NTuple::Item<double> m_em_a20Moment       ;
        NTuple::Item<double> m_em_a42Moment       ;

        NTuple::Item<double> m_ep_mc_mom   ;
        NTuple::Item<double> m_ep_mc_theta ;
        NTuple::Item<double> m_ep_mc_phi   ;
        NTuple::Item<double> m_em_mc_mom   ;
        NTuple::Item<double> m_em_mc_theta ;
        NTuple::Item<double> m_em_mc_phi   ;

        NTuple::Item<double> m_ep_track_N   ;
        NTuple::Item<double> m_ep_shower_N  ;
        NTuple::Item<double> m_ep_shower_N_v1;
        NTuple::Item<double> m_ep_cluster_N ;
        NTuple::Item<double> m_ep_trkext_N  ;
        NTuple::Item<double> m_em_track_N   ;
        NTuple::Item<double> m_em_shower_N  ;
        NTuple::Item<double> m_em_shower_N_v1;
        NTuple::Item<double> m_em_cluster_N ;
        NTuple::Item<double> m_em_trkext_N  ;

        NTuple::Item<double> m_E_shower_total;


        NTuple::Item<double> m_SumHitE   ;
        NTuple::Item<double> m_dumy   ;
};



#endif
