#include "/home/amit/Yorick/STECKMAP/Pierre/POP/pop_paths.i"
#include "/home/amit/Yorick/STECKMAP/Pierre/POP/sfit.i"

// Fitting parameters

ages=[5E8,13.6E9];      // ages domain in yr
wavel=[4050.,5500.];    // wavelength domain. If too wide, the output will be restricted to what is supplied by the chosen model.
basisfile="MILES";      // choices: "BC03", "PHR", "MILES"
nbins=30;               // number of age bins
//dlambda=2.5;             // sampling of the basis. If void, use the original sampling of the chosen model.
kin=1;                  // kin=0 -> no kinematics, kin=1 -> LOSVD search

epar=3;                 // with continuum matching (to deal with flux calibration errors)
nde=15;                 // number of control points for continuum matching
vlim=[-1000.,1000.];    // wavelength domain for the LOSVD
meval=500;              // maximum number of evalutations during the optimisation
//RMASK=[[3715.,3735.]];

mux=0.01;                  // smoothing parameter for the stellar age distribution
muz=10;                  // smoothing parameter for the age-Z relation
muv=100;                  // smoothing parameter for the LOSVD

L1="D2";                // smoothing kernel for stellar age distribution (here D2, i.e. square Laplacian)
L3="D1";                // smoothing kernel for age-metallicity relation (here D1, i.e. square gradient)

file = "/home/amit/Yorick/data/d158/d158_1D_2p7_flux.fits";
efile = "/home/amit/Yorick/data/d158/d158_1D_2p7_fsigma.fits";
a=convert_all(file,log=1,z0=0.0200794,errorfile=efile);    // Converting to steckmap format
b=bRbasis3(ages,4.4,wavel=wavel,basisfile=basisfile,nbins=nbins); // generate basis
ws;plb,b.flux,b.wave;   // plots the basis
x=sfit(a,b,kin=kin,epar=epar,noskip=0,vlim=vlim,meval=meval,nde=30,mux=mux,muv=muv,muz=muz,L3=L3,L1=L1,sav=1);

