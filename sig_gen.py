import numpy as np
import sys, time, array
import socket
import pythia8
import pickle
from ROOT import TLorentzVector, TGenPhaseSpace

ISDEB = 0
system_name = socket.gethostname()
if len(sys.argv)>1: jobid = int(sys.argv[1])
else: jobid = 0;ISDEB = 1


particles = []

py = pythia8.Pythia("", False)
py.readString("HardQCD:hardbbbar = on")
py.readString("Beams:eCM = 14000")
py.readString("Print:quiet = on")
py.readString("Random:setSeed = on")
py.readString("Random:seed = "+str(jobid))
py.init()

def toLV(track):
    lv = TLorentzVector()
    lv.SetPxPyPzE(track.px()*1000,track.py()*1000,track.pz()*1000,track.e()*1000)## in MeV!                                                                                                                 
    return lv

def get_DOCA(par1,par2):
   ## pass [TLorentzVector+pos] or HEPmc format                                                                                                                                                             
   if type(par1)==list:
      p1 = np.array([par1[0].Px(),par1[0].Py(),par1[0].Pz()])
      p2 = np.array([par2[0].Px(),par2[0].Py(),par2[0].Pz()])
      pos1 = np.array(par1[1])
      pos2 = np.array(par2[1])
   else:
      p1 = np.array([par1.px(),par1.py(),par1.pz()])
      p2 = np.array([par2.px(),par2.py(),par2.pz()])
      pos1 = np.array([par1.xProd(),par1.yProd(),par1.zProd()])
      pos2 = np.array([par2.xProd(),par2.yProd(),par2.zProd()])

   P1P2 = pos1-pos2
   #print(P1P2)                                                                                                                                                                                             
   normp1 = p1/np.sqrt(np.dot(p1,p1))
   normp2 = p2/np.sqrt(np.dot(p2,p2))
   #print(normp1)                                                                                                                                                                                           
   #print(normp2)                                                                                                                                                                                           
   n = np.cross(normp1,normp2)
   return abs(np.dot(P1P2,n))/np.sqrt(np.dot(n,n))

def get_IP(par,svpos,pvpos = 0):
      mom = np.array([par.Px(),par.Py(),par.Pz()])
      nmom = mom/np.sqrt(np.dot(mom,mom))
      if type(pvpos)==int: pos = np.array(svpos)
      else: pos = np.array(svpos)-np.array(pvpos)
      #dist(line,point) = |v1x(CA)|                                                                                                                                                                            
      cprod = np.cross(nmom,pos) ## PV at 0,0,0                                                                                                                                                                
      return np.sqrt(np.dot(cprod,cprod))

   ## are all the daughters reconstrutible? Require them to be in accceptance, with pT>250 GeV/c                                                                                                               
def daughters_reconst(daughs):
       for da in daughs:
           if not (da.PseudoRapidity()>2 and da.PseudoRapidity()<5): return False
           if da.Pt()<500: return False
       return True

   ## myd is already a tlorentz vector                                                                                                                                                                         
   ## decay D+ to kaon, pion and pion                                                                                                                                                                          
   #mka,mpi = 493.7,139.6 #masas do muon                                                                                                                                                                       
   #o IP teria que dar 0                                                                                                                                                                                       
mmuon = 105
def decay_Ds(myd):
    mygen = TGenPhaseSpace()
    mygen.SetDecay(myd,2,array.array("d",[mmuon,mmuon]))
    mygen.Generate()
    return [mygen.GetDecay(i) for i in range(2)]

def get_pos(track):
    return [track.xDec(),track.yDec(),track.zDec()]

def descendants(ev,par,myl,i0=-1):
    par0 = par
    if type(par)==list: par0 = par[1]
    for da in par0.daughterList():
        if i0>=0:
            myl.append([i0,ev[da].index(),ev[da].name()])
            descendants(ev,ev[da],myl,i0+1)
        else:
            myl.append(da)
            descendants(ev,ev[da],myl)
    return


def filter_Ds(event):
    myds = [[toLV(x),x.id(),get_pos(x),x] for x in event if abs(x.id())==DPLUSID]
    #print(myds)
    
    if len(myds)<1: return []
    for d in myds: d.insert(1,decay_Ds(d[0])) ## insert decay prods in second place
    
    myds = [x for x in myds if daughters_reconst(x[1])]
    
    if len(myds)<1: return []                                                                                          
    #print(myds)                  
    myds = [[x[0].Pt(),x] for x in myds];myds.sort(reverse=True)
    #print(myds)
    #l = [x[1] for x in myds[:2]]
    #print(l)
    return [x[1] for x in myds[:2]]

def get_DOF(pos):
    return np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)


def get_particles_filtered(mytracks):
	new_list = []
	for x in mytracks:	
		if((x[0].PseudoRapidity() > 2 and x[0].PseudoRapidity() < 5) and x[0].Pt()  >250 ):
			if ( get_IP(x[0],x[2]) > 0.1):
				new_list.append(x)
	return new_list
	
def get_isolation(m, tracks):
	isos = [get_DOCA(m,itrack) for itrack in tracks]
	#print(isos)
	if isos != []:
		iso = min(isos)
	else: iso = 1e6
	return iso

ids_mkp = [211, 321, 11, 13 , 2212]
def get_particles(event,ds):
  myds = [[toLV(x),x.id(),get_pos(x),x] for x in event if ((x.index() not in ds) and (abs(x.id())  in ids_mkp) )]
  return myds

df = []
if ISDEB: nevts = 10000
else: nevts = int(2e5)
DPLUSID = 531


time0 = time.time()
dfs = []
muones = []

for j in xrange(nevts):
	if j and (not j%10000):
		print("*********",j,"events gone *********")
		print("*********","took",(time.time()-time0),"s *********")
		time0 = time.time()
	if not py.next(): continue
	ds = filter_Ds(py.event)
	if len(ds): b1 = ds
	else: continue

	bs1,ba1,id1,pos1, part = b1[0]
	m1,m2 = ba1                                                                                                                                                                                        
	m = {'m1':[],'m2':[]}
	ip2 = get_IP(m2,pos1)
	ip1 = get_IP(m1,pos1)
	#print(ip1)
	ds = []
	if ( ip1 > 0.5 and  ip2 > 0.5):
		descendants(py.event,part,ds)
		particles = get_particles(py.event,ds)
		particles = get_particles_filtered(particles)
		iso1 = get_isolation([m1,pos1],particles)
		iso2 = get_isolation([m2,pos1],particles)
		m['m1'] = [m1, pos1, ip1, get_DOF(pos1), bs1, iso1]
		m['m2'] = [m2, pos1, ip2, get_DOF(pos1), bs1, iso2] 
		muones.append(m)
	
 



pickle.dump(muones, open('muones_nodos_53.pickle','wb'))
muones_con_iso = []
signal = []

for i in muones:

    if (i['m1'][0].Pt() > 500 and i['m2'][0].Pt() > 500):
        #print(i)
        if(i['m1'][3] > 0.5 and i['m2'][3] > 0.5):
            #print('ip bien')
			doca= get_DOCA(i['m1'],i['m2'])
			if (doca < 0.1):
				ip_nai = get_IP(i['m1'][4],i['m1'][1])
                #print(ip_nai)
				dof_m = get_DOF(i['m1'][1])
				pt_m = i['m1'][4].Pt()
				if (pt_m> 1000 and ip_nai < 0.1 and dof_m  > 3):
					aux ={'m1':{'pt': i['m1'][0].Pt(),'ip':i['m1'][2],'iso':i['m1'][5]}, 'm2':{'pt': i['m2'][0].Pt(),'ip':i['m2'][2],'iso':i['m2'][5]}, 'pt_m': i['m1'][4].Pt(), 'pt':pt_m,'doca':doca,'ip_m':ip_nai,'dof_m':dof_m}
					signal.append(aux)
                    #print(i)

#print(signal)
pickle.dump(signal,open('signal_'+system_name+'.pickle','wb'))
    #isos = [DOCA(mu,itrack) for itrack in mytracks2]                                                                                                                                                       
    #iso(mu) = min(isos)                                                                                                                                                                                    


