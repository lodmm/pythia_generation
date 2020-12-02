#Bkg generator
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

def get_pos(track):
    return [track.xDec(),track.yDec(),track.zDec()]

def descendants(ev,par,myl,i0=-1):
    par0 = par
    if type(par)==list: par0 = par[1]
    for da in par0.daughterList():
        if i0>=0:
            myl.append([i0,ev[da],ev[da].index(),ev[da].id(),ev[da].name()])
            descendants(ev,ev[da],myl,i0+1)
        else:
            myl.append(da)
            descendants(ev,ev[da],myl)
    return


def filter_Ds(event):
	myds = [[toLV(x),x.id(),get_pos(x),x] for x in event if (abs(x.id())==DPLUSID)]
	#print(myds)
    
	if len(myds)<2: return []
		
	
	myds = [[x[0].Pt(),x] for x in myds];myds.sort(reverse=True)
	
	return [x[1] for x in myds[:2] ]
    #print(l)

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
	if isos != []:
		iso = min(isos)
	else: iso = 1e6	
	return iso

ids_mkp =  [211, 321, 11, 13 , 2212]
def get_particles(event,ds):
  myds = [[toLV(x),x.id(),get_pos(x),x] for x in event if ((x.index() not in ds) and (abs(x.id())  in ids_mkp) )]
  return myds

def get_vertex2(par1,par2):
   pos1 = np.array([par1.xProd(),par1.yProd(),par1.zProd()])
   pos2 = np.array([par2.xProd(),par2.yProd(),par2.zProd()])

   p1 = np.array([par1.px(),par1.py(),par1.pz()])
   p2 = np.array([par2.px(),par2.py(),par2.pz()])
   
   normp1 = p1/np.sqrt(np.dot(p1,p1))
   normp2 = p2/np.sqrt(np.dot(p2,p2))
   
   pxs,pyx,pzs = map(lambda i: [pos1[i],pos2[i]],range(3))
   txs,tys,tzs = map(lambda i: [normp1[i],normp2[i]],range(3))
   return get_vertex(pxs,pyx,pzs,txs,tys,tzs)

#calcular as pendentes
def get_vertex(pxs,pys,pzs,txs,tys,tzs):
   allpoints = [pxs,pys,pzs]
   allslopes = [txs,tys,tzs]
   m = len(pxs)
   ## system of equations of the type
   ## term0 = coef00 * x + coef01 * y + coef02 * z
   ## term1 = coef10 * x + coef11 * y + coef12 * z
   ## term2 = coef20 * x + coef21 * y + coef22 * z
   ## value of coefficients (from equation in the web above)
   coef = [np.zeros(3) for x in range(3)]
   for i in range(3):
      for j in range(3):
         if i==j: coef[i][j] = m -reduce(lambda a,b: a+b,map(lambda c: c**2,allslopes[j]))
         else: coef[i][j] = -reduce(lambda a,b: a+b,map(lambda c,d: c*d,allslopes[i],allslopes[j]))
   ## value of terms (from equation in the web above)
   terms = np.zeros(3)
   for i in range(3):
      if DEBUG: print "********"
      fterm = reduce(lambda a,b: a+b,allpoints[i])
      if DEBUG: print fterm
      sterm = 0
      if DEBUG: print "----"
      for j in range(3):
         if DEBUG: print map(lambda c,d,e: c*d*e,allpoints[j],allslopes[j],allslopes[i])
         if DEBUG: print reduce(lambda a,b: a+b,map(lambda c,d,e: c*d*e,allpoints[j],allslopes[j],allslopes[i]))
         sterm += reduce(lambda a,b: a+b,map(lambda c,d,e: c*d*e,allpoints[j],allslopes[j],allslopes[i]))
      if DEBUG: print "----"
      if DEBUG: print sterm
      terms[i] = fterm-sterm
   if DEBUG: print coef, terms
   return np.linalg.solve(coef, terms)

def get_muons(myl):
	m_list = []
	for i in myl:
		#print(i.name())
		if abs(i.id()) == 13:

			m_list.append(i)
	return m_list
	
#comentar os codigos
def get_mother(m1, m2):
	mo1 = toLV(m1[0])
	mo2 = toLV(m2[0])
	mother = mo1 + mo2
	# get the possition
	pos = get_vertex2(m1[0],m2[0])
	return [mother,pos]


df = []
if ISDEB: nevts = 10000
else: nevts = int(3e4)
DPLUSID = 511
#valen todas as Bs

time0 = time.time()
dfs = []
muones = []

## is the particle a b hadron?
def isb(par):
   pdg = par.particleDataEntry()
   return (pdg.isHadron() and abs(pdg.heaviestQuark(1))==5)

## has the particle flown?
def isprompt(par):
   xdec,xprod = par.xDec(),par.xProd()
   ydec,yprod = par.yDec(),par.yProd()
   zdec,zprod = par.zDec(),par.zProd()
   return (xdec==xprod and ydec==yprod and zdec==zprod)


def get_B_muon(ev):
	bs0s = filter(lambda x: isb(x) and (not isprompt(x)),ev)
	muons = []
	if len(bs0s) < 2: return []
	bdescd = {}
	for b in bs0s:
		bdesc = [];descendants(ev,b,bdesc);bdesc = map(lambda y: ev[y],bdesc)
		muon = get_muons(bdesc)
		
		if muon != [] : muons.append(muon)
		else: continue
		bdescd[b] = muon
	if len(muons) <2: return []
		
	return bs0s, bdescd, muons

for j in xrange(nevts):
	if j and (not j%10000):
		print("*********",j,"events gone *********")
		print("*********","took",(time.time()-time0),"s *********")
		time0 = time.time()
	if not py.next(): continue
	#ds = filter_Ds(py.event)
	res = get_B_muon(py.event)
	if len(res): bs,bdesc,muons = res
	else: continue
	
	#print(b1)
	# bs1,id1,pos1, part = b1[0]
	# bs2,id2,pos2, part2 = b1[1]
	
	ms = []
	for i in bdesc.keys():
		m0 = bdesc[i]
		for m in m0:
			ms.append([m,i])
	final = []	
	
	if len(ms) > 2:		
		comb = []
		for i in ms:
			
			for m in ms:
				
				if i[0].index() == m[0].index(): continue
				c = [i[0].index(),m[0].index()]
				c.sort()
				if c in comb: continue
				comb.append(c)
				m1 = [toLV(i[0]),get_pos(i[0]),i[0],i[1]]
				m2 = [toLV(m[0]),get_pos(m[0]),m[0],m[1]]
				doca = get_DOCA(m1,m2)
				final.append([doca,c,m1,m2])
		#get muons
		
		final.sort()
		sol = final[0]
		m1 = sol[2]
		m2 = sol[3]
	else:
		m1 = [toLV(ms[0][0]),get_pos(ms[0][0]),ms[0][0],ms[0][1]]
		m2 = [toLV(ms[1][0]),get_pos(ms[1][0]),ms[1][0],ms[1][1]]

	
	
	
	#print(myl)
	
	d = [m1[0], m2[0]]
	if not(daughters_reconst(d)): continue
	m = {'m1':[],'m2':[]}
	ip2 = get_IP(m2[0],m2[1])
	ip1 = get_IP(m1[0],m1[1])
	ds = []
	if ( ip1 > 0.5 and  ip2 > 0.5):
		
		descendants(py.event,m1[3],ds)
		particles = get_particles(py.event,ds)
		particles = get_particles_filtered(particles)
		iso1 = get_isolation([m1[0],m1[1]],particles)
		ds = []
		descendants(py.event,m2[3],ds)
		particles = get_particles(py.event,ds)
		particles = get_particles_filtered(particles)
		iso2 = get_isolation([m2[0],m2[1]],particles)
		m['m1'] = [m1[0], m1[1], ip1, get_DOF(m1[1]), iso1,m1[2]]
		m['m2'] = [m2[0], m2[1], ip2, get_DOF(m2[1]), iso2, m2[2]] 
		muones.append(m)
	
 


bkg = []
print(muones)
#get the muons with less doca
for i in muones:

    if (i['m1'][0].Pt() > 500 and i['m2'][0].Pt() > 500):
        #print(i)
        if(i['m1'][3] > 0.5 and i['m2'][3] > 0.5):
			doca = get_DOCA(i['m1'],i['m2'])
			
			if (doca < 1):
				#Create mother
				DEBUG = True
				print(i)
				mother = get_mother([i['m1'][5],i['m1'][1]],[i['m2'][5],i['m2'][1]])
				ip_nai = get_IP(mother[0],mother[1])
				dof_m = get_DOF(mother[1])
				pt_m = mother[0].Pt()
				print(mother)
				if (pt_m> 1000 and ip_nai < 0.1 and dof_m > 3):
					aux ={'m1':{'pt': i['m1'][0].Pt(),'ip':i['m1'][2],'iso':i['m1'][4]}, 'm2':{'pt': i['m2'][0].Pt(),'ip':i['m2'][2],'iso':i['m2'][4]}, 'pt_m': i['m1'][4].Pt(), 'pt':pt_m,'doca':doca,'ip_m':ip_nai,'dof_m':dof_m}
					bkg.append(aux)
					#bkg.append(i)
print(bkg)          
pickle.dump(bkg,open('bkg_'+system_name+'.pickle','wb'))
                                                                                                                                                                                 


