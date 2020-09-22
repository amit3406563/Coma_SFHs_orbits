import numpy as np
import sys

def getpct(x,q,axis=0):
   x=np.asarray(x)
   if len(x.shape)> 1: x=np.sort(x,axis)
   else: x=np.sort(x)
   j = min([len(x)-1,int(0.01*q*len(x))])
   if len(x) > 2:
      if axis == 0: return x[j]
      else: return x[::,j]
   elif len(x): return np.sum(x)/(1.0*len(x))
   else: return 0.0

def readsection(f,startLine,numEntires=30,nPerLine=5):
    # assumes that f is a list constructed as f=file.readlines() on some file
    nLines=numEntries/nPerLine
    if (1.*numEntries/nPerLine)!=nLines: nLines=nLines+1
    l=[]
    for j in range(startLine,startLine+nLines):
        s=(f[j].strip()).split()
        [l.append(float(s[i])) for i in range(len(s))]
    return np.array(l)

for filename in [sys.argv[-1]+i for i in ['-AMR.txt','-MASS.txt','-SAD.txt','-SFR.txt']]:
    f=open(filename).readlines()
    g=open(filename.replace('txt','dat'),'w')
    numEntries=int((f[0].split())[-1])
    nPerLine=len(f[1].split())
    nLines=numEntries/nPerLine
    ages=readsection(f,1,numEntries,nPerLine)
    q=readsection(f,2+nLines,numEntries,nPerLine)
    if len(f)>2*nLines+2:
        MC=[]
        for i in range(int((f[2*nLines+2].strip()).split()[-1])):
            MC.append(readsection(f,(2+i)*(nLines+1)+1,numEntries,nPerLine))
        MC=np.array(MC)
        mid=getpct(MC,50)
        low1sigma=getpct(MC,16)
        high1sigma=getpct(MC,84)
        for j in range(len(ages)):
            ostr='%12.6g %12.6g %12.6g %12.6g %12.6g\n' % (ages[j],q[j],low1sigma[j],high1sigma[j],mid[j])
            g.write(ostr)
    else:
        for j in range(len(ages)):
            ostr='%12.6g %12.6g\n' % (ages[j],q[j])
            g.write(ostr)
    g.close()

filename=sys.argv[-1]+'-spectra.txt'
f=open(filename).readlines()
g=open(filename.replace('txt','dat'),'w')
numEntries=int((f[0].split())[-1])
nPerLine=len(f[1].split())
nLines=numEntries/nPerLine
if (1.*numEntries/nPerLine)!=nLines: nLines=nLines+1
wavelengths=readsection(f,1,numEntries,nPerLine)
data=readsection(f,2+nLines,numEntries,nPerLine)
fit=readsection(f,2*(nLines+1)+1,numEntries,nPerLine)
weights=readsection(f,3*(nLines+1)+1,numEntries,nPerLine)
extinct=readsection(f,4*(nLines+1)+1,numEntries,nPerLine)
for j in range(len(wavelengths)):
   ostr='%7.2f %8.6f %8.6f %8.6f %8.6f\n' % (wavelengths[j],data[j],fit[j],weights[j],extinct[j])
   g.write(ostr)
g.close()
