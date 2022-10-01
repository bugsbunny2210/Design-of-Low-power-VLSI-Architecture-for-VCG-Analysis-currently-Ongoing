from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np 
import wfdb as wf

#for displaying graph
def graph(signal,title):
    x = np.arange(0, 5000)
    y= signal
    plt.figure()
    plt.plot(x, y) 
    plt.title(title) 
    plt.xlabel("x axis") 
    plt.ylabel("y axis")
    return 0
    
#for displaying graph
def sep_graph(orig,idw,kor,title):
                        x = np.arange(0, 5000)
                        plt.figure()
                        plt.plot(x, orig, color='r', label='vcg')
                        plt.plot(x, idw, color='g', label='idw') 
                        plt.plot(x, kor, color='y', label='kors') 
                        plt.title(title) 
                        plt.xlabel("x axis") 
                        plt.ylabel("y axis")
                        plt.legend()
                        return 0
    
    
#function for inverse dowers transform
def mat(ecg):
      v = np.array([[-0.172,-0.074,0.122,0.231,0.239,0.194,0.156,-0.010],
                   [0.057,-0.019,-0.106,-0.022,0.041,0.048,-0.227,0.887],
                   [-0.229,-0.310,-0.246,0.063,0.055,0.108,0.022,0.102]])
      p=np.dot(v,ecg)  
      return p
  


#function for kors transform
def kor(ecg):
    a = np.array([[-0.130,0.050,-0.010,0.140,0.060,0.540,0.380,-0.070],
                  [0.060,-0.020,-0.050,0.060,-0.170,0.130,-0.070,0.930],
                  [-0.430,-0.060,-0.140,-0.200,-0.110,0.310,0.110,-0.230]])
    c=np.matmul(a,ecg)  
    return c

#function for calculation of mean square error
def mean_error(orig,calc):
    error_arr=np.zeros(orig.shape)
    
    for i in range(0, 3):
      for j in range(0, 5000):
          error_arr[i][j] = np.square(calc[i][j]-orig[i][j])
          error_arr[i][j]= error_arr[i][j]/5000
          j=j+1
         
    i=i+1
    
    return error_arr

def kor_vtoe(vcg):
    a = np.array([[-0.130,0.050,-0.010,0.140,0.060,0.540,0.380,-0.070],
                  [0.060,-0.020,-0.050,0.060,-0.170,0.130,-0.070,0.930],
                  [-0.430,-0.060,-0.140,-0.200,-0.110,0.310,0.110,-0.230]])
    
    
   
                
    c=np.dot(a.T,vcg)  
    return c

def mat_vtoe(vcg):
      v = np.array([[-0.172,-0.074,0.122,0.231,0.239,0.194,0.156,-0.010],
                   [0.057,-0.019,-0.106,-0.022,0.041,0.048,-0.227,0.887],
                   [-0.229,-0.310,-0.246,0.063,0.055,0.108,0.022,0.102]])
      
      x =np.dot(v.T,v)
      y =np.linalg.inv(x)
      z =np.dot(y,v.T)
      a =np.dot(z.T,z)
      b =np.linalg.inv(a)
      c =np.dot(b,z.T)
      p =np.dot(c.T,vcg) 
      p =p*90
      return p

    
  
    

# read ecg sample
record = wf.rdrecord('sample\s0010_re')
wf.plot_wfdb(record=record, title='Record a103l from PhysioNet Challenge 2015') 
# display(record.__dict__)
ecg_signals, fields = wf.rdsamp('sample\s0010_re', channels=[0,1,2,3,4,5,6,7,8,9,10,11],sampfrom=0, sampto=5000)
print('Original ECG signals')
# display(ecg_signals)
# display(fields)
graph(ecg_signals,"Original ECG Signal")
display(fields)


ecg_probes, fields = wf.rdsamp('sample\s0010_re', channels=[6,7,8,9,10,11,0,1],sampfrom=0, sampto=5000) 
graph(ecg_probes,"selected ECG Signal")
display(ecg_signals)
display(fields)

#read vcg sample
vcg_signals, fields = wf.rdsamp('sample\s0010_re', channels=[12,13,14],sampfrom=0, sampto=5000)
print('original VCG signal')
# display(vcg_signals)
# display(fields)
graph(vcg_signals,"Original VCG signal")


# ECG to VCG by using Inverse Dowers Transform
print('calculated VCG signal')
vcg_idt=mat(ecg_probes.T)
# display(vcg_idt)
# display(fields)
graph(vcg_idt.T,"Calculated VCG signal using Inverse Dowers Transform")


#ECG to VCG by using kors regression Transform
vcg_kors =kor(ecg_probes.T)
# display(vcg_kors)
# display(fields)
graph(vcg_kors.T,"Calculated VCG signal using kors regression Transform")
plt.show()


#VCG to ECG by using kors regression Transform
ecg_kors =kor_vtoe(vcg_signals.T)
# display(ecg_kors)
# display(fields)
graph(ecg_kors.T,"Calculated ECG signal using kors regression Transform")
plt.show()    

# VCG to ECG by using Dowers Transform
ecg_dt =mat_vtoe(vcg_signals.T)
# display(ecg_dt)
# display(fields)
graph(ecg_dt.T,"Calculated ECG signal using dowers Transform")

# mean error calculation after kors transform
error_kor = mean_error(vcg_signals.T,vcg_kors)
graph(error_kor.T,"error signal kors")

error_idt = mean_error(vcg_signals.T,vcg_idt)
graph(error_idt.T,"error signal inverse dowers transform")

title_v= np.array(["x curve","y curve","z curve"])
title_e= np.array(["V1 curve","V2 curve","V3 curve","V4 curve","V5 curve","V6 curve","I curve","II curve"])

#displaying individual signals
for i in range(0, 3):
    graph(vcg_signals.T[i],title_v[i])
    graph(vcg_idt[i],title_v[i])
    graph(vcg_kors[i],title_v[i])
    i=i+1
    
for i in range(0, 8):
        graph(ecg_probes.T[i],title_e[i])
        graph(ecg_dt[i],title_e[i])
        graph(ecg_kors[i],title_e[i])
        i=i+1

#comparison of different transform from original signal
for i in range(0, 3):
    sep_graph(vcg_signals.T[i],vcg_idt[i],vcg_kors[i],title_v[i])
    i=i+1

for i in range(0,8):
    sep_graph(ecg_probes.T[i],ecg_dt[i],ecg_kors[i],title_e[i])
    i=i+1    
plt.show()  
