#pi0 photoproduction analysis gamma+proton --> pi0 + proton

import numpy as np
import LT.box as B
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

#getting the data
F = B.get_file('pion-neutral.data.py')
Eg_MeV = B.get_data(F, 'Egamma')
cos = B.get_data(F, 'cos(theta)')
dsig_domega = B.get_data(F, 'dsigma/dOmega')
d_dsigdomega = B.get_data(F, 'Data errors')

#converting energies from MeV to GeV
Eg = Eg_MeV/1000

#masses in GeV
m2 = 0.938272081 #proton mass which is a byron
m3 = 0.135 #pi0 mass which is a meson
m4 = 0.938272081 #proton mass again to make the calculations of the 4-vector easier

#getting s and w using 4-vector
s = (m2)**2+(2*Eg*m2) #ask if this is correct
w = np.sqrt(s)

#converting dsig/domega to dsig/dcos
dsigdcos = 2*np.pi*dsig_domega
sig = d_dsigdomega*2*np.pi

#energies
Eg_cm = (s-(m2)**2)/(2*w) #energy of gamma
Eb_cm = (s+m4**2-m3**2)/(2*w) #energy of the proton
Em_cm = (s-m4**2+m3**2)/(2*w) #energy of the pi0

#momentum
Pg_cm = Eg_cm #momentum of gamma k=E1
Pb_cm = np.sqrt((Eb_cm)**2 - (m4)**2)  #momentum of byron (proton) at cm
Pm_cm = np.sqrt((Em_cm)**2 - (m3)**2) #momentum of meson (proton) at cm

t = (2.*Pg_cm*Pm_cm*cos+m3**2-2.*Eg_cm*Em_cm)
dsigdt = dsigdcos/ (2.*Pg_cm*Pm_cm)
sig_dt = sig/(2.*Pg_cm*Pm_cm)

#calculating the transverse momentum in several parts
pt2_1 = (s-m2**2)**2/(4*s)
pt2_2 = (((s+m3**2-m4**2)**2)/(4*s)) - m3**2
pt2_3 = ((1/(4*s)*(s-m2**2)*(s+m3**2-m4**2))+(t-m3**2)/2)**2
pt2_4 = ((s-m2**2)**2)/(4*s)
pt2 = (((pt2_1)*(pt2_2)-(pt2_3))/pt2_4)

#keep data points of the cosine bins closest to zero
mx = 0.15
mn = -0.15

cospt1 = cos[(cos <= mx)&(cos >= mn)] #cosine of angle 85 to 90 but in rad
s1 = s[(cos <= mx)&(cos >= mn)]
dsigdt1 = dsigdt[(cos <= mx)&(cos >= mn)]
sig1 = sig_dt[(cos <= mx)&(cos >= mn)] #is it necessary?
t1 = t[(cos <= mx)&(cos >= mn)]
pt21 = pt2[(cos <= mx)&(cos >= mn)]

x=np.array([cospt1, s1])

#alpha is the array to making the cuts
# first one goes from zero to 80% of the max value in steps of the
# max value divided by 100
alpha = np.arange(0.00*max(pt21), 0.8*max(pt21), max(pt21)/100) #max(pt21) is the max value of the transverse momentum
# second starts where the first one ended, and takes bigger steps to the max value
second = np.arange(0.8*max(pt21), 0.99*max(pt21), (max(pt21)-0.8*max(pt21))/6)
# join the two together
alpha = np.append(alpha, second)

#Put the plots in a PDF
PDF = PdfPages("pi0.pdf")

#Cosine bins in the plots
cosines = np.array([-0.15, -0.05, 0.05, 0.15])


for j in alpha[:-5]:
    
    #The fit for each cosine bin
    def fit(x, A, C, N):
            return (A + C*i)*x**(-N)

    
    #Exclude data points with p^2 lower than the minimum
    cosp = cospt1[pt21 >= j]
    dsigdtp = dsigdt1[pt21 >= j]
    sigp = sig1[pt21 >= j]
    sp = s1[pt21 >= j]
    ptp = pt21[pt21 >= j]

    plt.figure(figsize = (18,9))
    
    #Loop over all cosines
    for i in cosines:
        
        
        #This is a fit over all cosine bins at the same time
        def expanded_fit(x, A, C, N):
            return (A + C*x[0])*x[1]**(-N)
        
        popt, pcov = curve_fit(expanded_fit, (cosp, sp), dsigdtp, sigma=sigp, maxfev= 5000)
        # value of the fit/line
        y_pred = expanded_fit((cosp, sp), popt[0],popt[1], popt[2])
        # Chi-squared computation
        chi_squared = np.sum(((dsigdtp-y_pred)/sigp)**2)
        # Reduced chi-squared computation
        redchi = (chi_squared)/(len(cosp)-len(popt))
        # get N
        N = np.abs(popt[2])
        # get dN
        dN = pcov[2,2]**0.5
        
        if (i == -0.15):
            
            #Filter out data points that are not within this cosine bin
            coss= cosp[cosp == i]
            dsigdts = dsigdtp[cosp == i]
            sigs = sigp[cosp == i]
            ss = sp[cosp == i]
            pts = ptp[cosp == i]
            
            #plot dsigma/dt vs s
            plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'g', markersize = 10, label = r'$\cos \theta = -0.10$')
            #Fit
            popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
            #plot the fit
            plt.semilogy(ss, fit((ss), *popt), color = 'g', linestyle = '--', linewidth = 3)
       
        elif (i == -0.05):
            
            "Filter out data points that are not within this cosine bin"
            coss= cosp[(cosp == i)]
            # Multiply for better visualization
            dsigdts = 2*dsigdtp[cosp == i]
            sigs = 2*sigp[cosp == i]
            ss = sp[cosp == i]
            pts = ptp[cosp == i]
            
            "plot dsigma/dt vs s"
            plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'b', markersize = 10, label = r'$\cos \theta = 0.0$')
            "Fit"
            popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
            "plot the fit"
            plt.semilogy(ss, fit((ss), *popt), color = 'b', linestyle = '--')
            
        elif (i == 0.05):
            
            "Filter out data points that are not within this cosine bin"
            coss= cosp[cosp == i]
            # Multiply for better visualization
            dsigdts = 4*dsigdtp[cosp == i]
            sigs = 4*sigp[cosp == i]
            ss = sp[cosp == i]
            pts = ptp[cosp == i]
            
            "plot dsigma/dt vs s"
            plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'r', markersize = 10, label = r'$\cos \theta = 0.10$')
            "Fit"
            popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
            "plot the fit"
            plt.semilogy(ss, fit((ss), *popt), color = 'r', linestyle = '--', linewidth = 3)
        
       
        elif (i == 0.15):
            
            "Filter out data points that are not within this cosine bin"
            coss= cosp[cosp == i]
            # Multiply for better visualization
            dsigdts = 8*dsigdtp[cosp == i]
            sigs = 8*sigp[cosp == i]
            ss = sp[cosp == i]
            pts = ptp[cosp == i]
            
            "plot dsigma/dt vs s"
            plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'r', markersize = 10, label = r'$\cos \theta = 0.10$')
            "Fit"
            popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
            "plot the fit"
            plt.semilogy(ss, fit((ss), *popt), color = 'r', linestyle = '--', linewidth = 3)
        
            
        "Stuff being put in the plot"
        # legend with results of the fit of all cosine bins
        plt.legend(title = r'$N$ = %2.2f $\pm$  %2.2f ' %(N, dN), title_fontsize = 20)
        # title, includes cut [GeV^2] and reduced chi-squared    
        plt.title(r'$\gamma  p \rightarrow \pi^0 p$: $\frac{d\sigma}{dt}$=$(A + B \cos \Theta)s^{N \pm \delta N}$, $p_{\perp _{min}} ^2 = %2.2f}$, $\chi ^2 /df = %2.2f$' %(j,redchi)  , size = 25)
        # axis labels
        plt.ylabel(r'$\frac{d\sigma}{dt}$ $[\mu$bGeV$^{-2}]$', size =35)
        plt.xlabel('s [$GeV^2$]', size = 25)
        # semi-log scale
        plt.yscale('log')
        # size of the numbers in the axis
        plt.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
    
    #save each to the PDF
    PDF.savefig()

#close the PDF when the loop is done       
PDF.close()









