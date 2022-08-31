# This solves the differential equations via ENSEMBLE OVER CURRENTS
# Kelvin Titimbo, David C. Garrett, S. Suleyman Kahraman, and Lihong V. Wang
# California Institute of Technology
# August 2022

using Plots; gr()
using MAT
using Interpolations
using HypothesisTests
using LinearAlgebra
using Statistics
using DifferentialEquations, ODEInterfaceDiffEq
using Alert
using DelimitedFiles
using LaTeXStrings
using NaNStatistics
using Dates
LinearAlgebra.BLAS.set_num_threads(8)
BLAS.get_num_threads();
Threads.nthreads();
cd(dirname(@__FILE__)) # Set the working directory to the current location
hostname = gethostname();
t_start = Dates.now()

########## USER SETTINGS ##########
field = "quadrupole" # options: quadrupole OR exact
distri = "hs" # options: hs (heart-shape) OR iso (isotropic) OR avg (5*pi/8)
N_sampling = 10000  ; # total number of thn0,phn0 pairs
const tiny_angle = 1e-6;    # less than this gives errors
filename = "FS_spinflip_"*string(N_sampling)*"_"*string(Dates.format(Dates.now(), "yyyymmddTHHMMSS")) # filename
###################################

## SOLVER PARAMETERS [https://diffeq.sciml.ai/stable/basics/common_solver_opts/]
time_save_step=0.2e-7; # interval for saving each flight (unrelated to solver step size)
reltol=1e-10 ; # relative tolerance
abstol=1e-10 ; # absolute tolerance
maxiters=1000000000000 ;
dtmin=1e-40 ; # minimum time step (old 1e-23 )
force_dtmin = true ;
alg = radau() ; # algorithm -- radau()

## PHYSICAL CONSTANTS from NIST
# RSU : Relative Standard Uncertainty
hbar = 6.62607015e-34/2/pi ;# Reduced Planck constant (J s)
mp = 1.67262192369e-27 ;# proton mass (kg)
me = 9.1093837015e-31 ;# electron mass (kg)
qe = 1.602176634e-19 ;# elementary charge (C)
mu0 = 4*pi*1e-7 ; # Vacuum permeability (Tm/A)
muB = 9.2740100783e-24 ; # Bohr magneton (J/T). RSU = 3.0e-10
muN = 5.0507837461e-27 ; # Nuclear magneton (J/T). RSU = 3.1e-10
gammae = -1.76085963023e11 ; # Electron gyromagnetic ratio  (1/sT). RSU = 3.0e-10
gamma_neutron = 1.83247171e8 ;  # Neutron gyromagnetic ratio (1/sT). RSU = 2.4e-7
mue = 9.2847677043e-24 ; # Electron magnetic moment (J/T). RSU = 3.0e-10

## ATOM INFORMATION: Potassium-39
const R = 275e-12 ;     # van der Waals atomic radius (m) [https://periodic.lanl.gov/3.shtml]
const mun = 0.391470*muN ;  # Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]
const gamman = 1.2500612e7 ;    # Gyromagnetic ratio [gfactor*muB/hbar] (1/(Ts)) [http://triton.iqfr.csic.es/guide/eNMR/chem/]
const spin_I = 3/2 ;  # Nuclear spin
const gfactor = 0.00014193489 ; # g-factor [E. Arimondo, M. Inguscio, and P. Violino. Experimental determinations of the hyperfine structure in the alkali atoms. Rev. Mod. Phys., 49(1):31, January 1977.]

## ANGULAR DISTRIBUTIONS
# Heart-shape distribution
function thn_heart(size)  # Random generator
    2*asin.(rand(size).^(1/4))
end

function avg_heart(size) # Generates an array of thn0=5*pi/8
    5*pi/8 * ones(size)
end

# Isotropic random distributions
function thn_iso(size)
    2*asin.(sqrt.(rand(size)))
end

function phn_iso(size)
    2*pi*rand(size)
end

function theta_safe(x)
    theta_m = mod(x,float(pi))
    if theta_m < tiny_angle
        ts = tiny_angle
    elseif theta_m > pi - tiny_angle
        ts = - tiny_angle
    else
        ts = x
    end
    return ts
end

function theta_wrap(x)
    pi.-abs.(mod.(x,2*pi).-pi)
end

## EXPERIMENTAL PARAMETERS
v = 800; # atom speed (m/s)
za = 1.05e-4; # wire position (m)
Br = 0.42e-4; # remnant field (T)
FS_Iwire = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]; # in (A)
FS_data = [0.19, 6.14, 14.87, 26.68, 30.81, 26.8, 12.62, 0.1]/100; # FS exp prob
n_I = length(FS_Iwire);

# Flight time
tmax = 20e-6; # maximum flight time
tmin = -11e-6; # minimum flight time
tspan = (tmin,tmax);
ki = 0; # induction term, slows solution down // CQD-Theory

## ELECTRON/NUCLEAR MAGNETIC FIELDS
const Be = 5*mu0*mue/(16*pi*R^3) ;
const Bn = 5*mu0*mun/(16*pi*R^3) ;

## INITIAL CONDITIONS
# Nuclear Magnetic Moment
# thn_0 and phn_0 
if distri == "iso"
    thn0_list = thn_iso(N_sampling);
    print("Isotropic random sampling of co-quanta polar angles \n")
elseif distri == "hs"
    thn0_list = thn_heart(N_sampling);
    print("Heart-Shape random sampling of co-quanta polar angles\n")
elseif distri == "avg"
    thn0_list = avg_heart(N_sampling);
    print("<theta_n> of co-quanta polar angles\n")
else 
    print("Choose distribution for theta_n! \n") 
end
phn0_list = phn_iso(N_sampling);

# ELECTRON MAGNETIC MOMENT
# the_0 and phe_0 projected 
if field=="exact"
    the0 = 0 ; # initial theta_n0 (close but not equal to 0 unless additional steps taken to avoid cot(0))
    print("theta_e0 = 0 (exact field)  \n") 
elseif field=="quadrupole"  # Majorana assumes flipped when entering quadrupole due to full adiabatic rotation (flip normally from main field)
    the0 = pi ;    # initial theta_n0 (close but not equal to 0 unless additional steps taken to avoid cot(pi))
    print("theta_e0 = pi (quadrupole field) \n") 
end
phe0 = 0 ; # initial phi_e0

# Prepare for the final angles (nucleus and electron)
thef_list = zeros(n_I, N_sampling); # final theta_e
thnf_list = zeros(n_I, N_sampling); # final theta_n
phef_list = zeros(n_I, N_sampling); # final phi_e
phnf_list = zeros(n_I, N_sampling); # final phi_n


## MAIN LANDAU-LIFSHITZ EQUATION (BLOCH EQUATION)
if field =="exact"
    print("Using exact magnetic field! \n") 
    function BlochCQD!(du,u,p,t)
        # p[1]: Iwire_i             u[1]: theta_e
        # p[2]: v                   u[2]: theta_n
        # p[3]: za                  u[3]: phi_e
        #                           u[4]: phi_n

        # y- and z-components of the EXACT magnetic field: from the wire + remnant field
        Byt = mu0 * p[1] / (2*pi*(p[3]^2 + (p[2]*t)^2))* p[3] ;
        Bzt = Br - mu0 * p[1] / (2*pi*(p[3]^2 + (p[2]*t)^2))*p[2]*t ;

        du[1] = -gammae*(Bn*sin(u[2])*sin(u[4]-u[3]) + Byt*cos(u[3]))
        du[2] = 0 # -gamman*(Be*sin(u[1])*sin(u[3]-u[4]) + Byt*cos(u[4])) 

        u1_wrap = pi - abs(mod(u[1],2*pi)-pi) 
        if u1_wrap < tiny_angle || pi-u1_wrap < tiny_angle
            du[3] = 0
        else 
            du[3] = -gammae*(Bzt + Bn*cos(u[2]) - cot(u[1])*(Bn*sin(u[2])*cos(u[4]-u[3]) + Byt*sin(u[3])))
        end
        
        u2_wrap = pi - abs(mod(u[2],2*pi)-pi)
        if u2_wrap < tiny_angle || pi-u2_wrap < tiny_angle
            du[4] = 0
        else
            du[4] = -gamman*(Bzt + Be*cos(u[1]) - cot(u[2])*(Be*sin(u[1])*cos(u[3]-u[4]) + Byt*sin(u[4])))
        end
    end
elseif field == "quadrupole"
    print("Using quadrupole approximation of the magnetic field! \n") 
    function BlochCQD!(du,u,p,t)
        # p[1]: Iwire_i             u[1]: theta_e
        # p[2]: v                   u[2]: theta_n
        # p[3]: za                  u[3]: phi_e
        #                           u[4]: phi_n

        # y- and z- components of the QUADRUPOLE approximation of the field
        G = 2*pi*Br^2/(mu0*p[1])
        ynp = mu0*p[1]/(2*pi*Br)

        Byt = G*p[3]
        Bzt = G*(p[2]*t - ynp )

        du[1] = -gammae*(Bn*sin(u[2])*sin(u[4]-u[3]) + Byt*cos(u[3]))
        du[2] = 0 # -gamman*(Be*sin(u[1])*sin(u[3]-u[4]) + Byt*cos(u[4])) 

        u1_wrap = pi - abs(mod(u[1],2*pi)-pi) 
        if u1_wrap < tiny_angle || pi-u1_wrap < tiny_angle
            du[3] = 0
        else 
            du[3] = -gammae*(Bzt + Bn*cos(u[2]) - cot(u[1])*(Bn*sin(u[2])*cos(u[4]-u[3]) + Byt*sin(u[3])))
        end
        
        u2_wrap = pi - abs(mod(u[2],2*pi)-pi)
        if u2_wrap < tiny_angle || pi-u2_wrap < tiny_angle
            du[4] = 0
        else
            du[4] = -gamman*(Bzt + Be*cos(u[1]) - cot(u[2])*(Be*sin(u[1])*cos(u[3]-u[4]) + Byt*sin(u[4])))
        end
    end
else 
    print("Choose a defined field configuration for the inner rotation chamber \n") 
end


## FUNCTION TO ESTIMATE THE FINAL POLAR ANGLES FOR ELECTRON AND NUCLEUS
# theta_eF and theta_nF for a given initial condition (sweeps across Iwire)
function get_FS_run(tspan, u0_i, p_i)
    prob = ODEProblem(BlochCQD!,u0_i,tspan,p_i) # set up problem
    prob_func = (prob,i,repeat) -> remake(prob, p=[FS_Iwire[i],p_i[2],p_i[3]]) # repeat for each wire current
    ensemble_prob = EnsembleProblem(prob,prob_func=prob_func) # create ensemble problem
    @time sol = solve(ensemble_prob, alg, EnsembleThreads(), trajectories = n_I, saveat=time_save_step, dtmin=dtmin, maxiters=maxiters,reltol=reltol,abstol=abstol,force_dtmin=force_dtmin)

    Nt = length(sol.u[1][1,:]) # number of saved time steps (different from number of solver steps)
    start_t_ind = Int32(round(15*Nt/16)); # start point for integrating final value

    the_f_i = zeros(n_I); # final theta_e value at each wire current
    thn_f_i = zeros(n_I); # final theta_n value at each wire current
    phe_f_i = zeros(n_I); # final phi_e value at each wire current
    phn_f_i = zeros(n_I); # final phi_n value at each wire current

    for Ii=1:n_I
        the_f_i[Ii] = mean(sol.u[Ii][1,start_t_ind:end]);
        thn_f_i[Ii] = mean(sol.u[Ii][2,start_t_ind:end]);
        phe_f_i[Ii] = mean(sol.u[Ii][3,start_t_ind:end]);
        phn_f_i[Ii] = mean(sol.u[Ii][4,start_t_ind:end]);
    end

    return the_f_i, thn_f_i, phe_f_i, phn_f_i
end


# For each initial atom configuration (theta_e0, phi_e0, theta_n0, phi_n0)
# Find (theta_nF, theta_eF, phi_eF, phi_nF) over all wire currents
tot_iter = 1;   # counter
for l=1:N_sampling  # for each atom
    u0_i = [the0, thn0_list[l], phe0, phn0_list[l]]; # update only theta_n0 and phi_n0
    p_i = [FS_Iwire[1], v, za]; # reassign Iwire, v, za (these normally don't change unless multiple sweeps are done)

    print(tot_iter)
    global tot_iter +=1;
    thef_list[:,l], thnf_list[:,l], phef_list[:,l], phnf_list[:,l] = get_FS_run(tspan, u0_i, p_i)   # calculate theta_eF, theta_nF, phi_eF, phi_nF
end

thef_list = theta_wrap(thef_list); 
thnf_list = theta_wrap(thnf_list);

## CALCULATE FLIP RATE
boole_flip = thnf_list .< thef_list;  # boolean matrix corresponding to flip

branching = zeros(n_I,N_sampling) ;
for i=1:n_I
    for j=1:N_sampling
        if isnan(thnf_list[i,j]) || isnan(thef_list[i,j])
            branching[i,j] = NaN
        elseif thnf_list[i,j] < thef_list[i,j]
            branching[i,j] = 1
        else 
            branching[i,j] = 0
        end
    end
end

## SAVE DATA
# for python
data = zeros(N_sampling,2+5*n_I);
for k=1:N_sampling
    data[k,1] = thn0_list[k]
    data[k,2] = phn0_list[k]
    data[k,3:5:end] = thef_list'[k,:]
    data[k,4:5:end] = mod.(phef_list'[k,:],2*pi)
    data[k,5:5:end] = thnf_list'[k,:]
    data[k,6:5:end] = mod.(phnf_list'[k,:],2*pi)
    data[k,7:5:end] = branching'[k,:]
end
writedlm(filename*".csv",data,',')

nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)
FS_sfprob = vec(nanmean(data[:,7:5:end],1))
Rsquared = 1 - sum((FS_sfprob-FS_data).^2)/sum((FS_sfprob.-mean(FS_sfprob)).^2)

fig1=scatter(FS_Iwire,FS_data,
    label="Frisch–Segre experiment",
    markershape=:circle,
    markeralpha=0.85,
    markercolor=:red,
    markerstrokecolor=:red,
    markersize=4,
    xaxis=:log10,
    xminorticks=true,
    grid=true,
    xlim=(0.008,1),
    ylim=(-0.01,0.4),
    xlabel="Wire current (mA)", 
    ylabel="Fraction of spin-flip",  
    framestyle = :box,
    legend=:topleft,
    dpi=600)
fig1=annotate!(0.3, 0.35, text(L"R^2 = "*string(round(Rsquared; digits=4)), :black, :left, 10,"Computer Modern"))
fig1=scatter!([0.1],[0.1], label=" ", ms=0, mc=:white, msc=:white)
fig1=scatter!(FS_Iwire,FS_sfprob, label= "CQD-simulation",
markershape = :xcross,
markercolor= :blue,
markerstrokewidths=2,
markersize = 3,
dpi=600)
savefig(fig1,filename*".png")

t_run = Dates.canonicalize(Dates.now()-t_start)

open(filename*".txt","w") do file
    write(file,"Experiment \t = \t Frisch-Segre \n")
    write(file,"filename \t = \t $filename \n")
    write(file,"ODE System \t = \t BE3 \n")
    write(file,"ensemble \t = \t currents \n")
    write(file,"coord. \t = \t spherical \n")
    write(file,"Excep.Hand \t = \t wrapping \n")
    write(file,"z_a \t\t = \t $za \n")
    write(file,"k_i \t\t = \t $ki \n")
    write(file,"field \t = \t $field \n")
    write(file,"ang. dist. \t = \t $distri \n")
    write(file,"N sampling \t = \t $N_sampling \n")
    write(file,"time span \t = \t "*string(1e6.*tspan)*" us \n")
    write(file,"tiny angle \t = \t $tiny_angle \n")
    write(file,"reltol \t = \t $reltol \n")
    write(file,"abstol \t = \t $abstol \n")
    write(file,"dtmin \t = \t $dtmin \n")
    write(file,"start date \t = \t "*string(Dates.format(t_start, "yyyy-mm-ddTHH-MM-SS"))*"\n")
    write(file,"end date \t = \t "*string(Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS"))*"\n")
    write(file,"run time \t = \t $t_run \n")
    write(file,"hostname \t = \t $hostname \n")
    write(file,"Code name \t = \t $PROGRAM_FILE \n")
    write(file,"flip prob \t = \t "*string(FS_sfprob)*"\n" )
end

alert("script has finished!")
