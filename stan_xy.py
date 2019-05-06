from pystan import StanModel
import statistics
import numpy as np
import os

#load and compile model (may take a few minutes)
sm = StanModel(file="xy_model.stan")

#set out file
filename = "energy_data.csv"

#write file header if necessary
if os.path.exists(filename):
    pass
else:
    with open(filename, 'w') as f:
        f.write("temp,dim_x,dim_y,energy,energy_var,c,vortex_density,rhat,n_eff\n")

#basic parameters
chains = 4
vorticity_samples = 1000
iterations = 5000
dim_x = dim_y = 4

#create the initial state for the lowest temperature calculation
unit_vect_array = np.zeros((dim_x,dim_y,2),dtype='float')
unit_vect_array[:,:,1] = 1.
init = [dict(spin=unit_vect_array)] * chains

#calculate vorticity given a single set of spins
def vorticity_calc(spin_array):
    rolled = [spin_array,
            np.roll(spin_array,1,0),
            np.roll(spin_array,1,(0,1)),
            np.roll(spin_array,1,1)]

    #calculate the dot product for the shortest angle
    dotted = []
    for i in range(len(rolled)):
        dotted.append(np.sum(np.multiply(rolled[i-1],rolled[i]),axis=-1))

    #calculate the cross product for the direction of the shortest angle
    crossed = []
    for i in range(len(rolled)):
        crossed.append(np.cross(rolled[i-1],rolled[i],axis=-1))

    angles = [np.arccos(dot) * np.sign(cross) for dot, cross in zip(dotted,crossed)]

    #sum the angles around each lattice square
    around_angle = np.sum(angles,axis=0)

    #find vortices
    vortex = np.where(around_angle>1,1,0)

    #count the number of vortices
    return np.sum(vortex,axis=(-1,-2))


for i in range(20):
  temp = (i + 1.) * 0.1
  print("Using temp:", temp)

  #total number of lattice sites
  N = dim_x * dim_y

  #initialize the data dictionary
  data = dict(dim_x=dim_x, dim_y=dim_y, temp=temp)

  #simulate the distribution
  fit = sm.sampling(
        data=data,chains=chains,
        iter=iterations,init=init,
        verbose=True,check_hmc_diagnostics=True,
        control = dict(adapt_delta=0.7,
                       max_treedepth=15))
  
  print(fit.stansummary(pars="energy_per_spin"))
  
  #Calculate and print various statistics related to the distribution
  energy = statistics.mean(fit.extract(pars="energy_per_spin")["energy_per_spin"])
  energy_var = statistics.variance(fit.extract(pars="energy_per_spin")["energy_per_spin"])
  specific_heat = statistics.variance(fit.extract(pars="energy")["energy"]) \
          / N / temp / temp
  vorticity = np.mean([vorticity_calc(x) for x in 
          fit.extract(pars="spin",permuted=True)["spin"][:vorticity_samples]])
  vortex_density = vorticity / float(N)
  print("Energy:",energy,"Specific heat:",specific_heat,"Vortex Density:",vortex_density)

  #Update the initial state so that the next temperature simulation starts from the
  #last state of this one
  init = [dict(spin=block) for block in 
          fit.extract(pars="spin",permuted=False)["spin"][-1]]

  #get debugging and diagnostics
  colnames = fit.summary(pars="energy_per_spin")['summary_colnames']
  rhat_index = colnames.index("Rhat")
  n_eff_index = colnames.index("n_eff")
  rhat = fit.summary(pars="energy_per_spin")["summary"][0,rhat_index]
  n_eff = fit.summary(pars="energy_per_spin")["summary"][0,n_eff_index]

  #save results
  with open(filename, 'a') as f:
      f.write("{},{},{},{},{},{},{},{},{}\n"
              .format(
                  temp,dim_x,dim_y,
                  energy,energy_var,
                  specific_heat,
                  vortex_density,
                  rhat,n_eff))
