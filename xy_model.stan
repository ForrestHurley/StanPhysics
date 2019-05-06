    functions {
      real xy_energy(vector[,] spins) {

        real energy = 0;
        int spin_dim[4] = dims(spins);

        for (dir in 1:2) {
          
	  //calculate the offset for nearest neighbor calculations
          int dir_x = dir % 2 + spin_dim[1];
          int dir_y = dir / 2 + spin_dim[2];

          for (x in 1:spin_dim[1]) {
            for (y in 1:spin_dim[2]) {
	      //calculate the x,y index of the nearest neighbor
              int x_idx = ( (x + dir_x - 1) % spin_dim[1] ) + 1;
              int y_idx = ( (y + dir_y - 1) % spin_dim[2] ) + 1;

              energy += dot_product( spins[x_idx, y_idx], spins[x, y] );
            }
          }
        }
        
        return energy;
      }

      //given the temperature and energy, calculate the likelihood
      real canonical_ensemble_lpdf(real temperature, real energy) {
        return energy / temperature;
      }
    }
    data {
      real<lower=0> temp;
      int<lower=1> dim_x;
      int<lower=1> dim_y;
    }
    parameters {
      unit_vector[2] spin[dim_x, dim_y];
    }
    transformed parameters{
      real energy = xy_energy(spin);
      real energy_per_spin = energy / dim_x / dim_y;
    }
    model {
      temp ~ canonical_ensemble(energy);
    }
