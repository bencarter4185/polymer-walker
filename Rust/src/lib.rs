mod random;
use random::Ran2Generator;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use itertools::{iproduct, Itertools};

/*
Constants
*/
const PI_2: f64 = std::f64::consts::PI * 2.0;

/*
Structs
*/

struct Location {
    x: f64,
    y: f64,
    z: f64,
}

impl Location {
    fn new(x: f64, y: f64, z: f64) -> Location {
        let loc: Location = Location { x: x, y: y, z: z };

        loc
    }
}

fn next_angle(rng: &mut Ran2Generator) -> f64 {
    rng.next() * PI_2
}

#[allow(dead_code)]
fn main() {
    use pyo3::prelude::*;
    use pyo3::wrap_pyfunction;

    fn do_walk(a: f64, n: usize, seed: usize) -> (Vec<(f64, f64, f64)>, Vec<f64>) {
        // Generate vecs to hold the data for the path taken and squared displacement
        let mut data: Vec<(f64, f64, f64)> = Vec::new();
        let mut r_squared: Vec<f64> = Vec::new();

        // Let our initial seed = -1 and create our random number generator
        let idum: i32 = -1 * seed as i32;
        let mut rng: Ran2Generator = Ran2Generator::new(idum);

        // Warm up random number generator
        for _ in 0..100 {
            rng.next();
        }

        // Define our initial coordinates as (0.0, 0.0, 0.0)
        let mut location: Location = Location::new(0.0, 0.0, 0.0);

        // Push the initial point to the Vec
        data.push((location.x, location.y, location.z));
        r_squared.push(location.x.powi(2) + location.y.powi(2) + location.z.powi(2));

        // Should now have everything we need to do our random walk

        for _ in 0..n {
            let theta = next_angle(&mut rng);
            let phi = next_angle(&mut rng);
    
            // Update the new locations
            location.x += a * phi.cos() * theta.sin();
            location.y += a * phi.sin() * theta.sin();
            location.z += a * theta.cos();
    
            data.push((location.x, location.y, location.z));
    
            r_squared.push(location.x.powi(2) + location.y.powi(2) + location.z.powi(2))
        }

        (data, r_squared)
    }


    #[pyfunction]
    fn walk(a: f64, n: usize, seed: Option<usize>) -> PyResult<(Vec<(f64, f64, f64)>, Vec<f64>)> {
        /*
        Params:
            a: 64 bit float. Step distance of the random walk.
            n: 64 bit unsigned integer. Number of random steps to be taken.
        */

        let seed: usize = seed.unwrap_or(1);

        let (data, r_squared) = do_walk(a, n, seed);

        Ok((data, r_squared))
    }

    #[pyfunction]
    fn walk_parallel(a: f64, n: usize, max_seed: usize) -> PyResult<Vec<f64>> {
        /*
        In a parallel walk, we only want to store the r_squared values.

        Params:
            a: 64 bit float. Step distance of the random walk.
            n: 64 bit unsigned integer. Number of random steps to be taken.
            max_seed: 64 bit unsigned integer. Number of random number seeds to average over in our ensemble.
        */
        
        let results_len = (0..n).len();

        let seeds: Vec<usize> = (0..max_seed).collect_vec(); // Requires itertools

        let mut r_squared_vals = vec![0.0; results_len];

        let r_squared: Vec<Vec<f64>> = seeds.par_iter()
            .map(|seed| {
                let (_, r_squared) = do_walk(a, n, *seed);
                r_squared
            }).collect();
        
        // Average across the ensemble
        for (i, j) in iproduct!(0..max_seed, 0..results_len) {
            r_squared_vals[j] += r_squared[i][j] / max_seed as f64;
        }

        Ok(r_squared_vals)
    }

    /// A Python module implemented in Rust.
    #[pymodule]
    fn walker_rust(_: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(walk, m)?)?;
        m.add_function(wrap_pyfunction!(walk_parallel, m)?)?;

        Ok(())
    }
}
