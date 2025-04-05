use cust::{
    function::{BlockSize, GridSize},
    prelude::*,
};
use std::{error::Error, time::Instant};

const NUMBERS_LEN: usize = 2usize.pow(26);

static PTX: &str = include_str!("../saxpy.ptx");

fn main() -> Result<(), Box<dyn Error>> {
    let a: f32 = 2.0;
    let x = vec![2.0f32; NUMBERS_LEN];
    let y = vec![1.0f32; NUMBERS_LEN];
    let mut z = vec![0.0f32; NUMBERS_LEN];

    let total_bytes = (x.len() + y.len() + z.len()) * size_of::<f32>();
    let total_gigabytes = total_bytes as f64 / 10f64.powi(9);
    println!("starting saxpy with {total_gigabytes}GB of expected transfers");

    let _ctx = cust::quick_init()?;

    let module = Module::from_ptx(PTX, &[])?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let x_gpu = x.as_slice().as_dbuf()?;
    let y_gpu = y.as_slice().as_dbuf()?;
    let z_gpu = z.as_slice().as_dbuf()?;


    let func = module.get_function("saxpy")?;

    let grid_size = GridSize::x((NUMBERS_LEN / 128) as u32);
    let block_size = BlockSize::x(128);


    const ITERS: usize = 100;
    let start = Instant::now();
    for _ in 0..ITERS {
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                func<<<grid_size, block_size, 0, stream>>>(
                    a,
                    x_gpu.as_device_ptr(),
                    y_gpu.as_device_ptr(),
                    z_gpu.as_device_ptr(),
                    NUMBERS_LEN as u32
                )
            )?;
        }
    
        stream.synchronize()?;
    }
    let s = start.elapsed().as_secs_f64();
    
    println!("saxpy at {}GB/s",  total_gigabytes / s);
    
    z_gpu.copy_to(&mut z)?;

    println!("{} * {} + {} = {}", a, x[0], y[0], z[0]);

    Ok(())
}
