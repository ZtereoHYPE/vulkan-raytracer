use std::env;
use std::path::Path;
use std::fs;
use std::fs::File;
use std::io::BufWriter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read user input
    let args: Vec<String> = env::args().collect();
    let path = &args[1];
    let width: u32 = (&args)[2].parse().expect("usage: filename width height");
    let height: u32 = (&args)[3].parse().expect("usage: filename width height");


    // Read the dump file and convert the data
    let mut data = fs::read(path)?;
    for pixel in data.chunks_mut(4) {
        let swap: u8 = pixel[0];
        pixel[0] = pixel[2];
        pixel[2] = swap;
    }
    

    // Encode as PNG
    let path = Path::new(r"output.png");
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);
    
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::new(0.5));
    let mut writer = encoder.write_header().unwrap();
    
    writer.write_image_data(&data).unwrap();

    Ok(())
}
