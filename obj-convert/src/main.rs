use std::{error::Error, fs::File, io::stderr};

use serde::{Serialize, Deserialize};
use tobj::{Material, Model};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct SceneFile<'a> {
    version: &'a str,
    scene: Box<Vec<Mesh<'a>>>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Mesh<'a> {
    #[serde(rename = "type")]
    type_field: &'a str,
    material: SceneMaterial,
    data: Data,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct SceneMaterial {
    base_color: Option<[f32; 3]>,
    emission: Option<[f32; 3]>,
    reflectiveness: Option<f32>,
    roughness: Option<f32>,
    ior: Option<f32>,
    is_glass: Option<bool>,
    smooth_shading: Option<bool>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Data {
    vertices: Box<Vec<f32>>,
    normals: Box<Vec<f32>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut meshes: Vec<Mesh> = Vec::new();

    let (models, materials) = load_obj();

    for model in models.iter() {
        let mesh = &model.mesh;

        let mut vertices: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let material = get_material(materials.iter().nth(mesh.material_id.unwrap()).unwrap());

        let mut next_face = 0;
        for face in 0..mesh.face_arities.len() {
            let end = next_face + mesh.face_arities[face] as usize;

            // for quad faces
            if mesh.face_arities[face] == 4 {
                let face_indices = &mesh.indices[next_face..end];
                // 3 vertices of first triangle
                vertices.push(mesh.positions[(face_indices[0]*3+0) as usize]);
                vertices.push(mesh.positions[(face_indices[0]*3+1) as usize]);
                vertices.push(mesh.positions[(face_indices[0]*3+2) as usize]);
                vertices.push(mesh.positions[(face_indices[1]*3+0) as usize]);
                vertices.push(mesh.positions[(face_indices[1]*3+1) as usize]);
                vertices.push(mesh.positions[(face_indices[1]*3+2) as usize]);
                vertices.push(mesh.positions[(face_indices[2]*3+0) as usize]);
                vertices.push(mesh.positions[(face_indices[2]*3+1) as usize]);
                vertices.push(mesh.positions[(face_indices[2]*3+2) as usize]);

                // 3 vertices of second triangle
                vertices.push(mesh.positions[(face_indices[0]*3+0) as usize]);
                vertices.push(mesh.positions[(face_indices[0]*3+1) as usize]);
                vertices.push(mesh.positions[(face_indices[0]*3+2) as usize]);
                vertices.push(mesh.positions[(face_indices[2]*3+0) as usize]);
                vertices.push(mesh.positions[(face_indices[2]*3+1) as usize]);
                vertices.push(mesh.positions[(face_indices[2]*3+2) as usize]);
                vertices.push(mesh.positions[(face_indices[3]*3+0) as usize]);
                vertices.push(mesh.positions[(face_indices[3]*3+1) as usize]);
                vertices.push(mesh.positions[(face_indices[3]*3+2) as usize]);
    
                if !mesh.normal_indices.is_empty() {
                    let normal_face_indices = &mesh.normal_indices[next_face..end];

                    normals.push(mesh.normals[(normal_face_indices[0]*3+0) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[0]*3+1) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[0]*3+2) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[1]*3+0) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[1]*3+1) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[1]*3+2) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[2]*3+0) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[2]*3+1) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[2]*3+2) as usize]);
                    
                    normals.push(mesh.normals[(normal_face_indices[0]*3+0) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[0]*3+1) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[0]*3+2) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[2]*3+0) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[2]*3+1) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[2]*3+2) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[3]*3+0) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[3]*3+1) as usize]);
                    normals.push(mesh.normals[(normal_face_indices[3]*3+2) as usize]);
                }
            } else {
                panic!("Only quads are supported at the moment"); 
            }

            next_face = end;
        }

        meshes.push(Mesh {
            type_field: "TriMesh",
            material,
            data: Data { 
                vertices: Box::new(vertices), 
                normals: Box::new(normals) 
            }
        });
    }

    let scene = SceneFile {
        version: "0.1",
        scene: Box::new(meshes),
    };

    let file: File = File::create("out.yaml")?;
    serde_yaml::to_writer(file, &scene)?;

    Ok(())
}

fn load_obj() -> (Vec<Model>, Vec<Material>) {
    let obj_file = std::env::args()
    .skip(1)
    .next()
    .expect("Please input the obj filepath");

    let (models, materials) =
    tobj::load_obj(
        &obj_file,
        &tobj::LoadOptions::default()
        )
        .expect("Failed to obj");

    let materials = materials.expect("materials file not found");

    return (models, materials);
}

fn get_material(mat: &Material) -> SceneMaterial {
    let ke = mat.unknown_param.get("Ke");
    let emi: [f32; 3] = if let None = ke {
        [0.0, 0.0, 0.0]
    } else {
        let a = ke.unwrap().split(" ").filter_map(|s| s.parse::<f32>().ok()).collect::<Vec<f32>>();
        [a[0], a[1], a[2]]
    };

    SceneMaterial {
        base_color: mat.diffuse.or(Some([0.0,0.0,0.0])),
        emission: Some(emi),
        reflectiveness: None,
        roughness: None,
        ior: None,
        is_glass: None,
        smooth_shading: None,
    }
}