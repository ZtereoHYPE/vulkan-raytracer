## This file is just a persistent clipboard for the developer to keep various versions of useful code snippets.

### Simpler hit_meshes (20% slower)
```glsl
HitRecord hit_meshes_old(Ray ray) {
    HitRecord hit;
    hit.t = 1.0 / 0.0; // infinity
    hit.did_hit = false;

    for (int i = 0; i < mesh_info.mesh_count; i++) {
        Mesh mesh = mesh_info.meshes[i];
        HitRecord current_hit;

        // automatically skipped for spheres
        for (int tri = 0; tri < mesh.triangle_count; tri++) {
            Triangle triangle = tri_buf.triangles[mesh.offset + tri];

            current_hit = hit_triangle(triangle, ray);

            if (current_hit.did_hit && current_hit.t < hit.t) {
                hit = current_hit;
                hit.material = mesh.material;
            }
        }

        // no triangles = it's a sphere
        if (mesh.triangle_count == 0) {
            vec3 center = tri_buf.triangles[mesh.offset].vertices[0];
            float radius = tri_buf.triangles[mesh.offset].vertices[1].r;

            current_hit = hit_sphere(center, radius, ray);

            if (current_hit.did_hit && current_hit.t < hit.t) {
                hit = current_hit;
                hit.material = mesh.material;
            }
        }
    }

    return hit;
}```