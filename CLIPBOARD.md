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

### Faster hit_meshes
```glsl
/*
 * This is a version of hit_meshes where during the search for the closest object
 * only the index of the found mesh and/or triangle are stored, and one last ray
 * trace is performed at the end to obtain more details. This avoids a bunch of 
 * memory writes and yields alone a 20% performance improvement over the older.
 */
HitRecord hit_meshes_old(Ray ray, inout uint mesh_tests) {
    float t = 1.0 / 0.0; // infinity
    uint triIdx = uint(-1);
    uint meshIdx;

    for (int i = 0; i < mesh_info.mesh_count; i++) {
        Mesh mesh = mesh_info.meshes[i];
        HitRecord current_hit;

        // automatically skipped for spheres
        for (int tri = 0; tri < mesh.triangle_count; tri++) {
            Triangle triangle = tri_buf.triangles[mesh.offset + tri];
            mesh_tests = mesh_tests + 1;

            current_hit = hit_triangle(triangle, ray, false);

            if (current_hit.did_hit && current_hit.t < t) {
                t = current_hit.t;
                meshIdx = i;
                triIdx = tri;
            }
        }

        if (mesh.triangle_count == 0) {
            vec3 center = tri_buf.triangles[mesh.offset].vertices[0];
            float radius = tri_buf.triangles[mesh.offset].vertices[1].r;

            current_hit = hit_sphere(center, radius, ray);

            if (current_hit.did_hit && current_hit.t < t) {
                t = current_hit.t;
                meshIdx = i;
                triIdx = uint(-1);
            }
        }
    }

    HitRecord hit;
    hit.did_hit = false;

    if (t < 1.0 / 0.0) {
        uint offset = mesh_info.meshes[meshIdx].offset;

        if (triIdx != uint(-1)) {
            Triangle triangle = tri_buf.triangles[offset + triIdx];

            hit = hit_triangle(triangle, ray, mesh_info.meshes[meshIdx].material.shaded_smooth);
            hit.material = mesh_info.meshes[meshIdx].material;

        } else {
            vec3 center = tri_buf.triangles[offset].vertices[0];
            float radius = tri_buf.triangles[offset].vertices[1].r;

            hit = hit_sphere(center, radius, ray);
            hit.material = mesh_info.meshes[meshIdx].material;
        }
    }

    return hit;
}```
