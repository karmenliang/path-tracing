#ifndef RAY_H
#define RAY_H

#include "../vec3.h"

class ray
{
    public:
        ray() {}
	
        vec3 A; // ray origin
        vec3 B; // ray direction

        ray(const vec3& a, const vec3& b) { 
            A = a;
            B = b;
        }

        vec3 origin() const { return A; }

        vec3 direction() const { return B; }

        // t: value that changes 'destination' along ray
        vec3 point_at_parameter(float t) const {
            return A + t*B;
        }

};

#endif