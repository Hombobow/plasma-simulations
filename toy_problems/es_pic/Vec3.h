#ifndef _VEC3_H
#define _VEC3_H

template <typename T>
struct vec3 {
    vec3(const T u, const T v, const T w) : d{u, v, w} {}
    vec3(const T a[3]) : d{a[0], a[1], a[2]} {}
    vec3() : d{0, 0, 0} {}

    T &operator[](int i) { return d[i]; }
    T operator()(int i) const { return d[i]; }

    vec3<T> &operator=(const T s)
    {
        d[0] = s;
        d[1] = s;
        d[2] = s;
        return *this;
    }

    vec3<T> &operator+=(vec3<T> o)
    {
        d[0] += o[0];
        d[1] += o[1];
        d[2] += o[2];
        return *this;
    }

    vec3<T> &operator-=(vec3<T> o)
    {
        d[0] -= o[0];
        d[1] -= o[1];
        d[2] -= o[2];
        return *this;
    }

protected:
    T d[3];
};

using double3 = vec3<double>;
using int3 = vec3<int>;

#endif // _VEC3_H
