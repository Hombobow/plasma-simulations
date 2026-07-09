#ifndef _FIELD_H
#define _FIELD_H

template <typename T>

class Field_
{
public:
    // constructor
    Field_(int ni, int nj, int nk) : ni{ni}, nj{nj}, nk{nk}
    {
        data = new T **[ni]; // ni pointers-to-pointers
        for (int i = 0; i < ni; i++)
        {
            data[i] = new T *[nj]; // nj pointers to Ts
            for (int j = 0; j < nj; j++)
                data[i][j] = new T[nk]; // nk Ts
        }

        operator=(0); // explicitly use the assignment operator
    }

    // destructor, frees memory in reverse order
    ~Field_()
    {
        if (data == nullptr)
            return; // return if unallocated

        for (int i = 0; i < ni; i++)
        { // release memory in reverse order
            for (int j = 0; j < nj; j++)
                delete[] data[i][j];
            delete[] data[i];
        }

        delete[] data;
        data = nullptr; // mark as free
    }

    // copy constructor
    Field_(const Field_ &other) : Field_{other.ni, other.nj, other.nk}
    {
        for (int i = 0; i < ni; i++)
        {
            for (int j = 0; j < nj; j++)
            {
                for (int k = 0; k < nk; k++)
                {
                    data[i][j][k] = other(i, j, k);
                }
            }
        }
    }

    // move constructor
    Field_(Field_ &&other) : ni{other.ni}, nj{other.nj}, nk{other.nk}
    {
        if (data)
            ~Field_();
        data = other.data;
        other.data = nullptr; // steal the data
    }

    // move assignment operator
    Field_ &operator=(Field_ &&f)
    {
        if (data)
        {
            ~Field_(); // deallocate own data
        }

        data = f.data;
        f.data = nullptr;
        return *this;
    }

    T **operator[](int i) { return data[i]; }

    Field_<T> &operator=(T s)
    {
        for (int i = 0; i < ni; i++)
        {
            for (int j = 0; j < nj; j++)
            {
                for (int k = 0; k < nk; k++)
                {
                    data[i][j][k] = s;
                }
            }
        }
        return *this;
    }

    // read-only access to data[i][j][k]
    T operator()(int i, int j, int k) const { return data[i][j][k]; }

    void operator/=(const Field_ &other)
    {
        for (int i = 0; i < ni; i++)
        {
            for (int j = 0; j < nj; j++)
            {
                for (int k = 0; k < nk; k++)
                {
                    if (other.data[i][j][k] != 0)
                    {
                        data[i][j][k] /= other[i][j][k];
                    }
                    else
                    {
                        data[i][j][k] = 0;
                    }
                }
            }
        }
    }

    Field_ &operator+=(const Field_ &other)
    {
        for (int i = 0; i < ni; i++)
        {
            for (int j = 0; j < nj; j++)
            {
                for (int k = 0; k < nk; k++)
                {
                    data[i][j][k] += other(i, j, k)
                }
            }
        }
        return (*this)
    }

    Field_ &operator*=(double s)
    {
        for (int i = 0; i < ni; i++)
        {
            for (int j = 0; j < nj; j++)
            {
                for (int k = 0; k < nk; k++)
                {
                    data[i][j][k] *= s;
                }
            }
        }
        return (*this)
    }

    friend Field_<T> operator*(double s, const Field_<T> &f)
    {
        Field_<T> r(f);
        return std::move(r *= s); // force move
    }

    std::ostream &operator<<(std::ostream &out, Field_<T> &f)
    {
        for (int k = 0; k < f.nk; k++, out << "\n")
            for (int j = 0; j < f.nj; j++)
                for (int i = 0; i < f.ni; i++)
                    out << f.data[i][j][k] << " ";
        return out;
    }

    const int ni, nj, nk; // number of nodes

protected:
    T ***data;
};

using Field = Field_<double>; // field of doubles
using FieldI = Field_<int>;   // field of integers

#endif // _FIELD_H
