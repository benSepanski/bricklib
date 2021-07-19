#include "fft.h"

constexpr unsigned EXTENT = 4;
typedef Brick<Dim<2,2,2>, Dim<1> > BrickType;
typedef BricksCuFFTPlan<BrickType, Dim<1> > PlanType;

int main()
{
    // PlanType myPlan({EXTENT, EXTENT, EXTENT}, CUFFT_C2C);
    // std::cout << "Fourier dims: ";
    // for(unsigned i = 0; i < 1; ++i)
    // {
    //     std::cout << decltype(myPlan)::fourierDims[i] << " " ;
    // }
    // std::cout << std::endl;
    // std::cout << "Non-fourier dims: ";
    // for(unsigned i = 0; i < 2; ++i)
    // {
    //     std::cout << decltype(myPlan)::nonFourierDims[i] << " " ;
    // }
    // std::cout << std::endl;
}