#include "fft.h"

constexpr unsigned EXTENT = 4;
typedef Brick<Dim<2,2,2>, Dim<1> > BrickType;
typedef BricksCuFFTPlan<BrickType, Dim<1> > PlanType;

int main()
{
    PlanType myPlan({EXTENT, EXTENT, EXTENT}, CUFFT_C2C);
}