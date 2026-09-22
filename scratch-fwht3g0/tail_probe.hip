#define main fwht3_g0_harness_main
#include "fwht3_g0.hip"
#undef main
int main() {
  HIP(hipSetDevice(0));hipDeviceProp_t p;HIP(hipGetDeviceProperties(&p,0));
  if(std::string(p.gcnArchName)!="gfx1100")return 2;
  EventPair ev;
  for(int n:{65,511,513,8191}) {
    Fixture f(253952,n);
    parity(f,ev);
  }
  return 0;
}
