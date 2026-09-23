#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "../../kernels/src/gdn_chunk_scan_prep.gfx1201.hip"
#include "../../kernels/src/gdn_chunk_scan_kkt_solve.gfx1201.hip"

extern "C" __global__ void gdn_chunk_scan(
    const unsigned short*, const unsigned short*, const unsigned short*, const unsigned short*,
    const float*, const float*, signed char*, float*, _Float16*, float*, int, int);
extern "C" __global__ void gdn_chunk_scan_baseline(
    const unsigned short*, const unsigned short*, const unsigned short*, const unsigned short*,
    const float*, const float*, signed char*, float*, _Float16*, float*, int, int);

#define HIP_OK(x) do { hipError_t z=(x); if(z!=hipSuccess) { \
    std::fprintf(stderr,"HIP:%s\n",hipGetErrorString(z)); std::exit(2); } } while(0)

static constexpr int HD=128, HK=16, HV=48, XD=10240, C=64;
static uint64_t hash64(const void* p, size_t n) {
    const uint8_t* b=(const uint8_t*)p; uint64_t h=1469598103934665603ull;
    for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ull;} return h;
}

template<class T> static size_t byte_mismatches(const std::vector<T>& a,const std::vector<T>& b) {
    const uint8_t* x=(const uint8_t*)a.data(); const uint8_t* y=(const uint8_t*)b.data();
    size_t bad=0; for(size_t i=0;i<a.size()*sizeof(T);++i) bad+=x[i]!=y[i]; return bad;
}

static float median(std::vector<float> x) {
    std::sort(x.begin(),x.end()); return x[x.size()/2];
}

int main(int argc,char** argv) {
    const int T=argc>1?std::atoi(argv[1]):512;
    if(T<1||T>512) return 3;
    std::mt19937 rng(20260919+T);
    std::uniform_real_distribution<float> dx(-.25f,.25f),dw(-.08f,.08f),da(-2,2),db(-3,3),de(-2e-5f,2e-5f);
    std::vector<float> x(size_t(T)*XD),w(size_t(XD)*4),cs(size_t(XD)*3),G(size_t(T)*HV),beta(size_t(T)*HV),dt(HV),al(HV);
    for(auto&z:x)z=dx(rng); for(auto&z:w)z=dw(rng); for(auto&z:cs)z=dx(rng);
    for(auto&z:G)z=da(rng); for(auto&z:beta)z=db(rng); for(auto&z:dt)z=.25f*dx(rng); for(auto&z:al)z=-3+dx(rng);
    std::vector<uint16_t> q(size_t(T)*HK*HD),k(q.size()),v(size_t(T)*HV*HD),A(size_t((T+C-1)/C*C)*HV*C);
    const size_t state_n=size_t(HV)*HD*HD,scale_n=size_t(HV)*HD,out_n=size_t(T)*HV*HD;
    std::vector<signed char> sq0(state_n); std::vector<float> sc0(scale_n); std::vector<_Float16> ef0(state_n);
    for(int h=0;h<HV;++h)for(int vv=0;vv<HD;++vv){float mx=0,vals[HD];for(int d=0;d<HD;++d){float z=.04f*dx(rng);vals[d]=z;mx=std::max(mx,std::abs(z));}float scale=mx/127;sc0[h*HD+vv]=scale;for(int d=0;d<HD;++d){size_t o=(size_t(h)*HD+vv)*HD+d;float z=std::rint(vals[d]/scale);sq0[o]=(signed char)z;ef0[o]=(_Float16)de(rng);}}

    float *xd,*wd,*csd,*gd,*bd,*dtd,*ald,*sc_base_d,*sc_new_d,*out_base_d,*out_new_d;
    uint16_t *qd,*kd,*vd,*Ad; signed char *sq_base_d,*sq_new_d; _Float16 *ef_base_d,*ef_new_d;
    HIP_OK(hipMalloc(&xd,x.size()*4)); HIP_OK(hipMalloc(&wd,w.size()*4)); HIP_OK(hipMalloc(&csd,cs.size()*4));
    HIP_OK(hipMalloc(&gd,G.size()*4)); HIP_OK(hipMalloc(&bd,beta.size()*4)); HIP_OK(hipMalloc(&dtd,dt.size()*4)); HIP_OK(hipMalloc(&ald,al.size()*4));
    HIP_OK(hipMalloc(&qd,q.size()*2)); HIP_OK(hipMalloc(&kd,k.size()*2)); HIP_OK(hipMalloc(&vd,v.size()*2)); HIP_OK(hipMalloc(&Ad,A.size()*2));
    HIP_OK(hipMalloc(&sq_base_d,state_n)); HIP_OK(hipMalloc(&sq_new_d,state_n)); HIP_OK(hipMalloc(&sc_base_d,scale_n*4)); HIP_OK(hipMalloc(&sc_new_d,scale_n*4));
    HIP_OK(hipMalloc(&ef_base_d,state_n*2)); HIP_OK(hipMalloc(&ef_new_d,state_n*2)); HIP_OK(hipMalloc(&out_base_d,out_n*4)); HIP_OK(hipMalloc(&out_new_d,out_n*4));
    HIP_OK(hipMemcpy(xd,x.data(),x.size()*4,hipMemcpyHostToDevice)); HIP_OK(hipMemcpy(wd,w.data(),w.size()*4,hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(csd,cs.data(),cs.size()*4,hipMemcpyHostToDevice)); HIP_OK(hipMemcpy(gd,G.data(),G.size()*4,hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(bd,beta.data(),beta.size()*4,hipMemcpyHostToDevice)); HIP_OK(hipMemcpy(dtd,dt.data(),dt.size()*4,hipMemcpyHostToDevice)); HIP_OK(hipMemcpy(ald,al.data(),al.size()*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(gdn_chunk_prep,dim3((T+63)/64,10),dim3(256),0,0,xd,wd,csd,gd,bd,dtd,ald,qd,kd,vd,T,.08838834764831845f,1e-6f);
    HIP_OK(hipGetLastError()); hipLaunchKernelGGL(gdn_chunk_kkt_solve,dim3((T+63)/64,16),dim3(128),0,0,kd,gd,bd,Ad,0,T); HIP_OK(hipGetLastError()); HIP_OK(hipDeviceSynchronize());

    auto reset=[&](signed char* sq,float* sc,_Float16* ef){HIP_OK(hipMemcpy(sq,sq0.data(),state_n,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(sc,sc0.data(),scale_n*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(ef,ef0.data(),state_n*2,hipMemcpyHostToDevice));};
    reset(sq_base_d,sc_base_d,ef_base_d); reset(sq_new_d,sc_new_d,ef_new_d);
    hipLaunchKernelGGL(gdn_chunk_scan_baseline,dim3(2,48),dim3(256),0,0,qd,kd,vd,Ad,gd,bd,sq_base_d,sc_base_d,ef_base_d,out_base_d,0,T); HIP_OK(hipGetLastError());
    hipLaunchKernelGGL(gdn_chunk_scan,dim3(1,48),dim3(512),0,0,qd,kd,vd,Ad,gd,bd,sq_new_d,sc_new_d,ef_new_d,out_new_d,0,T); HIP_OK(hipGetLastError()); HIP_OK(hipDeviceSynchronize());
    std::vector<signed char> sq_base(state_n),sq_new(state_n); std::vector<float> sc_base(scale_n),sc_new(scale_n),out_base(out_n),out_new(out_n); std::vector<_Float16> ef_base(state_n),ef_new(state_n);
    HIP_OK(hipMemcpy(sq_base.data(),sq_base_d,state_n,hipMemcpyDeviceToHost)); HIP_OK(hipMemcpy(sq_new.data(),sq_new_d,state_n,hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(sc_base.data(),sc_base_d,scale_n*4,hipMemcpyDeviceToHost)); HIP_OK(hipMemcpy(sc_new.data(),sc_new_d,scale_n*4,hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(ef_base.data(),ef_base_d,state_n*2,hipMemcpyDeviceToHost)); HIP_OK(hipMemcpy(ef_new.data(),ef_new_d,state_n*2,hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(out_base.data(),out_base_d,out_n*4,hipMemcpyDeviceToHost)); HIP_OK(hipMemcpy(out_new.data(),out_new_d,out_n*4,hipMemcpyDeviceToHost));
    size_t ob=byte_mismatches(out_base,out_new),qb=byte_mismatches(sq_base,sq_new),sb=byte_mismatches(sc_base,sc_new),eb=byte_mismatches(ef_base,ef_new);
    size_t float_bad=0,first_bad=out_n; float max_abs=0; size_t head_bad[HV]={0};
    for(size_t i=0;i<out_n;++i)if(std::memcmp(&out_base[i],&out_new[i],4)!=0){++float_bad;if(first_bad==out_n)first_bad=i;max_abs=std::max(max_abs,std::abs(out_base[i]-out_new[i]));++head_bad[(i/HD)%HV];}
    std::printf("float_bad=%zu first_bad=%zu base=%.9g new=%.9g max_abs=%.9g heads",float_bad,first_bad,first_bad<out_n?out_base[first_bad]:0,first_bad<out_n?out_new[first_bad]:0,max_abs);
    for(int h=0;h<HV;++h)if(head_bad[h])std::printf(" %d:%zu",h,head_bad[h]);std::printf("\n");

    hipEvent_t start,stop; HIP_OK(hipEventCreate(&start)); HIP_OK(hipEventCreate(&stop)); std::vector<float> bt,nt; bt.reserve(51);nt.reserve(51);
    for(int arm=0;arm<2;++arm)for(int it=0;it<52;++it){reset(arm?sq_new_d:sq_base_d,arm?sc_new_d:sc_base_d,arm?ef_new_d:ef_base_d);HIP_OK(hipEventRecord(start));if(arm)hipLaunchKernelGGL(gdn_chunk_scan,dim3(1,48),dim3(512),0,0,qd,kd,vd,Ad,gd,bd,sq_new_d,sc_new_d,ef_new_d,out_new_d,0,T);else hipLaunchKernelGGL(gdn_chunk_scan_baseline,dim3(2,48),dim3(256),0,0,qd,kd,vd,Ad,gd,bd,sq_base_d,sc_base_d,ef_base_d,out_base_d,0,T);HIP_OK(hipGetLastError());HIP_OK(hipEventRecord(stop));HIP_OK(hipEventSynchronize(stop));float ms=0;HIP_OK(hipEventElapsedTime(&ms,start,stop));if(it) (arm?nt:bt).push_back(ms*1000.0f);}
    float bm=median(bt),nm=median(nt);
    std::printf("T=%d output_byte_bad=%zu code_byte_bad=%zu scale_byte_bad=%zu ef_byte_bad=%zu baseline_us=%.3f dedup_us=%.3f gain_pct=%.3f out_hash=%016llx state_hash=%016llx\n",T,ob,qb,sb,eb,bm,nm,(bm/nm-1)*100,(unsigned long long)hash64(out_new.data(),out_n*4),(unsigned long long)hash64(sq_new.data(),state_n));
    return (ob||qb||sb||eb)?1:0;
}
