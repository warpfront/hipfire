#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../../../kernels/src/gdn_conv_prep_bf16.gfx1201.hip"

#define HIP_OK(x) do { hipError_t e=(x); if(e!=hipSuccess){std::fprintf(stderr,"HIP %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e));std::exit(2);} } while(0)
static constexpr int HD=128, HK=16, HV=48, XD=10240;

static uint16_t bf16(float f) {
    uint32_t u; std::memcpy(&u,&f,4);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        if (u & 0x007fffffu) u |= 0x00400000u;
        return uint16_t(u >> 16);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
static float f32(uint16_t b) { uint32_t u=uint32_t(b)<<16; float f; std::memcpy(&f,&u,4); return f; }
static float silu(float x) { return x/(1.0f+std::exp(-x)); }
static float softplus(float x) { if(x>20) return x; if(x<-20) return std::exp(x); return std::log(1.0f+std::exp(x)); }

struct Result { double nrmse, max_abs; size_t differing, nonfinite; };
static Result compare(const std::vector<uint16_t>& got,const std::vector<uint16_t>& ref) {
    long double e2=0,r2=0; double ma=0; size_t d=0,nf=0;
    for(size_t i=0;i<got.size();++i){ double g=f32(got[i]),r=f32(ref[i]); if(!std::isfinite(g)) ++nf; double e=g-r; e2+=e*e; r2+=r*r; ma=std::max(ma,std::abs(e)); d += got[i]!=ref[i]; }
    return {std::sqrt((double)(e2/got.size()))/std::sqrt((double)(r2/got.size())),ma,d,nf};
}

static void run_case(int T, bool timed) {
    std::mt19937 rng(20260919u + unsigned(T));
    std::uniform_real_distribution<float> dx(-0.25f,0.25f), dw(-0.08f,0.08f), da(-2.0f,2.0f), db(-3.0f,3.0f);
    std::vector<float> x(size_t(T)*XD), w(size_t(XD)*4), state(size_t(XD)*3), a(size_t(T)*HV), b(size_t(T)*HV), dt(HV), al(HV);
    for(auto& z:x) z=dx(rng); for(auto& z:w) z=dw(rng); for(auto& z:state) z=dx(rng); for(auto& z:a) z=da(rng); for(auto& z:b) z=db(rng); for(auto& z:dt) z=0.25f*dx(rng); for(auto& z:al) z=-3.0f+dx(rng);
    const auto state0=state, a0=a, b0=b;
    std::vector<uint16_t> q(size_t(T)*HK*HD),k(q.size()),v(size_t(T)*HV*HD), qr(q.size()),kr(k.size()),vr(v.size());
    std::vector<float> ar=a,br=b, sr=state;
    const float qscale=0.08838834764831845f, eps=1e-6f;

    std::vector<float> conv(size_t(T)*XD);
    for(int c=0;c<XD;++c){ float z0=state0[c*3+2],z1=state0[c*3+1],z2=state0[c*3]; for(int t=0;t<T;++t){float cur=x[size_t(t)*XD+c]; float y=std::fma(w[c*4+3],cur,w[c*4+2]*z2); y=std::fma(w[c*4+1],z1,y); y=std::fma(w[c*4],z0,y); conv[size_t(t)*XD+c]=silu(y); z0=z1;z1=z2;z2=cur;} sr[c*3]=x[size_t(T-1)*XD+c]; sr[c*3+1]=T>=2?x[size_t(T-2)*XD+c]:state0[c*3]; sr[c*3+2]=T>=3?x[size_t(T-3)*XD+c]:(T==2?state0[c*3]:state0[c*3+1]); }
    for(int t=0;t<T;++t) for(int h=0;h<HK;++h){ float qs=0,ks=0; for(int d=0;d<HD;++d){float qv=conv[size_t(t)*XD+h*HD+d],kv=conv[size_t(t)*XD+(HK+h)*HD+d];qs+=qv*qv;ks+=kv*kv;} float qi=1/std::sqrt(qs+eps),ki=1/std::sqrt(ks+eps); for(int d=0;d<HD;++d){float qv=conv[size_t(t)*XD+h*HD+d]*qi; qv*=qscale; qr[(size_t(t)*HK+h)*HD+d]=bf16(qv); kr[(size_t(t)*HK+h)*HD+d]=bf16(conv[size_t(t)*XD+(HK+h)*HD+d]*ki);} }
    for(int t=0;t<T;++t) for(int h=0;h<HV;++h) for(int d=0;d<HD;++d) vr[(size_t(t)*HV+h)*HD+d]=bf16(conv[size_t(t)*XD+(2*HK+h)*HD+d]);
    for(int c0=0;c0<T;c0+=64) for(int h=0;h<HV;++h){float acc=0;for(int r=0;r<std::min(64,T-c0);++r){size_t o=size_t(c0+r)*HV+h;float g=-std::exp(al[h])*softplus(a0[o]+dt[h]);acc+=g;ar[o]=acc;br[o]=1/(1+std::exp(-b0[o]));}}

    float *xd,*wd,*sd,*ad,*bd,*dtd,*ald; uint16_t *qd,*kd,*vd;
    HIP_OK(hipMalloc(&xd,x.size()*4)); HIP_OK(hipMalloc(&wd,w.size()*4)); HIP_OK(hipMalloc(&sd,state.size()*4)); HIP_OK(hipMalloc(&ad,a.size()*4)); HIP_OK(hipMalloc(&bd,b.size()*4)); HIP_OK(hipMalloc(&dtd,dt.size()*4)); HIP_OK(hipMalloc(&ald,al.size()*4)); HIP_OK(hipMalloc(&qd,q.size()*2)); HIP_OK(hipMalloc(&kd,k.size()*2)); HIP_OK(hipMalloc(&vd,v.size()*2));
    HIP_OK(hipMemcpy(xd,x.data(),x.size()*4,hipMemcpyHostToDevice)); HIP_OK(hipMemcpy(wd,w.data(),w.size()*4,hipMemcpyHostToDevice)); HIP_OK(hipMemcpy(dtd,dt.data(),dt.size()*4,hipMemcpyHostToDevice)); HIP_OK(hipMemcpy(ald,al.data(),al.size()*4,hipMemcpyHostToDevice));
    auto reset=[&](){HIP_OK(hipMemcpy(sd,state0.data(),state0.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(ad,a0.data(),a0.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(bd,b0.data(),b0.size()*4,hipMemcpyHostToDevice));};
    auto launch=[&](){hipLaunchKernelGGL(gdn_conv_prep_bf16_gfx1201,dim3((T+63)/64,10),dim3(256),0,0,xd,wd,sd,ad,bd,dtd,ald,qd,kd,vd,T,qscale,eps);HIP_OK(hipGetLastError());};
    reset(); launch(); HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(q.data(),qd,q.size()*2,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(k.data(),kd,k.size()*2,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(v.data(),vd,v.size()*2,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(state.data(),sd,state.size()*4,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(a.data(),ad,a.size()*4,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(b.data(),bd,b.size()*4,hipMemcpyDeviceToHost));
    Result cq=compare(q,qr),ck=compare(k,kr),cv=compare(v,vr); size_t ring_bad=0,gate_nf=0,mono_bad=0; double gate_max=0,beta_max=0; for(size_t i=0;i<state.size();++i)ring_bad+=std::memcmp(&state[i],&sr[i],4)!=0; for(int c0=0;c0<T;c0+=64)for(int h=0;h<HV;++h){float prev=0;for(int r=0;r<std::min(64,T-c0);++r){size_t o=size_t(c0+r)*HV+h;gate_nf+=!std::isfinite(a[o])||!std::isfinite(b[o]);mono_bad+=a[o]>prev;prev=a[o];gate_max=std::max(gate_max,std::abs(double(a[o]-ar[o])));beta_max=std::max(beta_max,std::abs(double(b[o]-br[o])));}}
    std::printf("T=%d ring_bad=%zu gate_nonfinite=%zu gate_monotone_bad=%zu gate_max_abs=%.9g beta_max_abs=%.9g\n",T,ring_bad,gate_nf,mono_bad,gate_max,beta_max);
    std::printf("T=%d q_nrmse=%.9g q_max_abs=%.9g q_diff=%zu q_nonfinite=%zu k_nrmse=%.9g k_max_abs=%.9g k_diff=%zu k_nonfinite=%zu v_nrmse=%.9g v_max_abs=%.9g v_diff=%zu v_nonfinite=%zu\n",T,cq.nrmse,cq.max_abs,cq.differing,cq.nonfinite,ck.nrmse,ck.max_abs,ck.differing,ck.nonfinite,cv.nrmse,cv.max_abs,cv.differing,cv.nonfinite);
    if(ring_bad||gate_nf||mono_bad||cq.nrmse>0.002||ck.nrmse>0.002||cv.nrmse>0.002||cq.nonfinite||ck.nonfinite||cv.nonfinite) std::exit(3);
    if(timed){ hipEvent_t start,stop;HIP_OK(hipEventCreate(&start));HIP_OK(hipEventCreate(&stop));std::vector<float> ms;for(int i=0;i<101;++i){reset();HIP_OK(hipEventRecord(start));launch();HIP_OK(hipEventRecord(stop));HIP_OK(hipEventSynchronize(stop));float z;HIP_OK(hipEventElapsedTime(&z,start,stop));if(i)ms.push_back(z);}std::sort(ms.begin(),ms.end());std::printf("T=512 prep_median_us=%.3f prep_p10_us=%.3f prep_p90_us=%.3f\n",ms[ms.size()/2]*1000,ms[ms.size()/10]*1000,ms[ms.size()*9/10]*1000);HIP_OK(hipEventDestroy(start));HIP_OK(hipEventDestroy(stop));}
    hipFree(xd);hipFree(wd);hipFree(sd);hipFree(ad);hipFree(bd);hipFree(dtd);hipFree(ald);hipFree(qd);hipFree(kd);hipFree(vd);
}
int main(){run_case(512,true);run_case(129,false);return 0;}
