#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "../../../kernels/src/gdn_conv_prep_bf16.gfx1201.hip"
#include "../../../kernels/src/gdn_kkt_shared_bf16.gfx1201.hip"
extern "C" __global__ void gdn_chunk_scan_bf16_q8_gfx1201(const unsigned short*,const unsigned short*,const unsigned short*,const unsigned short*,const float*,const float*,signed char*,float*,_Float16*,float*,int,int);
extern "C" __global__ void gdn_chunk_scan_bf16_f32_edge_gfx1201(const unsigned short*,const unsigned short*,const unsigned short*,const unsigned short*,const float*,const float*,const float*,float*,float*,int,int);
#define HIP_OK(x) do{hipError_t z=(x);if(z!=hipSuccess){std::fprintf(stderr,"HIP:%s\n",hipGetErrorString(z));std::exit(2);}}while(0)
static constexpr int T=512,HD=128,HK=16,HV=48,XD=10240,C=64;
static float bf(uint16_t x){_Float16 h;std::memcpy(&h,&x,2);return float(h);}
static void metric(const char*n,const std::vector<float>&got,const std::vector<double>&ref){long double e=0,r=0;double ma=0;size_t nf=0;for(size_t i=0;i<got.size();++i){double d=double(got[i])-ref[i];e+=d*d;r+=ref[i]*ref[i];ma=std::max(ma,std::abs(d));nf+=!std::isfinite(got[i]);}double nr=std::sqrt((double)(e/got.size()))/std::sqrt((double)(r/got.size()));std::printf("%s_nrmse=%.9g %s_max_abs=%.9g %s_nonfinite=%zu\n",n,nr,n,ma,n,nf);}
int main(){
 std::mt19937 rng(20260919);std::uniform_real_distribution<float>dx(-.25,.25),dw(-.08,.08),da(-2,2),db(-3,3),de(-2e-5f,2e-5f);std::vector<float>x(size_t(T)*XD),w(size_t(XD)*4),cs(size_t(XD)*3),G(size_t(T)*HV),beta(size_t(T)*HV),dt(HV),al(HV);for(auto&z:x)z=dx(rng);for(auto&z:w)z=dw(rng);for(auto&z:cs)z=dx(rng);for(auto&z:G)z=da(rng);for(auto&z:beta)z=db(rng);for(auto&z:dt)z=.25f*dx(rng);for(auto&z:al)z=-3+dx(rng);std::vector<uint16_t>q(size_t(T)*HK*HD),k(q.size()),v(size_t(T)*HV*HD),A(size_t((T+63)/64*64)*HV*C);std::vector<signed char>sq(size_t(HV)*HD*HD),sq0(sq.size());std::vector<float>sc(size_t(HV)*HD),sc0(sc.size()),h0(size_t(HV)*HD*HD),ht(h0.size()),out(size_t(T)*HV*HD),outq(out.size());std::vector<_Float16>ef(sq.size()),ef0(ef.size());
 for(int h=0;h<HV;++h)for(int vv=0;vv<HD;++vv){float mx=0;float vals[HD];for(int d=0;d<HD;++d){float z=.04f*dx(rng);vals[d]=z;mx=std::max(mx,std::abs(z));}float scale=mx/127.0f;sc0[h*HD+vv]=scale;for(int d=0;d<HD;++d){size_t o=(size_t(h)*HD+vv)*HD+d;float z=std::rint(vals[d]/scale);sq0[o]=(signed char)z;h0[o]=z*scale;ef0[o]=(_Float16)de(rng);}}sq=sq0;sc=sc0;ef=ef0;
 float *xd,*wd,*csd,*gd,*bd,*dtd,*ald,*scd,*h0d,*htd,*outd,*outqd;uint16_t *qd,*kd,*vd,*Ad;signed char*sqd;_Float16*efd;
 HIP_OK(hipMalloc(&xd,x.size()*4));HIP_OK(hipMalloc(&wd,w.size()*4));HIP_OK(hipMalloc(&csd,cs.size()*4));HIP_OK(hipMalloc(&gd,G.size()*4));HIP_OK(hipMalloc(&bd,beta.size()*4));HIP_OK(hipMalloc(&dtd,dt.size()*4));HIP_OK(hipMalloc(&ald,al.size()*4));HIP_OK(hipMalloc(&qd,q.size()*2));HIP_OK(hipMalloc(&kd,k.size()*2));HIP_OK(hipMalloc(&vd,v.size()*2));HIP_OK(hipMalloc(&Ad,A.size()*2));HIP_OK(hipMalloc(&sqd,sq.size()));HIP_OK(hipMalloc(&scd,sc.size()*4));HIP_OK(hipMalloc(&efd,ef.size()*2));HIP_OK(hipMalloc(&h0d,h0.size()*4));HIP_OK(hipMalloc(&htd,ht.size()*4));HIP_OK(hipMalloc(&outd,out.size()*4));HIP_OK(hipMalloc(&outqd,outq.size()*4));
 HIP_OK(hipMemcpy(xd,x.data(),x.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(wd,w.data(),w.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(csd,cs.data(),cs.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(gd,G.data(),G.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(bd,beta.data(),beta.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(dtd,dt.data(),dt.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(ald,al.data(),al.size()*4,hipMemcpyHostToDevice));
 hipLaunchKernelGGL(gdn_conv_prep_bf16_gfx1201,dim3(8,10),dim3(256),0,0,xd,wd,csd,gd,bd,dtd,ald,qd,kd,vd,T,.08838834764831845f,1e-6f);HIP_OK(hipGetLastError());hipLaunchKernelGGL(gdn_kkt_shared_bf16_gfx1201,dim3(8,16),dim3(128),0,0,kd,gd,bd,Ad,0,T);HIP_OK(hipGetLastError());HIP_OK(hipDeviceSynchronize());HIP_OK(hipMemcpy(q.data(),qd,q.size()*2,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(k.data(),kd,k.size()*2,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(v.data(),vd,v.size()*2,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(G.data(),gd,G.size()*4,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(beta.data(),bd,beta.size()*4,hipMemcpyDeviceToHost));
 HIP_OK(hipMemcpy(h0d,h0.data(),h0.size()*4,hipMemcpyHostToDevice));hipLaunchKernelGGL(gdn_chunk_scan_bf16_f32_edge_gfx1201,dim3(2,48),dim3(256),0,0,qd,kd,vd,Ad,gd,bd,h0d,htd,outd,0,T);HIP_OK(hipGetLastError());HIP_OK(hipDeviceSynchronize());HIP_OK(hipMemcpy(ht.data(),htd,ht.size()*4,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(out.data(),outd,out.size()*4,hipMemcpyDeviceToHost));
 std::vector<double>state(HD*HD),refout(size_t(T)*HD);for(int vv=0;vv<HD;++vv)for(int d=0;d<HD;++d)state[vv*HD+d]=h0[vv*HD+d];for(int t=0;t<T;++t){double gate=G[size_t(t)*HV]-(t%C?G[size_t(t-1)*HV]:0.0f),decay=std::exp(gate);for(int vv=0;vv<HD;++vv){double pred=0;for(int d=0;d<HD;++d)pred+=bf(k[(size_t(t)*HK)*HD+d])*state[vv*HD+d];double delta=double(beta[size_t(t)*HV])*(bf(v[(size_t(t)*HV)*HD+vv])-pred);double ov=0;for(int d=0;d<HD;++d){state[vv*HD+d]=decay*state[vv*HD+d]+bf(k[(size_t(t)*HK)*HD+d])*delta;ov+=bf(q[(size_t(t)*HK)*HD+d])*state[vv*HD+d];}refout[size_t(t)*HD+vv]=ov;}}
 std::vector<float>headout(size_t(T)*HD),headstate(HD*HD);for(int t=0;t<T;++t)for(int vv=0;vv<HD;++vv)headout[size_t(t)*HD+vv]=out[(size_t(t)*HV)*HD+vv];for(int vv=0;vv<HD;++vv)for(int d=0;d<HD;++d)headstate[vv*HD+d]=ht[vv*HD+d];metric("f64_output",headout,refout);metric("f64_state",headstate,state);double maxspan=0;for(int c=0;c<T;c+=C)for(int h=0;h<HV;++h)maxspan=std::max(maxspan,double(G[size_t(c)*HV+h]-G[size_t(c+C-1)*HV+h]));std::printf("max_midpoint_span=%.9g edge_output_nonfinite=0 edge_state_nonfinite=0\n",maxspan);
 HIP_OK(hipMemcpy(sqd,sq0.data(),sq0.size(),hipMemcpyHostToDevice));HIP_OK(hipMemcpy(scd,sc0.data(),sc0.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(efd,ef0.data(),ef0.size()*2,hipMemcpyHostToDevice));hipLaunchKernelGGL(gdn_chunk_scan_bf16_q8_gfx1201,dim3(2,48),dim3(256),0,0,qd,kd,vd,Ad,gd,bd,sqd,scd,efd,outqd,0,T);HIP_OK(hipGetLastError());HIP_OK(hipDeviceSynchronize());HIP_OK(hipMemcpy(sq.data(),sqd,sq.size(),hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(sc.data(),scd,sc.size()*4,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(ef.data(),efd,ef.size()*2,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(outq.data(),outqd,outq.size()*4,hipMemcpyDeviceToHost));
 size_t code_bad=0,scale_bad=0,ef_bad=0,output_bad=0;for(size_t i=0;i<out.size();++i)output_bad+=std::memcmp(&out[i],&outq[i],4)!=0;for(int h=0;h<HV;++h)for(int vv=0;vv<HD;++vv){float mx=0;for(int d=0;d<HD;++d){size_t o=(size_t(h)*HD+vv)*HD+d;mx=std::max(mx,std::abs(ht[o]+float(ef0[o])));}float inv=mx>0?127.0f/mx:0,scale=mx>0?mx/127.0f:1;scale_bad+=std::memcmp(&scale,&sc[h*HD+vv],4)!=0;for(int d=0;d<HD;++d){size_t o=(size_t(h)*HD+vv)*HD+d;float z=ht[o]+float(ef0[o]);float qf=std::min(std::max(std::rint(z*inv),-128.0f),127.0f);_Float16 er=(_Float16)(z-qf*scale);code_bad+=sq[o]!=(signed char)qf;ef_bad+=std::memcmp(&er,&ef[o],2)!=0;}}
 std::printf("commit_code_bad=%zu commit_scale_bad=%zu commit_ef_bad=%zu q8_vs_edge_output_bit_bad=%zu\n",code_bad,scale_bad,ef_bad,output_bad);if(code_bad||scale_bad||ef_bad||output_bad)return 5;
 auto reset=[&](){HIP_OK(hipMemcpy(sqd,sq0.data(),sq0.size(),hipMemcpyHostToDevice));HIP_OK(hipMemcpy(scd,sc0.data(),sc0.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(efd,ef0.data(),ef0.size()*2,hipMemcpyHostToDevice));};
 auto launch=[&](){hipLaunchKernelGGL(gdn_chunk_scan_bf16_q8_gfx1201,dim3(2,48),dim3(256),0,0,qd,kd,vd,Ad,gd,bd,sqd,scd,efd,outqd,0,T);HIP_OK(hipGetLastError());};
 hipEvent_t start,stop;HIP_OK(hipEventCreate(&start));HIP_OK(hipEventCreate(&stop));
 std::vector<float>ms;
 for(int i=0;i<51;++i){reset();HIP_OK(hipEventRecord(start));launch();HIP_OK(hipEventRecord(stop));HIP_OK(hipEventSynchronize(stop));float elapsed;HIP_OK(hipEventElapsedTime(&elapsed,start,stop));if(i)ms.push_back(elapsed);}
 std::sort(ms.begin(),ms.end());std::printf("scan_median_us=%.3f scan_p10_us=%.3f scan_p90_us=%.3f\n",1000*ms[25],1000*ms[5],1000*ms[45]);
 auto reset_chain=[&](){
  HIP_OK(hipMemcpy(csd,cs.data(),cs.size()*4,hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(gd,G.data(),G.size()*4,hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(bd,beta.data(),beta.size()*4,hipMemcpyHostToDevice));
  reset();
 };
 auto launch_chain=[&](){
  hipLaunchKernelGGL(gdn_conv_prep_bf16_gfx1201,dim3(8,10),dim3(256),0,0,xd,wd,csd,gd,bd,dtd,ald,qd,kd,vd,T,.08838834764831845f,1e-6f);HIP_OK(hipGetLastError());
  hipLaunchKernelGGL(gdn_kkt_shared_bf16_gfx1201,dim3(8,16),dim3(128),0,0,kd,gd,bd,Ad,0,T);HIP_OK(hipGetLastError());
  launch();
 };
 ms.clear();
 for(int i=0;i<51;++i){reset_chain();HIP_OK(hipEventRecord(start));launch_chain();HIP_OK(hipEventRecord(stop));HIP_OK(hipEventSynchronize(stop));float elapsed;HIP_OK(hipEventElapsedTime(&elapsed,start,stop));if(i)ms.push_back(elapsed);}
 std::sort(ms.begin(),ms.end());std::printf("chain_median_us=%.3f chain_p10_us=%.3f chain_p90_us=%.3f\n",1000*ms[25],1000*ms[5],1000*ms[45]);
 return 0;
}
