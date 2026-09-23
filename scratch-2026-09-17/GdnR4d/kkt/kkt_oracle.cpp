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
#define HIP_OK(x) do{hipError_t z=(x);if(z!=hipSuccess){std::fprintf(stderr,"HIP:%s\n",hipGetErrorString(z));std::exit(2);}}while(0)
static constexpr int T=512,HD=128,HK=16,HV=48,XD=10240,C=64;
static float bf(uint16_t x){_Float16 h;std::memcpy(&h,&x,2);return float(h);}
using Mat=std::vector<double>;
static Mat inverse_ref(const uint16_t* k,const float* G,const float* beta,int c0,int h){
 Mat m(C*C);for(int i=0;i<C;++i){m[i*C+i]=1;for(int j=0;j<i;++j){double dot=0;for(int d=0;d<HD;++d)dot+=double(bf(k[(size_t(c0+i)*HK+h)*HD+d]))*bf(k[(size_t(c0+j)*HK+h)*HD+d]);m[i*C+j]=double(beta[size_t(c0+i)*HV+h*3])*std::exp(double(G[size_t(c0+i)*HV+h*3]-G[size_t(c0+j)*HV+h*3]))*dot;}}
 Mat a(C*C);for(int i=0;i<C;++i)for(int col=0;col<C;++col){double z=i==col?1:0;for(int j=0;j<i;++j)z-=m[i*C+j]*a[j*C+col];a[i*C+col]=z;}return a;
}
struct ScanResult{std::vector<double> out,state;};
static ScanResult scan_one(const std::vector<uint16_t>& q,const std::vector<uint16_t>& k,const std::vector<uint16_t>& v,const std::vector<float>& G,const std::vector<float>& beta,const std::vector<Mat>& aa){
 std::mt19937 rng(17);std::uniform_real_distribution<double> ds(-.02,.02);ScanResult z;z.out.resize(size_t(T)*HD);z.state.resize(HD*HD);for(auto&x:z.state)x=ds(rng);
 for(int chunk=0;chunk<8;++chunk){int c0=chunk*C;const Mat&A=aa[chunk];std::vector<double> U(C*HD),W(C*HD),D(C*HD),qk(C*C);
  for(int i=0;i<C;++i)for(int vv=0;vv<HD;++vv){double u=0;for(int d=0;d<HD;++d)u+=bf(k[(size_t(c0+i)*HK)*HD+d])*z.state[vv*HD+d];U[i*HD+vv]=u;W[i*HD+vv]=beta[size_t(c0+i)*HV]*(bf(v[(size_t(c0+i)*HV)*HD+vv])-std::exp(G[size_t(c0+i)*HV])*u);}
  for(int i=0;i<C;++i)for(int vv=0;vv<HD;++vv){double d=0;for(int j=0;j<C;++j)d+=A[i*C+j]*W[j*HD+vv];D[i*HD+vv]=d;}
  double mid=.5*(G[size_t(c0)*HV]+G[size_t(c0+C-1)*HV]);
  for(int i=0;i<C;++i)for(int j=0;j<=i;++j){double d=0;for(int x=0;x<HD;++x)d+=bf(q[(size_t(c0+i)*HK)*HD+x])*bf(k[(size_t(c0+j)*HK)*HD+x]);qk[i*C+j]=d;}
  for(int i=0;i<C;++i)for(int vv=0;vv<HD;++vv){double o=0;for(int d=0;d<HD;++d)o+=bf(q[(size_t(c0+i)*HK)*HD+d])*z.state[vv*HD+d];o*=std::exp(G[size_t(c0+i)*HV]);for(int j=0;j<=i;++j)o+=std::exp(G[size_t(c0+i)*HV]-mid)*qk[i*C+j]*std::exp(mid-G[size_t(c0+j)*HV])*D[j*HD+vv];z.out[size_t(c0+i)*HD+vv]=o;}
  std::vector<double> ns(HD*HD);double tail=std::exp(G[size_t(c0+C-1)*HV]-mid),em=std::exp(mid);for(int vv=0;vv<HD;++vv)for(int d=0;d<HD;++d){double u=0;for(int i=0;i<C;++i)u+=std::exp(mid-G[size_t(c0+i)*HV])*D[i*HD+vv]*bf(k[(size_t(c0+i)*HK)*HD+d]);ns[vv*HD+d]=tail*(em*z.state[vv*HD+d]+u);}z.state.swap(ns);
 }return z;
}
static void metric(const char*n,const std::vector<double>&a,const std::vector<double>&b){long double e=0,r=0;double ma=0;size_t nf=0;for(size_t i=0;i<a.size();++i){double d=a[i]-b[i];e+=d*d;r+=b[i]*b[i];ma=std::max(ma,std::abs(d));nf+=!std::isfinite(a[i]);}double nr=std::sqrt((double)(e/a.size()))/std::sqrt((double)(r/a.size()));std::printf("%s_nrmse=%.9g %s_max_abs=%.9g %s_nonfinite=%zu\n",n,nr,n,ma,n,nf);if(nr>.002||nf)std::exit(4);}
int main(){
 std::mt19937 rng(20260919);std::uniform_real_distribution<float> dx(-.25,.25),dw(-.08,.08),da(-2,2),db(-3,3);std::vector<float>x(size_t(T)*XD),w(size_t(XD)*4),s(size_t(XD)*3),G(size_t(T)*HV),beta(size_t(T)*HV),dt(HV),al(HV);for(auto&z:x)z=dx(rng);for(auto&z:w)z=dw(rng);for(auto&z:s)z=dx(rng);for(auto&z:G)z=da(rng);for(auto&z:beta)z=db(rng);for(auto&z:dt)z=.25f*dx(rng);for(auto&z:al)z=-3+dx(rng);std::vector<uint16_t>q(size_t(T)*HK*HD),k(q.size()),v(size_t(T)*HV*HD),A(size_t(T)*HV*C);
 float *xd,*wd,*sd,*gd,*bd,*dtd,*ald;uint16_t *qd,*kd,*vd,*Ad;HIP_OK(hipMalloc(&xd,x.size()*4));HIP_OK(hipMalloc(&wd,w.size()*4));HIP_OK(hipMalloc(&sd,s.size()*4));HIP_OK(hipMalloc(&gd,G.size()*4));HIP_OK(hipMalloc(&bd,beta.size()*4));HIP_OK(hipMalloc(&dtd,dt.size()*4));HIP_OK(hipMalloc(&ald,al.size()*4));HIP_OK(hipMalloc(&qd,q.size()*2));HIP_OK(hipMalloc(&kd,k.size()*2));HIP_OK(hipMalloc(&vd,v.size()*2));HIP_OK(hipMalloc(&Ad,A.size()*2));HIP_OK(hipMemcpy(xd,x.data(),x.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(wd,w.data(),w.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(sd,s.data(),s.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(gd,G.data(),G.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(bd,beta.data(),beta.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(dtd,dt.data(),dt.size()*4,hipMemcpyHostToDevice));HIP_OK(hipMemcpy(ald,al.data(),al.size()*4,hipMemcpyHostToDevice));
 hipLaunchKernelGGL(gdn_conv_prep_bf16_gfx1201,dim3(8,10),dim3(256),0,0,xd,wd,sd,gd,bd,dtd,ald,qd,kd,vd,T,.08838834764831845f,1e-6f);HIP_OK(hipGetLastError());HIP_OK(hipDeviceSynchronize());HIP_OK(hipMemcpy(q.data(),qd,q.size()*2,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(k.data(),kd,k.size()*2,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(v.data(),vd,v.size()*2,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(G.data(),gd,G.size()*4,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(beta.data(),bd,beta.size()*4,hipMemcpyDeviceToHost));
 auto launch=[&](){hipLaunchKernelGGL(gdn_kkt_shared_bf16_gfx1201,dim3(8,16),dim3(128),0,0,kd,gd,bd,Ad,0,T);HIP_OK(hipGetLastError());};launch();HIP_OK(hipDeviceSynchronize());HIP_OK(hipMemcpy(A.data(),Ad,A.size()*2,hipMemcpyDeviceToHost));
 double residual=0,ainv_e2=0,ainv_r2=0,ainv_ma=0;size_t nf=0,pad_bad=0;std::vector<Mat> ref8,cand8;for(int chunk=0;chunk<8;++chunk)for(int kh=0;kh<HK;++kh){for(int sub=0;sub<3;++sub){int vh=kh*3+sub;Mat m(C*C),ref(C*C),cand(C*C);for(int i=0;i<C;++i){m[i*C+i]=1;for(int j=0;j<i;++j){double dot=0;for(int d=0;d<HD;++d)dot+=double(bf(k[(size_t(chunk*C+i)*HK+kh)*HD+d]))*bf(k[(size_t(chunk*C+j)*HK+kh)*HD+d]);m[i*C+j]=double(beta[size_t(chunk*C+i)*HV+vh])*std::exp(double(G[size_t(chunk*C+i)*HV+vh]-G[size_t(chunk*C+j)*HV+vh]))*dot;}}for(int i=0;i<C;++i)for(int col=0;col<C;++col){double z=i==col?1:0;for(int j=0;j<i;++j)z-=m[i*C+j]*ref[j*C+col];ref[i*C+col]=z;cand[i*C+col]=bf(A[(size_t(chunk*C+i)*HV+vh)*C+col]);double d=cand[i*C+col]-z;ainv_e2+=d*d;ainv_r2+=z*z;ainv_ma=std::max(ainv_ma,std::abs(d));nf+=!std::isfinite(cand[i*C+col]);}for(int i=0;i<C;++i)for(int col=0;col<C;++col){double z=0;for(int j=0;j<C;++j)z+=m[i*C+j]*cand[j*C+col];residual=std::max(residual,std::abs(z-(i==col)));}if(vh==0){ref8.push_back(ref);cand8.push_back(cand);}}}
 std::printf("A_nrmse=%.9g A_max_abs=%.9g inverse_residual_inf=%.9g A_nonfinite=%zu padded_identity_bad=%zu\n",std::sqrt(ainv_e2/A.size())/std::sqrt(ainv_r2/A.size()),ainv_ma,residual,nf,pad_bad);if(nf||residual>.02) return 3;ScanResult sr=scan_one(q,k,v,G,beta,ref8),sc=scan_one(q,k,v,G,beta,cand8);metric("scan_output",sc.out,sr.out);metric("scan_state",sc.state,sr.state);
 hipEvent_t start,stop;HIP_OK(hipEventCreate(&start));HIP_OK(hipEventCreate(&stop));std::vector<float>ms;for(int i=0;i<101;++i){HIP_OK(hipEventRecord(start));launch();HIP_OK(hipEventRecord(stop));HIP_OK(hipEventSynchronize(stop));float elapsed;HIP_OK(hipEventElapsedTime(&elapsed,start,stop));if(i)ms.push_back(elapsed);}std::sort(ms.begin(),ms.end());std::printf("kkt_median_us=%.3f kkt_p10_us=%.3f kkt_p90_us=%.3f\n",1000*ms[50],1000*ms[10],1000*ms[90]);
 HIP_OK(hipMemset(Ad,0,A.size()*2));hipLaunchKernelGGL(gdn_kkt_shared_bf16_gfx1201,dim3(3,16),dim3(128),0,0,kd,gd,bd,Ad,0,129);HIP_OK(hipGetLastError());HIP_OK(hipDeviceSynchronize());HIP_OK(hipMemcpy(A.data(),Ad,A.size()*2,hipMemcpyDeviceToHost));size_t ragged_bad=0,ragged_nonfinite=0;for(int h=0;h<HV;++h)for(int r=1;r<C;++r)for(int c=0;c<C;++c){float av=bf(A[(size_t(128+r)*HV+h)*C+c]);ragged_nonfinite+=!std::isfinite(av);ragged_bad+=av!=(r==c?1.0f:0.0f);}std::printf("T129_padded_identity_bad=%zu T129_nonfinite=%zu\n",ragged_bad,ragged_nonfinite);if(ragged_bad||ragged_nonfinite)return 5;return 0;
}
