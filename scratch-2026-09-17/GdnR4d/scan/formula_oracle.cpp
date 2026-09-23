#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" __global__ void gdn_chunk_scan_bf16_f32_edge_gfx1201(
    const unsigned short*, const unsigned short*, const unsigned short*,
    const unsigned short*, const float*, const float*, const float*, float*,
    float*, int, int);

#define HIP_OK(x) do { hipError_t z=(x); if(z!=hipSuccess) { \
    std::fprintf(stderr,"HIP: %s\n",hipGetErrorString(z)); std::exit(2); } } while(0)

static constexpr int T=64, HD=128, HK=16, HV=48, C=64;

static uint16_t to_f16(float x) {
    _Float16 h=(_Float16)x; uint16_t u; std::memcpy(&u,&h,2); return u;
}
static double from_f16(uint16_t u) {
    _Float16 h; std::memcpy(&h,&u,2); return (double)h;
}

static void metric_dd(const char* name, const std::vector<double>& got,
                      const std::vector<double>& ref) {
    long double e2=0, r2=0; double ma=0;
    for(size_t i=0;i<got.size();++i) {
        double d=got[i]-ref[i]; e2+=d*d; r2+=ref[i]*ref[i];
        ma=std::max(ma,std::abs(d));
    }
    std::printf("%s_nrmse=%.12g %s_max_abs=%.12g\n",name,
        std::sqrt((double)e2/(double)r2),name,ma);
}
static void metric_fd(const char* name, const std::vector<float>& got,
                      const std::vector<double>& ref) {
    long double e2=0, r2=0; double ma=0;
    for(size_t i=0;i<got.size();++i) {
        double d=(double)got[i]-ref[i]; e2+=d*d; r2+=ref[i]*ref[i];
        ma=std::max(ma,std::abs(d));
    }
    std::printf("%s_nrmse=%.12g %s_max_abs=%.12g\n",name,
        std::sqrt((double)e2/(double)r2),name,ma);
}

struct Ref { std::vector<double> out, state; };

static Ref serial(const std::vector<uint16_t>& q,
                  const std::vector<uint16_t>& k,
                  const std::vector<uint16_t>& v,
                  const std::vector<float>& G,
                  const std::vector<float>& beta,
                  bool decay_prediction) {
    Ref r{std::vector<double>(T*HD),std::vector<double>(HD*HD,0.0)};
    for(int t=0;t<T;++t) {
        const double g=(double)G[t*HV]-(t?G[(t-1)*HV]:0.0f);
        const double alpha=std::exp(g);
        for(int vv=0;vv<HD;++vv) {
            double sk=0;
            for(int d=0;d<HD;++d)
                sk+=r.state[vv*HD+d]*from_f16(k[(t*HK)*HD+d]);
            const double predicted=decay_prediction?alpha*sk:sk;
            const double delta=(double)beta[t*HV]*
                (from_f16(v[(t*HV)*HD+vv])-predicted);
            double y=0;
            for(int d=0;d<HD;++d) {
                r.state[vv*HD+d]=alpha*r.state[vv*HD+d]+
                    from_f16(k[(t*HK)*HD+d])*delta;
                y+=r.state[vv*HD+d]*from_f16(q[(t*HK)*HD+d]);
            }
            r.out[t*HD+vv]=y;
        }
    }
    return r;
}

static std::vector<double> inverse_for_head(const std::vector<uint16_t>& k,
                                             const std::vector<float>& G,
                                             const std::vector<float>& beta,
                                             int head) {
    const int kh=head/3;
    std::vector<double> a(C*C,0.0);
    for(int i=0;i<C;++i) {
        for(int col=0;col<C;++col) {
            double z=(i==col)?1.0:0.0;
            for(int j=0;j<i;++j) {
                double dot=0;
                for(int d=0;d<HD;++d)
                    dot+=from_f16(k[(i*HK+kh)*HD+d])*
                         from_f16(k[(j*HK+kh)*HD+d]);
                const double lij=(double)beta[i*HV+head]*
                    std::exp((double)G[i*HV+head]-(double)G[j*HV+head])*dot;
                z-=lij*a[j*C+col];
            }
            a[i*C+col]=z;
        }
    }
    return a;
}

static Ref plan(const std::vector<uint16_t>& q,
                const std::vector<uint16_t>& k,
                const std::vector<uint16_t>& v,
                const std::vector<float>& G,
                const std::vector<float>& beta,
                const std::vector<double>& a) {
    Ref r{std::vector<double>(T*HD),std::vector<double>(HD*HD,0.0)};
    std::vector<double> w(T*HD),dlt(T*HD);
    for(int i=0;i<T;++i) for(int vv=0;vv<HD;++vv)
        w[i*HD+vv]=(double)beta[i*HV]*from_f16(v[(i*HV)*HD+vv]);
    for(int i=0;i<T;++i) for(int vv=0;vv<HD;++vv) {
        double z=0; for(int j=0;j<=i;++j) z+=a[i*C+j]*w[j*HD+vv];
        dlt[i*HD+vv]=z;
    }
    for(int i=0;i<T;++i) for(int vv=0;vv<HD;++vv) {
        double y=0;
        for(int j=0;j<=i;++j) {
            double qk=0;
            for(int x=0;x<HD;++x)
                qk+=from_f16(q[(i*HK)*HD+x])*from_f16(k[(j*HK)*HD+x]);
            y+=std::exp((double)G[i*HV]-(double)G[j*HV])*qk*dlt[j*HD+vv];
        }
        r.out[i*HD+vv]=y;
    }
    for(int vv=0;vv<HD;++vv) for(int x=0;x<HD;++x) {
        double z=0;
        for(int j=0;j<T;++j)
            z+=std::exp((double)G[(T-1)*HV]-(double)G[j*HV])*
               from_f16(k[(j*HK)*HD+x])*dlt[j*HD+vv];
        r.state[vv*HD+x]=z;
    }
    return r;
}

int main() {
    std::mt19937 rng(20260919);
    std::normal_distribution<float> normal(0.0f,1.0f);
    std::uniform_real_distribution<float> uv(-0.08f,0.08f), ub(0.15f,0.85f),
        ug(-0.07f,-0.015f);
    std::vector<uint16_t> q((size_t)T*HK*HD),k(q.size()),v((size_t)T*HV*HD);
    std::vector<float> G((size_t)T*HV),beta((size_t)T*HV);
    for(int t=0;t<T;++t) for(int kh=0;kh<HK;++kh) {
        std::vector<float> qrow(HD),krow(HD); double q2=0,k2=0;
        for(int d=0;d<HD;++d) { qrow[d]=normal(rng); krow[d]=normal(rng);
            q2+=(double)qrow[d]*qrow[d]; k2+=(double)krow[d]*krow[d]; }
        for(int d=0;d<HD;++d) {
            q[(t*HK+kh)*HD+d]=to_f16(qrow[d]/std::sqrt(q2)/std::sqrt((double)HD));
            k[(t*HK+kh)*HD+d]=to_f16(krow[d]/std::sqrt(k2));
        }
    }
    for(int h=0;h<HV;++h) {
        float cumulative=0;
        for(int t=0;t<T;++t) { cumulative+=ug(rng); G[t*HV+h]=cumulative;
            beta[t*HV+h]=ub(rng); for(int d=0;d<HD;++d)
                v[(t*HV+h)*HD+d]=to_f16(uv(rng)); }
    }

    auto exact=serial(q,k,v,G,beta,true);
    auto no_decay_prediction=serial(q,k,v,G,beta,false);
    auto a0=inverse_for_head(k,G,beta,0);
    auto algebra=plan(q,k,v,G,beta,a0);
    metric_dd("plan_vs_serial_output",algebra.out,exact.out);
    metric_dd("plan_vs_serial_state",algebra.state,exact.state);
    metric_dd("no_decay_prediction_vs_serial_output",no_decay_prediction.out,exact.out);
    metric_dd("no_decay_prediction_vs_serial_state",no_decay_prediction.state,exact.state);

    std::vector<uint16_t> A((size_t)T*HV*C);
    for(int h=0;h<HV;++h) {
        auto ah=inverse_for_head(k,G,beta,h);
        for(int i=0;i<T;++i) for(int j=0;j<C;++j)
            A[((size_t)i*HV+h)*C+j]=to_f16((float)ah[i*C+j]);
    }
    std::vector<float> h0((size_t)HV*HD*HD,0.0f),ht(h0.size()),out((size_t)T*HV*HD);
    uint16_t *qd,*kd,*vd,*Ad; float *Gd,*bd,*h0d,*htd,*outd;
    HIP_OK(hipMalloc(&qd,q.size()*2)); HIP_OK(hipMalloc(&kd,k.size()*2));
    HIP_OK(hipMalloc(&vd,v.size()*2)); HIP_OK(hipMalloc(&Ad,A.size()*2));
    HIP_OK(hipMalloc(&Gd,G.size()*4)); HIP_OK(hipMalloc(&bd,beta.size()*4));
    HIP_OK(hipMalloc(&h0d,h0.size()*4)); HIP_OK(hipMalloc(&htd,ht.size()*4));
    HIP_OK(hipMalloc(&outd,out.size()*4));
    HIP_OK(hipMemcpy(qd,q.data(),q.size()*2,hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(kd,k.data(),k.size()*2,hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(vd,v.data(),v.size()*2,hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(Ad,A.data(),A.size()*2,hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(Gd,G.data(),G.size()*4,hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(bd,beta.data(),beta.size()*4,hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(h0d,h0.data(),h0.size()*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(gdn_chunk_scan_bf16_f32_edge_gfx1201,dim3(2,48),dim3(256),0,0,
        qd,kd,vd,Ad,Gd,bd,h0d,htd,outd,0,T);
    HIP_OK(hipGetLastError()); HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(ht.data(),htd,ht.size()*4,hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(out.data(),outd,out.size()*4,hipMemcpyDeviceToHost));
    std::vector<float> headout(T*HD),headstate(HD*HD);
    for(int t=0;t<T;++t) for(int vv=0;vv<HD;++vv)
        headout[t*HD+vv]=out[((size_t)t*HV)*HD+vv];
    for(int vv=0;vv<HD;++vv) for(int d=0;d<HD;++d)
        headstate[vv*HD+d]=ht[vv*HD+d];
    metric_fd("kernel_vs_serial_output",headout,exact.out);
    metric_fd("kernel_vs_serial_state",headstate,exact.state);
    metric_fd("kernel_vs_plan_output",headout,algebra.out);
    metric_fd("kernel_vs_plan_state",headstate,algebra.state);
    double worst_t0=0.0; int worst_head=-1;
    for(int h=0;h<HV;++h) {
        double qk=0.0, e2=0.0, r2=0.0;
        for(int x=0;x<HD;++x)
            qk+=from_f16(q[(h/3)*HD+x])*from_f16(k[(h/3)*HD+x]);
        for(int vv=0;vv<HD;++vv) {
            const double ref=(double)beta[h]*from_f16(v[h*HD+vv])*qk;
            const double err=(double)out[h*HD+vv]-ref;
            e2+=err*err; r2+=ref*ref;
        }
        const double nr=std::sqrt(e2/r2);
        if(nr>worst_t0) { worst_t0=nr; worst_head=h; }
    }
    std::printf("kernel_t0_all_heads_worst_nrmse=%.12g head=%d\n",worst_t0,worst_head);
    std::printf("serial recurrence: delta=beta*(v-alpha*(S^T*k)); S=alpha*S+k*delta; y=S^T*q\n");
    return 0;
}
