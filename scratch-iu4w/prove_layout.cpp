#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static size_t packed_dst(size_t row, size_t nibble, size_t k) {
  const size_t lane = 16 * ((nibble >> 4) & 1) + (row & 15);
  return ((((row >> 4) * (k >> 6) + (nibble >> 6)) * 32 + lane) * 16)
       + ((nibble >> 5) & 1) * 8 + ((nibble & 15) >> 1);
}
static size_t header_dst(size_t row, size_t half, size_t plane,
                         size_t m, size_t k) {
  return m * k / 2 + (((half * (m >> 4) + (row >> 4)) * 4 + plane) * 16)
       + (row & 15);
}
static bool prove(const char* name, size_t m, size_t k) {
  const size_t gpr = k / 256, bytes = m * gpr * 136;
  std::vector<uint8_t> src(bytes), dst(bytes, 0), inv(bytes, 0), seen(bytes, 0);
  for (size_t i=0;i<bytes;i++) src[i]=(uint8_t)((i*1315423911u + i/97u) >> 9);
  bool bij=true;
  for (size_t row=0;row<m;row++) {
    for (size_t pc=0;pc<k/2;pc++) {
      size_t nib=pc*2;
      size_t s=(row*gpr+nib/256)*136+8+(nib&255)/2;
      size_t d=packed_dst(row,nib,k);
      bij &= d<bytes && !seen[d]; seen[d]=1; dst[d]=src[s]; inv[s]=dst[d];
    }
    for (size_t half=0;half<k/128;half++) {
      size_t s=(row*gpr+half/2)*136+(half&1)*4;
      const size_t planes[4]={1,0,2,3};
      for (size_t b=0;b<4;b++) {
        size_t d=header_dst(row,half,planes[b],m,k);
        bij &= d<bytes && !seen[d]; seen[d]=1; dst[d]=src[s+b]; inv[s+b]=dst[d];
      }
    }
  }
  bij &= std::all_of(seen.begin(),seen.end(),[](uint8_t x){return x==1;});
  bool exact = inv == src;
  std::printf("PROOF,%s,M=%zu,K=%zu,bytes=%zu,padded128=%zu,bij=%s,rowmajor_recovered=%s\n",
              name,m,k,bytes,((m+127)/128)*128,bij?"PASS":"FAIL",exact?"PASS":"FAIL");
  return bij && exact;
}
int main() {
  bool ok=true;
  ok &= prove("gate",17408,5120);
  ok &= prove("up",17408,5120);
  ok &= prove("down",5120,17408);
  ok &= prove("qkvza_tail96",16480,5120);
  ok &= prove("qkv",14336,5120);
  return ok?0:1;
}
