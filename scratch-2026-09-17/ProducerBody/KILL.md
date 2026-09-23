KILL: gfx1201 four-candidate producer quant bundle

base bc1a55de9; device GPU-6109a4cb5f833235; HOME ab3

KLD (same bundled eval binary):
- c2 GDN OFF 0.060029
- c2 GDN ON  0.059423 (ceiling 0.060360: pass)
- c24 GDN OFF 0.081170
- c24 GDN ON  0.080450 (ceiling 0.078514: fail by 0.001936)
- c24 ON - OFF = -0.000720 (relative GDN gate passes)

The shared absolute c24 quality ceiling failed, so the abandon rule applies. No paired performance or TTFT was run after the quality failure. Source restored to updated baseline; no commit.
