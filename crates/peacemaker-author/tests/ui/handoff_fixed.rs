// The handoff as the gate/up epilogue uses it: writers publish and leave,
// readers read and store up to the kernel exit.
use peacemaker_author::{trace::Trace, Backend, Gfx1201, Workgroup};

enum Gate {}

fn epilogue<B: Backend<Insn = String>>(wg: &mut Workgroup<Gfx1201, B>) -> Result<(), String> {
    let gate = wg.lds::<Gate>("gate", 0, 2048)?;
    let end = wg.exit(".Lend")?;
    let readers = wg.scmp("s_cmp_ge_u32 s4, 2".into())?;
    wg.handoff(readers, ".Lup", end, gate, |w, gate| {
        let out = w.ds_store(gate, "ds_store_b128 v1, v[2:5]".into())?;
        w.ds_store(out, "ds_store_b128 v1, v[6:9] offset:16".into())
    }, |w, gate, _| {
        w.ds_load(&gate, "ds_load_b128 v[8:11], v1".into())?;
        w.ds_load(&gate, "ds_load_b128 v[12:15], v1 offset:16".into())
    })
}

fn main() {
    let mut b = Trace::new("gfx1201");
    epilogue(&mut Workgroup::<Gfx1201, Trace>::new(&mut b).unwrap()).unwrap();
    b.finish().unwrap();
    assert_eq!((b.count("s_barrier_signal"), b.count("s_barrier_wait"), b.count("s_wait_dscnt")), (2, 2, 1));
    assert_eq!(b.text.last().map(String::as_str), Some("s_endpgm"));
}
