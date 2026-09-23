#!/usr/bin/env python3
import subprocess, sys, threading
def pump(src, dst):
    try:
        while True:
            c = src.read(65536)
            if not c: break
            dst.write(c); dst.flush()
    except Exception: pass
    finally:
        try: dst.close()
        except Exception: pass
p = subprocess.Popen(sys.argv[1:], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
t1 = threading.Thread(target=pump, args=(sys.stdin.buffer, p.stdin), daemon=True)
t2 = threading.Thread(target=pump, args=(p.stdout, sys.stdout.buffer), daemon=True)
t1.start(); t2.start(); sys.exit(p.wait())
