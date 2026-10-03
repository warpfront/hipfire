/* hipfire chat UI — chat history store.
 *
 * IndexedDB, built into every browser: one object store of conversation
 * records keyed by id. History lives in this browser profile only; serve
 * never sees it, so nothing new is exposed on the unauthenticated listener.
 */
"use strict";

const ChatDB = (() => {
  const NAME = "hipfire-chat";
  const VERSION = 1;
  const STORE = "convs";
  let opening = null;

  function open() {
    if (!opening) {
      opening = new Promise((resolve, reject) => {
        if (!("indexedDB" in window)) {
          reject(new Error("IndexedDB is unavailable"));
          return;
        }
        const req = indexedDB.open(NAME, VERSION);
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(STORE)) {
            db.createObjectStore(STORE, { keyPath: "id" }).createIndex("updated", "updated");
          }
        };
        req.onsuccess = () => {
          const db = req.result;
          // A newer tab upgrading the schema must not be blocked by this one.
          db.onversionchange = () => { db.close(); opening = null; };
          resolve(db);
        };
        req.onerror = () => reject(req.error);
        req.onblocked = () => reject(new Error("history database is blocked by another tab"));
      });
      opening.catch(() => { opening = null; });
    }
    return opening;
  }

  async function run(mode, fn) {
    const db = await open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const req = fn(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(req ? req.result : undefined);
      tx.onabort = tx.onerror = () => reject(tx.error || new Error("transaction aborted"));
    });
  }

  return {
    open,
    all: () => run("readonly", (s) => s.getAll()),
    get: (id) => run("readonly", (s) => s.get(id)),
    put: (conv) => run("readwrite", (s) => s.put(conv)),
    putMany: (convs) => run("readwrite", (s) => { for (const c of convs) s.put(c); }),
    remove: (id) => run("readwrite", (s) => s.delete(id)),
    clear: () => run("readwrite", (s) => s.clear()),
  };
})();
