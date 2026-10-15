// Load the embedded-V8 bundle the way the packages do -- in a bare node:vm
// context, with no TextEncoder/TextDecoder/atob/console of its own -- and run
// one pipe request through each transport.
//
//   node rust/wasm-v8/smoke.mjs <bundle dir> <audio file>
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const [dir, audio] = process.argv.slice(2);
const ctx = vm.createContext({ WebAssembly });
for (const f of ['shim.js', 'praatfan.js']) {
  vm.runInContext(fs.readFileSync(path.join(dir, f), 'utf8'), ctx, { filename: f });
}
ctx.wasm_b64 = fs.readFileSync(path.join(dir, 'praatfan_bg.wasm')).toString('base64');
const t0 = Date.now();
vm.runInContext('wasm_bindgen.initSync({module: praatfan_b64bytes(wasm_b64)}); delete globalThis.wasm_b64;', ctx);
const tInit = Date.now() - t0;

const req = JSON.stringify({
  wav_path: path.basename(audio),
  analyses: [{ type: 'pitch_ac' }, { type: 'formant_burg' }, { type: 'intensity' },
             { type: 'harmonicity_cc' }, { type: 'spectral_moments', times: [0.5] }],
});
ctx.audio_b64 = fs.readFileSync(audio).toString('base64');
ctx.req = req;
const fromBytes = JSON.parse(vm.runInContext(
  'wasm_bindgen.pipe_request_bytes(praatfan_b64bytes(audio_b64), req)', ctx));
if (!fromBytes.ok) throw new Error('bytes transport: ' + fromBytes.error);

// The samples transport, fed the bytes transport's own decode (round-tripped
// through the typed-class API) must give the same answer.
const fromSamples = JSON.parse(vm.runInContext(`
  (function () {
    var s = wasm_bindgen.Sound.from_wav(praatfan_b64bytes(audio_b64));
    var r = JSON.parse(req); delete r.wav_path;
    return wasm_bindgen.pipe_request_samples(s.samples(), typeof s.sample_rate === 'function' ? s.sample_rate() : s.sample_rate, JSON.stringify(r));
  })()`, ctx));
if (JSON.stringify(fromSamples) !== JSON.stringify(fromBytes)) {
  throw new Error('samples transport differs from bytes transport');
}
console.log(`ok: praatfan ${vm.runInContext('wasm_bindgen.version()', ctx)}, init ${tInit} ms, ` +
            `${fromBytes.results.length} analyses, ${fromBytes.results[0].times.length} pitch frames`);
