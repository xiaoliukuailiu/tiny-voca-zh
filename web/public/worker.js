// Runs the phonemizer (pypinyin, through Pyodide) and the model (ONNX Runtime Web)
// off the main thread. Messages in: {type: "load"} and {type: "speak", id, text,
// voice, speed}. Messages out: status, ready, sentence (with its audio, as soon as
// it is made), done (with the whole clip), error (plain: true when the message is
// meant for the reader as it is).

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

importScripts(PYODIDE + "pyodide.js", ORT + "ort.wasm.min.js");

let pyodide, phonemize, session, config;

const status = (text, progress) => postMessage({ type: "status", text, progress });

// Bump when the files under assets/ change, so browsers fetch them again.
const VERSION = "3";

// Downloaded files are kept in the browser's cache storage: a second visit starts without the network.
async function download(url, label) {
  const key = `${url}?v=${VERSION}`;
  let store = null;
  try {
    store = await caches.open("tiny-voca-" + VERSION);
    const hit = await store.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (error) { store = null; }
  const bytes = await fetchBytes(key, label);
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function loadPhonemizer() {
  status("正在加载音素化器");
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  const source = await download("assets/zh-frontend.zip", "正在加载音素化器");
  status("正在启动音素化器");
  pyodide.unpackArchive(source.buffer, "zip", { extractDir: "/zh-src" });
  pyodide.runPython(`
import json, sys
sys.path.insert(0, "/zh-src")
import zh_text

def phonemize_json(text):
    try:
        result = zh_text.phonemize(text)
    except zh_text.NothingToPronounce:
        # Nothing to pronounce (another script, a face, signs only) is no failure:
        # "null", and the sentence is skipped.
        return "null"
    return json.dumps({"ids": list(result.ids), "stress": list(result.stress),
                       "text": " ".join(result.symbols)})

phonemize_json("你好。")
`);
  phonemize = pyodide.globals.get("phonemize_json");
}

async function loadModel() {
  config = await (await fetch("assets/config.json")).json();
  const bytes = await download("assets/model.onnx", "正在加载模型");
  status("正在启动模型");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

// Sentences end right after 。！？… (Mandarin has no spaces and no capitals).
// tiny_voca.infer.sentences in the Python package splits the same way; keep the two alike.
function sentences(text) {
  return text.replace(/\s+/g, " ").trim().split(/(?<=[。！？…!?])/)
    .map((part) => part.trim())
    .filter((part) => /[\p{L}\p{N}]/u.test(part));
}

// A full stop is added when the text ends without a mark: before a final mark the voice lengthens the last
// syllable and lets the sound fall, without one the audio breaks off. tiny_voca.infer.finished does the
// same; keep the two alike.
function finished(text) {
  const body = text.replace(/\s+$/, "");
  if (!body) return text;
  const marked = ".!?…,;:。，；：、".includes(body.slice(-1));
  return marked ? text : body + "。";
}

// One utterance holds at most max_tokens phonemes; longer sentences are cut after commas and the like,
// then in half. A sentence with nothing to pronounce gives no utterance.
function fit(sentence) {
  const result = JSON.parse(phonemize(sentence));
  if (!result) return [];
  if (result.ids.length <= config.model.max_tokens - 12) return [result];
  const pieces = sentence.split(/(?<=[，,；;：:、—])/).filter((piece) => piece.trim());
  if (pieces.length > 1) return pieces.flatMap(fit);
  if (sentence.length < 8) return [];
  const half = Math.floor(sentence.length / 2);
  return [...fit(sentence.slice(0, half)), ...fit(sentence.slice(half))];
}

async function synthesize(phones, voice, speed) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids), stress: integers(phones.stress),
    voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voice)]), [1]),
    speed: scalar(speed), pitch_shift: scalar(0), pitch_range: scalar(1),
  });
  return output.audio.data;
}

// The model sometimes peaks above 1.0, which would clip in the 16-bit file. Scale the whole clip down then.
const PEAK = 0.99;
function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

// Shown when no sentence of a text could be read. tiny_voca.infer.speak_text says the same.
const NOTHING_READABLE = "没有可读的内容：模型只能朗读中文文本。";

async function speak({ id, text, voice, speed }) {
  const started = performance.now();
  const list = sentences(finished(text));
  const chunks = [];
  let phonemeTime = 0;
  let modelTime = 0;
  for (let index = 0; index < list.length; index++) {
    let mark = performance.now();
    const utterances = fit(list[index]);
    phonemeTime += performance.now() - mark;
    for (const phones of utterances) {
      mark = performance.now();
      const audio = await synthesize(phones, voice, speed);
      modelTime += performance.now() - mark;
      chunks.push(audio);
      // The page plays this sentence at once, before the whole clip is known: a copy, kept below full scale by itself.
      const heard = audio.slice();
      limitPeak(heard);
      postMessage({ type: "sentence", id, index, total: list.length, phonemes: phones.text, audio: heard,
                    sampleRate: config.sample_rate }, [heard.buffer]);
    }
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: NOTHING_READABLE, plain: true });
    return;
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  limitPeak(audio);
  postMessage({ type: "done", id, audio, sampleRate: config.sample_rate, phonemeTime, modelTime,
                totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await Promise.all([loadPhonemizer(), loadModel()]);
      postMessage({ type: "ready", voices: config.voices, sampleRate: config.sample_rate });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error) });
  }
};
