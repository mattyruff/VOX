/* =====================================================================
   VOX LISTENING ENGINE — tuned for the singing voice.
   (Versioned with the app: see the vox-version meta in index.html.)

   A self-contained, DOM-free vocal-analysis engine. It owns the
   microphone + Web Audio graph and runs the DSP pipeline:

     mic -> input boost -> high-pass clean -> VAD -> tone segmentation ->
     F0 tracking (normalized autocorrelation) -> amplitude/breath envelope ->
     spectral features (centroid, harmonic richness, timbre) ->
     formant estimation (F1/F2) -> jitter / shimmer / HNR ->
     vocal coherence score.

   It exposes a live `state` object (read each frame by the UI), an
   accumulated `session` log, and onFrame / onToneEnd subscriptions.
   It knows nothing about the DOM, so it can back any interface.

   Usage:
     const engine = new Vox.Engine({ tuneA: 432 });
     engine.onFrame(() => render(engine.state));
     const ok = await engine.start();   // prompts for mic
     ...
     engine.stop();

   Ships as a classic script that attaches a `Vox` global (works over
   file:// and static hosting with no build step). Structured so it can
   be turned into an ES module later.
   ===================================================================== */
(function (global) {
"use strict";

const clamp = (v, a, b) => (v < a ? a : (v > b ? b : v));
const NOTES = ['C','C♯','D','D♯','E','F','F♯','G','G♯','A','A♯','B'];
const midiName = m => NOTES[m % 12] + (Math.floor(m / 12) - 1);

// vowel formant map for the resonance readout
const VOWELS = [['ee',280,2250],['ih',400,1900],['eh',550,1770],['ah',710,1150],['aw',590,880],['oh',450,800],['oo',310,880]];
function vowelName(f1, f2){
  if(!f1||!f2) return '—';
  let best='—',bd=1e9;
  for(const [n,a,b] of VOWELS){
    const d=Math.pow(Math.log(f1/a),2)+Math.pow(Math.log(f2/b),2);
    if(d<bd){bd=d;best=n;}
  }
  return best;
}

const ACW = 2048;   // autocorrelation window

// voice-gate tuning — all levels are RMS measured after the input boost
const ANALYSIS_MIN = 0.0015;  // below this a frame is true silence; skip the pitch work
const GATE_MIN     = 0.004;   // the quietest voice we accept outright (quiet desktop mics)
const FLOOR_MULT   = 2.2;     // gate = noise floor x this; periodicity does the real work
const FLOOR_MAX    = 0.008;   // hard cap so noise can never run the gate away

class VoxEngine {
  constructor(opts){
    opts = opts || {};
    this.tuneA = opts.tuneA === 440 ? 440 : 432;   // concert pitch A4
    // input boost: desktop/headset mics are often far quieter than phone mics,
    // leaving real voices under the detection gate. Boosting is analysis-only
    // (nothing is played back) and the noise gate adapts, so hot mics stay fine.
    this.inputGain = (+opts.inputGain > 0) ? +opts.inputGain : 4;

    // ---- audio graph ----
    this.mode = 'idle';           // 'idle' | 'mic'
    this.SR = 48000;
    this.ctx = null; this.analyser = null; this.hpf = null; this.gainNode = null;
    this.srcNode = null; this.micStream = null;
    this.td = null; this.fd = null; this.binHz = 0;

    // ---- DSP working state ----
    this.linBuf = null;
    this.f0Med = [];                       // median-of-5 anti-octave-glitch
    this.pHist = new Float32Array(72);     // ~1.2s of cents history for stability
    this.pHistN = 0; this.pHistI = 0;
    this.tone = null;                      // current tone being segmented

    // ---- live feature state (read by the UI each frame) ----
    this.state = {
      lvl:0, floor:1e-4,                 // rms level + adaptive noise floor
      voiced:false, vadHold:0,           // voice activity
      f0:0, clarity:0,                   // pitch + periodicity confidence
      midiC:0,                           // continuous midi (fractional)
      stab:99,                           // rolling pitch std-dev, cents
      jitter:0, shimmer:0, hnr:0,        // voice quality
      centroid:0, rich:0, nHarm:0,       // brightness + harmonic richness
      f1:0, f2:0, vowel:'—',             // resonance mapping
      breath:0,                          // current tone duration, s
      coh:0, cohShow:0,                  // vocal coherence score
      prevF0:0, prevRms:0,
    };

    // ---- session log ----
    this.session = { start:0, tones:[] };

    // ---- subscriptions + loop ----
    this._frameCbs = [];
    this._toneEndCbs = [];
    this._looping = false; this._loopTimer = null;
    this._mobile = (typeof matchMedia !== 'undefined' && matchMedia('(pointer:coarse)').matches) ||
      (navigator.maxTouchPoints > 0 && Math.min(screen.width || 1024, screen.height || 768) <= 820);
  }

  // ---------- public accessors ----------
  get running(){ return this.mode === 'mic'; }
  get audioContext(){ return this.ctx; }
  get sampleRate(){ return this.SR; }

  midiToHz(m){ return this.tuneA * Math.pow(2, (m - 69) / 12); }
  setTuneA(hz){ this.tuneA = hz === 440 ? 440 : 432; }
  // target note for pitch-gated coherence: null = no target (free toning, score is
  // pure tone quality). Octave-folded, so the target counts in any octave.
  setTarget(m){ this.targetMidi = (m == null ? null : +m); }
  setInputGain(v){
    this.inputGain = (+v > 0) ? +v : this.inputGain;
    if(this.gainNode) this.gainNode.gain.value = this.inputGain;
  }

  onFrame(cb){ this._frameCbs.push(cb); return () => { const i=this._frameCbs.indexOf(cb); if(i>=0) this._frameCbs.splice(i,1); }; }
  onToneEnd(cb){ this._toneEndCbs.push(cb); return () => { const i=this._toneEndCbs.indexOf(cb); if(i>=0) this._toneEndCbs.splice(i,1); }; }
  _emitFrame(){ for(const cb of this._frameCbs) cb(this.state); }
  _emitToneEnd(rec){ for(const cb of this._toneEndCbs) cb(rec); }

  // ---------- audio context (shared with the caller's drone/chime) ----------
  ensureContext(){
    if(this.ctx) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.SR = this.ctx.sampleRate;
    // signal cleaning: high-pass removes room rumble & handling noise below the voice
    this.gainNode = this.ctx.createGain();
    this.gainNode.gain.value = this.inputGain;
    this.hpf = this.ctx.createBiquadFilter();
    this.hpf.type='highpass'; this.hpf.frequency.value=70; this.hpf.Q.value=0.71;
    this.gainNode.connect(this.hpf);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize=4096; this.analyser.smoothingTimeConstant=0.5;
    this.hpf.connect(this.analyser);
    this.td = new Float32Array(this.analyser.fftSize);
    this.fd = new Float32Array(this.analyser.frequencyBinCount);
    this.binHz = this.SR / this.analyser.fftSize;
  }

  // ---------- mic lifecycle ----------
  async start(){
    if(this.mode === 'mic') return true;
    if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return false;
    // create + start resuming the context while the click gesture is still live —
    // resuming only AFTER the permission prompt can hang forever on iOS Safari,
    // because the prompt consumes the user activation the resume needs
    this.ensureContext();
    const resuming = this.ctx.resume().catch(()=>{});
    let stream;
    try{
      stream = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
    }catch(e){ return false; }
    await resuming;
    this.micStream = stream;
    this.mode = 'mic';
    this.srcNode = this.ctx.createMediaStreamSource(stream);
    this.srcNode.connect(this.gainNode);
    if(!this.session.start) this.session.start = Date.now();
    this._startLoop();
    return true;
  }
  stop(){
    if(this.srcNode){ try{this.srcNode.disconnect();}catch(e){} this.srcNode=null; }
    if(this.micStream){ this.micStream.getTracks().forEach(t=>t.stop()); this.micStream=null; }
    this.mode='idle';
    this.endTone();
    // halt the analysis loop and clear voice-activity state, so a stopped engine
    // costs nothing and a restart can't inherit a phantom "voiced" moment
    if(this._loopTimer){ clearTimeout(this._loopTimer); this._loopTimer=null; }
    this._looping = false;
    this.state.voiced = false; this.state.vadHold = 0; this.state.breath = 0;
  }

  _startLoop(){
    if(this._looping) return;
    this._looping = true;
    const tick = () => {
      this.analyze();
      if(this._looping)                    // stop() may have been called mid-frame
        this._loopTimer = setTimeout(tick, document.hidden ? 250 : (this._mobile ? 33 : 1000/60));
    };
    tick();
  }

  // ---------- F0: normalized autocorrelation with octave guard ----------
  trackPitch(){
    const SR=this.SR, td=this.td;
    const minLag=Math.floor(SR/800), maxLag=Math.min(Math.floor(SR/70), td.length-ACW-1);
    let e0=0; for(let i=0;i<ACW;i++) e0+=td[i]*td[i];
    if(e0<1e-7) return {f0:0,r:0};
    // sliding window energy
    let eW=0; for(let i=minLag;i<minLag+ACW;i++) eW+=td[i]*td[i];
    let bestR=0, bestLag=0;
    const rArr=new Float32Array(maxLag+1);
    for(let lag=minLag;lag<=maxLag;lag++){
      let c=0;
      for(let i=0;i<ACW;i++) c+=td[i]*td[i+lag];
      const r=c/Math.sqrt(e0*eW+1e-12);
      rArr[lag]=r;
      if(r>bestR){ bestR=r; bestLag=lag; }
      eW+=td[lag+ACW]*td[lag+ACW]-td[lag]*td[lag];
    }
    if(bestR<0.3) return {f0:0,r:bestR};
    // octave guard: take the FIRST local peak that gets within 88% of the global max
    let lag=bestLag;
    for(let L=minLag+1;L<maxLag;L++){
      if(rArr[L]>=rArr[L-1]&&rArr[L]>=rArr[L+1]&&rArr[L]>=bestR*0.88){ lag=L; break; }
    }
    // parabolic refinement
    const a=rArr[lag-1]||0,b=rArr[lag],c2=rArr[lag+1]||0;
    const den=a-2*b+c2;
    const sh=den?clamp(0.5*(a-c2)/den,-0.5,0.5):0;
    return {f0:SR/(lag+sh), r:rArr[lag]};
  }

  // ---------- tone segmentation ----------
  beginTone(){
    this.tone={t0:performance.now(), n:0, f0s:[], coh:0, stab:0, hnr:0, jit:0, shim:0, rich:0, rms:0};
  }
  endTone(){
    const tone=this.tone;
    if(!tone) return;
    const dur=(performance.now()-tone.t0)/1000;
    if(dur>=1.0 && tone.n>=20){
      const f0s=tone.f0s.slice().sort((x,y)=>x-y);
      const f0m=f0s[f0s.length>>1];
      const midi=Math.round(69+12*Math.log2(f0m/440));
      const rec={
        dur, f0:f0m, note:midiName(midi),
        coh:tone.coh/tone.n, stab:tone.stab/tone.n, hnr:tone.hnr/tone.n,
        jit:tone.jit/tone.n, shim:tone.shim/tone.n, rich:tone.rich/tone.n,
      };
      this.session.tones.push(rec);
      this._emitToneEnd(rec);
    }
    this.tone=null;
  }

  // dB spectrum -> linear magnitude
  specLin(){
    const fd=this.fd;
    if(!this.linBuf) this.linBuf=new Float32Array(fd.length);
    const linBuf=this.linBuf;
    for(let i=0;i<fd.length;i++) linBuf[i]=Math.pow(10,fd[i]/20);
    return linBuf;
  }
  // F1/F2: strongest peaks in the classic formant bands (comb-sampled envelope)
  formants(lin,f0){
    const binHz=this.binHz;
    const find=(lo,hi)=>{
      let bi=-1,bv=0;
      const a=Math.max(2,(lo/binHz)|0), b=Math.min(lin.length-2,(hi/binHz)|0);
      for(let i=a;i<=b;i++){ const v=lin[i-1]+lin[i]*2+lin[i+1]; if(v>bv){bv=v;bi=i;} }
      return bi>0?bi*binHz:0;
    };
    const f1=find(Math.max(220,f0*0.8),1100);
    if(!f1) return null;
    const f2=find(Math.max(1250,f1+300),3200);
    return f2?[f1,f2]:[f1,0];
  }

  // ---------- per-frame analysis ----------
  analyze(){
    const eng=this.state;
    if(!this.analyser||this.mode!=='mic') return;
    const td=this.td, fd=this.fd, SR=this.SR, binHz=this.binHz, tuneA=this.tuneA;
    const f0Med=this.f0Med, pHist=this.pHist;
    this.analyser.getFloatTimeDomainData(td);
    this.analyser.getFloatFrequencyData(fd);

    // amplitude envelope
    let s=0; for(let i=0;i<td.length;i++) s+=td[i]*td[i];
    const rms=Math.sqrt(s/td.length);
    eng.lvl+=(rms-eng.lvl)*0.3;

    // pitch + periodicity — measured whenever there is anything at all to measure,
    // not merely above the gate, so the floor logic below can tell a voice from noise
    let p={f0:0,r:0};
    if(rms>ANALYSIS_MIN) p=this.trackPitch();
    const periodic=p.r>0.5;                       // a voice or a drone; never room noise

    // voice activity detection: energy above the adaptive floor AND periodic
    const gate=Math.max(eng.floor*FLOOR_MULT, GATE_MIN);
    const voicedNow=(rms>gate && p.r>0.60 && p.f0>=70 && p.f0<=800);
    if(voicedNow) eng.vadHold=14;                 // ~0.25s hangover
    else if(eng.vadHold>0) eng.vadHold--;
    const wasVoiced=eng.voiced;
    eng.voiced=eng.vadHold>0;
    // Learn the noise floor ONLY from aperiodic quiet. Adapting while the user was
    // voicing-but-undetected let the gate chase their own voice upward and lock them
    // out for good. Rises slowly, falls quickly, and is capped.
    if(!eng.voiced && !periodic){
      eng.floor+=(rms-eng.floor)*(rms>eng.floor?0.008:0.05);
      if(eng.floor>FLOOR_MAX) eng.floor=FLOOR_MAX;
    }

    // tone segmentation
    if(eng.voiced&&!wasVoiced) this.beginTone();
    if(!eng.voiced&&wasVoiced) this.endTone();
    const tone=this.tone;

    if(voicedNow){
      // median-of-5 F0 smoothing kills octave glitches
      f0Med.push(p.f0); if(f0Med.length>5) f0Med.shift();
      const srt=f0Med.slice().sort((a,b)=>a-b);
      const f0=srt[srt.length>>1];
      eng.clarity=p.r;

      // jitter: frame-to-frame period perturbation (RAP-like)
      if(eng.prevF0>0){
        const jRaw=Math.abs(1/f0-1/eng.prevF0)*f0;
        eng.jitter+=(clamp(jRaw,0,0.08)-eng.jitter)*0.06;
      }
      eng.prevF0=f0;
      // shimmer: frame-to-frame amplitude perturbation
      if(eng.prevRms>0){
        const shRaw=Math.abs(rms-eng.prevRms)/Math.max(rms,eng.prevRms);
        eng.shimmer+=(clamp(shRaw,0,0.6)-eng.shimmer)*0.06;
      }
      eng.prevRms=rms;
      // HNR from normalized ACF peak (Boersma)
      const r=clamp(p.r,0.01,0.999);
      const hnrRaw=10*Math.log10(r/(1-r));
      eng.hnr+=(clamp(hnrRaw,0,35)-eng.hnr)*0.08;

      eng.f0=f0;
      eng.midiC=69+12*Math.log2(f0/tuneA);
      // pitch stability: rolling std-dev in cents over ~1.2s
      pHist[this.pHistI]=eng.midiC*100; this.pHistI=(this.pHistI+1)%pHist.length; if(this.pHistN<pHist.length)this.pHistN++;
      if(this.pHistN>18){
        let m=0; for(let i=0;i<this.pHistN;i++)m+=pHist[i]; m/=this.pHistN;
        let v=0; for(let i=0;i<this.pHistN;i++){const d=pHist[i]-m;v+=d*d;}
        eng.stab+=(Math.min(99,Math.sqrt(v/this.pHistN))-eng.stab)*0.15;
      }

      // ---- spectral features ----
      const lin=this.specLin();
      // brightness: spectral centroid 80–6000 Hz
      let num=0,den=0;
      const i0=Math.max(1,(80/binHz)|0), i1=Math.min(lin.length,(6000/binHz)|0);
      for(let i=i0;i<i1;i++){ num+=i*binHz*lin[i]; den+=lin[i]; }
      if(den>0) eng.centroid+=((num/den)-eng.centroid)*0.12;
      // harmonic richness + timbre profile: overtone energy relative to the fundamental
      let h1=0, hs=0, nH=0;
      for(let k=1;k<=10;k++){
        const f=f0*k; if(f>SR/2-200) break;
        const b=Math.round(f/binHz);
        let a=0; for(let d=-1;d<=1;d++) a=Math.max(a,lin[b+d]||0);
        if(k===1) h1=a; else { hs+=a; if(h1>0&&a>h1*0.02) nH++; }
      }
      const richRaw=h1>0?clamp(hs/h1/1.6,0,1):0;
      eng.rich+=(richRaw-eng.rich)*0.1;
      eng.nHarm=nH;
      // formants / resonance mapping: strongest spectral peaks under the vocal envelope
      const fm=this.formants(lin,f0);
      if(fm){ eng.f1+=(fm[0]-eng.f1)*0.15; eng.f2+=(fm[1]-eng.f2)*0.15; eng.vowel=vowelName(eng.f1,eng.f2); }

      // breath: current tone length
      if(tone) eng.breath=(performance.now()-tone.t0)/1000;

      // ---- vocal coherence score ----
      const sStab =clamp(1-(eng.stab-3)/45,0,1);            // 3¢ steady -> 48¢ scattered
      const sClar =clamp((eng.hnr-5)/17,0,1);               // 5 dB -> 22 dB
      const sSmooth=clamp(1-(eng.jitter/0.012+eng.shimmer/0.10)/2,0,1);
      const sRich =clamp(eng.rich/0.6,0,1);
      const sBreath=clamp(eng.breath/8,0,1);
      // pitch accuracy gates the whole score when a target note is set: a steady,
      // clear tone on the WRONG note must not read as coherent. Octave-folded;
      // full credit within ±35¢, sliding to a 0.25 floor at 150¢ off.
      let sPitch=1;
      if(this.targetMidi!=null){
        let dev=(eng.midiC-this.targetMidi)*100;
        dev-=Math.round(dev/1200)*1200;
        const a=Math.abs(dev);
        sPitch=a<=35?1:(a>=150?0.25:1-(a-35)/(150-35)*0.75);
      }
      eng.pitch01=sPitch;                                   // exposed for UI cues
      eng.coh=100*(0.30*sStab+0.25*sClar+0.20*sSmooth+0.15*sRich+0.10*sBreath)*sPitch;
      eng.cohShow+=(eng.coh-eng.cohShow)*0.08;

      if(tone){
        tone.n++; tone.f0s.push(f0);
        tone.coh+=eng.coh; tone.stab+=eng.stab; tone.hnr+=eng.hnr;
        tone.jit+=eng.jitter; tone.shim+=eng.shimmer; tone.rich+=eng.rich;
      }
    } else {
      eng.prevF0=0; eng.prevRms=0; f0Med.length=0;
      if(!eng.voiced){ eng.cohShow*=0.985; eng.breath=0; this.pHistN=0; this.pHistI=0; }
    }
    this._emitFrame();
  }
}

global.Vox = { Engine: VoxEngine, NOTES, midiName };

if (typeof module !== 'undefined' && module.exports) module.exports = global.Vox;

})(typeof window !== 'undefined' ? window : this);
