'use strict';
const $ = (selector) => document.querySelector(selector);
const FALLBACK = {do_sample:true,temperature:.35,top_p:.85,top_k:30,repetition_penalty:1.1,no_repeat_ngram_size:4,max_new_tokens:128,seed:42,context_tokens:1024};
const STORAGE = 'clover.local-chat.v1';
// randomUUID requires HTTPS on public origins; getRandomValues also works on HTTP.
function newId() {
  if(typeof crypto.randomUUID==='function') return crypto.randomUUID();
  const bytes=crypto.getRandomValues(new Uint8Array(16));
  bytes[6]=(bytes[6]&15)|64;bytes[8]=(bytes[8]&63)|128;
  const hex=[...bytes].map(n=>n.toString(16).padStart(2,'0')).join('');
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
}
let defaults = {...FALLBACK}, settings = {...FALLBACK}, messages = [], active = null, ready = false;
let userScrolled = false, saveTimer = null, toastTimer = null;
let awaitingLanguageConfirm = false;
let fixedSeed = !$('#fixed-seed');

const compactLayout = window.matchMedia('(max-width:760px)');
function syncSettingsPanel() {
  const toggle=$('#settings-toggle');if(!toggle)return;
  const open=!compactLayout.matches||$('#settings-panel').classList.contains('is-expanded');
  toggle.setAttribute('aria-expanded',String(open));
  toggle.setAttribute('aria-label',open?'收起生成参数':'展开生成参数');
  $('#settings-toggle-label').textContent=open?'收起':'展开';
  $('#settings-toggle-icon').textContent=open?'−':'+';
}
function renderView(moveFocus=false) {
  const about=$('#about-view'),chat=$('#chat-view');if(!about||!chat)return;
  const isAbout=location.hash==='#about';
  about.hidden=!isAbout;chat.hidden=isAbout;
  document.body.classList.toggle('view-about',isAbout);
  document.querySelectorAll('[data-view]').forEach(link=>{
    if(link.dataset.view===(isAbout?'about':'chat'))link.setAttribute('aria-current','page');
    else link.removeAttribute('aria-current');
  });
  document.title=isAbout?'Clover v0.1.0 — About':'Clover v0.1.0 — English Chat';
  if(moveFocus)(isAbout?about:$('#chat-heading')).focus({preventScroll:true});
}
if($('#settings-toggle')) {
  $('#settings-toggle').addEventListener('click',()=>{$('#settings-panel').classList.toggle('is-expanded');syncSettingsPanel();});
  compactLayout.addEventListener('change',syncSettingsPanel);
  syncSettingsPanel();
}
window.addEventListener('hashchange',()=>renderView(true));
renderView();

function toast(text) {
  $('#toast').textContent=text; $('#toast').hidden=false;
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>{$('#toast').hidden=true;},2600);
}
function persistState() {
  try { localStorage.setItem(STORAGE,JSON.stringify({settings,fixedSeed,messages:messages.filter(m=>m.status!=='streaming'),draft:$('#prompt').value,system:$('#system-message').value,theme:document.body.classList.contains('rose')?'rose':'paper'})); }
  catch { /* Full/unavailable storage never prevents local inference. */ }
}
function save() {
  clearTimeout(saveTimer); saveTimer=setTimeout(persistState,180);
}
function notice(text,error=false) { $('#notice').textContent=text; $('#notice').hidden=!text; $('#notice').classList.toggle('error',error); }
function chooseTheme(theme) {
  document.body.classList.toggle('rose',theme==='rose');
  document.querySelectorAll('[data-theme]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.theme===theme)));
}
function paintSettings(preserveInput=null) {
  for(const [key,value] of Object.entries(settings)) {
    const input=$('#'+key); if(!input) continue;
    if(key==='do_sample') input.checked=value; else if(input!==preserveInput) input.value=value;
    const number=$(`[data-number="${key}"]`); if(number && number!==preserveInput) number.value=value;
    if(input.type==='range') input.style.setProperty('--range-fill',`${(Number(value)-Number(input.min))/(Number(input.max)-Number(input.min))*100}%`);
  }
  ['temperature','top_p','top_k'].forEach(key=>{
    const control=$(`[data-key="${key}"]`); control.classList.toggle('disabled',!settings.do_sample);
    control.querySelectorAll('input').forEach(i=>i.disabled=!settings.do_sample);
  });
  $('#sampling-caption').textContent=settings.do_sample?'开启：允许更自然的表达':'关闭：greedy，每步选最高概率词';
  if($('#fixed-seed'))$('#fixed-seed').checked=fixedSeed;
  $('#seed').disabled=!fixedSeed;
  if($('#seed-caption'))$('#seed-caption').textContent=fixedSeed?'使用指定值，便于复现':'每条新消息随机生成，发送时更新';
  $('#preset-name').textContent=Object.keys(defaults).every(k=>(k==='seed'&&!fixedSeed)||settings[k]===defaults[k])?'Balanced / 推荐':'Custom / 自定义';
}
function seedForMessage() {
  if(!fixedSeed) {
    let next;
    do {next=crypto.getRandomValues(new Uint32Array(1))[0]&0x7fffffff;} while(next===settings.seed);
    settings.seed=next;
    paintSettings();
  }
  return settings.seed;
}
function readNumber(input) {
  let value=Number(input.value); if(!Number.isFinite(value)) return null;
  value=Math.min(Number(input.max),Math.max(Number(input.min),value));
  const step=Number(input.step)||1,min=Number(input.min);
  value=Math.round((min+Math.round((value-min)/step)*step)*1e8)/1e8;
  return value;
}
function scrollBottom(force=false) {
  if(!messages.length) { $('#conversation').scrollTop=0; return; }
  if(force||!userScrolled) $('#conversation').scrollTop=$('#conversation').scrollHeight;
}
function renderMessages() {
  $('#welcome').hidden=messages.length>0;
  const container=$('#messages'); container.replaceChildren();
  for(const m of messages) {
    const article=document.createElement('article'); article.className=`message ${m.role} ${m.status||''}`; article.dataset.id=m.id;
    const header=document.createElement('div'); header.className='message-head';
    const avatar=document.createElement('span'); avatar.className='avatar';
    if(m.role==='user') {
      const icon=document.createElementNS('http://www.w3.org/2000/svg','svg');
      for(const [key,value] of Object.entries({class:'arrow-icon',viewBox:'0 0 16 16',fill:'none',stroke:'currentColor','stroke-width':'1.5','stroke-linecap':'square','aria-hidden':'true',focusable:'false'}))icon.setAttribute(key,value);
      const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d','M3 13 13 3M4 3h9v9');icon.append(path);avatar.append(icon);
    } else avatar.textContent='✳';
    const name=document.createElement('span'); name.className='message-role'; name.textContent=m.role==='user'?'YOU':'CLOVER';
    const meta=document.createElement('span'); meta.className='message-meta'; meta.textContent=m.meta||'';
    header.append(avatar,name,meta);
    if(m.role==='assistant' && m.content) {
      const copy=document.createElement('button'); copy.className='copy-button'; copy.textContent='复制'; copy.setAttribute('aria-label','复制这条回答');
      copy.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(m.content);toast('已复制回答。');}catch{toast('无法访问剪贴板，请选择文字复制。');}}); header.append(copy);
    }
    const body=document.createElement('div'); body.className='message-body'; body.textContent=m.content;
    if(!m.content && m.status==='error') {body.textContent='未生成回答。可重新发送这条消息。';body.classList.add('empty-answer');}
    article.append(header,body);container.append(article);
  }
  $('#export-chat').disabled=messages.length===0||!!active;
  $('#new-chat').disabled=messages.length===0||!!active;
  scrollBottom();
}
function updateBubble(m) {
  const article=document.querySelector(`[data-id="${m.id}"]`); if(!article) return;
  article.querySelector('.message-body').textContent=m.content;
  article.querySelector('.message-meta').textContent=m.meta||'';
  scrollBottom();
}
function busy(value) {
  $('#send').hidden=value; $('#stop').hidden=!value;
  $('#prompt').disabled=value; $('#new-chat').disabled=value||messages.length===0;$('#export-chat').disabled=value||messages.length===0;
  $('#status-dot').classList.toggle('busy',value);
  if(value) $('#generation-status').textContent='Waiting for the first token…';
}
function requestMessages(latestUser) {
  const history=[];
  if($('#system-message').value.trim()) history.push({role:'system',content:$('#system-message').value.trim()});
  // Failed/empty turns remain visible but are excluded as complete pairs.
  for(let i=0;i<messages.length-2;i++) {
    const u=messages[i],a=messages[i+1];
    if(u.role==='user' && a?.role==='assistant' && a.content.trim() && ['done','stopped'].includes(a.status)) {
      history.push({role:'user',content:u.content},{role:'assistant',content:a.content}); i++;
    }
  }
  history.push({role:'user',content:latestUser});return history;
}
function retryDelay(ms,signal) {
  return new Promise((resolve,reject)=>{
    const aborted=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'));};
    const timer=setTimeout(()=>{signal.removeEventListener('abort',aborted);resolve();},ms);
    if(signal.aborted)aborted();else signal.addEventListener('abort',aborted,{once:true});
  });
}
async function chatResponse(request,history,params,answer) {
  const retryable=new Set([409,429,502,503,504]);
  for(let attempt=0;attempt<=5;attempt++) {
    request.waiting=false;
    const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},signal:request.abort.signal,
      body:JSON.stringify({request_id:request.id,messages:history,parameters:params})});
    if(response.ok)return response;
    if(response.status===403) {
      request.reloading=true;
      if(messages.at(-1)===answer)messages.splice(-2,2);
      $('#prompt').value=history.at(-1).content;
      clearTimeout(saveTimer);persistState();
      window.location.reload();
      throw new DOMException('Refreshing page','AbortError');
    }
    let data={};try{data=await response.json();}catch{}
    if(!retryable.has(response.status)||attempt===5)
      throw new Error(attempt===5?'服务器仍然忙碌或暂时不可用，请稍后再试。':data.error||`HTTP ${response.status}`);
    request.waiting=true;
    for(let remaining=3;remaining>0;remaining--) {
      answer.meta=`等待重试 ${attempt+1}/5`;updateBubble(answer);
      $('#generation-status').textContent=`服务器忙碌或暂时不可用，${remaining} 秒后重试（${attempt+1}/5）…`;
      await retryDelay(1000,request.abort.signal);
    }
    answer.meta='正在重新连接';updateBubble(answer);
  }
}
function confirmEnglishInput() {
  return new Promise(resolve=>{
    const dialog=$('#english-warning'),cancel=$('#english-cancel'),send=$('#english-continue');
    const cancelled=event=>{event.preventDefault();finish(false);};
    const yes=()=>finish(true),no=()=>finish(false);
    const finish=value=>{
      dialog.removeEventListener('cancel',cancelled);cancel.removeEventListener('click',no);send.removeEventListener('click',yes);
      dialog.close();resolve(value);
    };
    dialog.addEventListener('cancel',cancelled);cancel.addEventListener('click',no);send.addEventListener('click',yes);
    dialog.showModal();
  });
}
async function send(text) {
  if(active||awaitingLanguageConfirm||!text.trim()) return;
  text=text.trim();
  if(/[^\x00-\x7F]/.test(text+$('#system-message').value)) {
    awaitingLanguageConfirm=true;let accepted;
    try{accepted=await confirmEnglishInput();}finally{awaitingLanguageConfirm=false;}
    if(!accepted)return;
  }
  notice('');userScrolled=false;
  seedForMessage();
  const user={id:newId(),role:'user',content:text,status:'done'};
  const answer={id:newId(),role:'assistant',content:'',status:'streaming',meta:'生成中'};
  messages.push(user,answer);
  const request={id:newId(),abort:new AbortController(),stopped:false};active=request;
  const history=requestMessages(text);const params={...settings};
  $('#prompt').value='';$('#prompt').style.height='';busy(true);renderMessages();scrollBottom(true);
  let completed=false,first=true,meta=null;
  try {
    const response=await chatResponse(request,history,params,answer);
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    const consume=packet=>{
      const line=packet.split('\n').find(l=>l.startsWith('data: '));if(!line) return;
      const event=JSON.parse(line.slice(6));
      if(event.type==='meta') {
        meta=event;$('#context-meter').textContent=`${event.prompt_tokens.toLocaleString()} / ${event.context_tokens.toLocaleString()} tokens`;
        if(event.dropped_turns) notice(`本轮上下文已省略最早 ${event.dropped_turns} 轮。完整记录仍在页面中。`);
      } else if(event.type==='delta') {
        if(first) {first=false;$('#generation-status').textContent='Receiving tokens…';}
        answer.content+=event.text;updateBubble(answer);
      } else if(event.type==='done') {
        completed=true;answer.status=event.finish_reason==='stopped'?'stopped':'done';
        answer.meta=`${event.output_tokens} tokens · ${event.elapsed_seconds.toFixed(1)}s`+(event.finish_reason==='stopped'?' · 已停止':event.finish_reason==='length'?' · 达到长度上限':event.finish_reason==='timeout'?' · 达到时间上限':'');
        const used=(meta?.prompt_tokens||0)+event.output_tokens;
        $('#context-meter').textContent=`${used.toLocaleString()} / ${(meta?.context_tokens||1024).toLocaleString()} tokens`;
        $('#generation-status').textContent=`${event.tokens_per_second.toFixed(1)} tokens/s · seed ${params.seed}`;
      } else if(event.type==='error') throw new Error(event.message);
    };
    while(true) {
      const chunk=await reader.read();buffer+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done});
      let cut;while((cut=buffer.indexOf('\n\n'))!==-1){consume(buffer.slice(0,cut));buffer=buffer.slice(cut+2);}
      if(chunk.done) break;
    }
    if(buffer.trim()) consume(buffer);
    if(!completed && !request.stopped) throw new Error('连接已结束，回答可能未完成。');
  } catch(error) {
    if(request.reloading)return;
    if(request.stopped||error.name==='AbortError') {
      answer.status='stopped';answer.meta='已停止';$('#generation-status').textContent='Stopped.';
    } else {
      answer.status='error';answer.meta='生成失败';notice(error.message,true);$('#generation-status').textContent='Please try again.';
    }
  } finally {
    if(!request.reloading) {
      if(!completed && !['error','stopped'].includes(answer.status)) {answer.status='stopped';answer.meta='已停止';}
      // Also signal cancellation when an HTTP/SSE error interrupts consumption.
      if(!completed) fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({request_id:request.id})}).catch(()=>{});
      if(active===request) active=null;busy(false);renderMessages();save();
      if(!$('#chat-view')?.hidden&&!window.matchMedia('(pointer:coarse)').matches)$('#prompt').focus();
    }
  }
}
async function stop() {
  if(!active) return;const request=active;request.stopped=true;
  request.abort.abort();
  $('#generation-status').textContent='Stopping…';
  try {if(!request.waiting)await fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({request_id:request.id})});}
  catch {} finally {request.abort.abort();}
}

document.querySelectorAll('input[type=range]').forEach(input=>{
  input.addEventListener('input',()=>{settings[input.id]=Number(input.value);paintSettings();save();});
});
document.querySelectorAll('[data-number]').forEach(input=>{
  input.addEventListener('input',()=>{
    if(input.value.trim() && input.validity.valid) {settings[input.dataset.number]=Number(input.value);paintSettings(input);save();}
  });
  input.addEventListener('change',()=>{const value=readNumber(input);if(value!==null)settings[input.dataset.number]=value;paintSettings();save();});
});
$('#seed').addEventListener('change',()=>{const value=readNumber($('#seed'));if(value!==null)settings.seed=value;paintSettings();save();});
$('#seed').addEventListener('input',()=>{const input=$('#seed');if(input.value.trim()&&input.validity.valid){settings.seed=Number(input.value);paintSettings(input);save();}});
$('#fixed-seed')?.addEventListener('change',()=>{fixedSeed=$('#fixed-seed').checked;paintSettings();save();});
$('#do_sample').addEventListener('change',()=>{settings.do_sample=$('#do_sample').checked;paintSettings();save();});
$('#system-message').addEventListener('input',save);
$('#reset-settings').addEventListener('click',()=>{settings={...defaults};fixedSeed=!$('#fixed-seed');$('#system-message').value='';paintSettings();save();toast('已恢复推荐参数。');});
document.querySelectorAll('[data-theme]').forEach(button=>button.addEventListener('click',()=>{chooseTheme(button.dataset.theme);save();}));
document.querySelectorAll('.starter').forEach(button=>button.addEventListener('click',()=>{$('#prompt').value=button.dataset.prompt;$('#prompt').focus();}));
$('#composer').addEventListener('submit',event=>{event.preventDefault();send($('#prompt').value);});
$('#prompt').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing&&!window.matchMedia('(pointer:coarse)').matches){event.preventDefault();send($('#prompt').value);}});
$('#prompt').addEventListener('input',()=>{$('#prompt').style.height='auto';$('#prompt').style.height=Math.min(160,$('#prompt').scrollHeight)+'px';});
$('#stop').addEventListener('click',stop);
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&active)stop();});
$('#new-chat').addEventListener('click',()=>{if(active)return;messages=[];renderMessages();notice('');$('#generation-status').textContent='A fresh start.';$('#context-meter').textContent=`0 / ${settings.context_tokens.toLocaleString()} tokens`;save();$('#prompt').focus();});
$('#export-chat').addEventListener('click',()=>{
  const blob=new Blob([JSON.stringify({model:$('#model-name').textContent,parameters:settings,messages},null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='clover-conversation.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
});
$('#conversation').addEventListener('scroll',()=>{const el=$('#conversation');userScrolled=el.scrollHeight-el.clientHeight-el.scrollTop>85;});
window.addEventListener('beforeunload',()=>{if(active&&!active.reloading)navigator.sendBeacon('/api/stop',new Blob([JSON.stringify({request_id:active.id})],{type:'application/json'}));});

async function init() {
  try {
    const stored=JSON.parse(localStorage.getItem(STORAGE)||'null');
    if(stored) {
      settings={...FALLBACK,...stored.settings};messages=Array.isArray(stored.messages)?stored.messages.filter(m=>['user','assistant'].includes(m.role)&&typeof m.content==='string').slice(-200).map(m=>({...m,id:typeof m.id==='string'&&/^[a-zA-Z0-9_-]+$/.test(m.id)?m.id:newId()})):[];
      fixedSeed=!$('#fixed-seed')||stored.fixedSeed===true;
      $('#prompt').value=typeof stored.draft==='string'?stored.draft.slice(0,12000):'';
      $('#system-message').value=stored.system||'';chooseTheme(stored.theme||'paper');
    }
  } catch {}
  paintSettings();renderMessages();scrollBottom(true);
  try {
    const response=await fetch('/api/state');if(!response.ok)throw new Error('状态接口不可用');const state=await response.json();
    defaults={...FALLBACK,...state.defaults};ready=state.ready;
    for(const [key,[min,max]] of Object.entries(state.limits||{})) {
      const input=$('#'+key),number=$(`[data-number="${key}"]`);
      for(const control of [input,number]) if(control){control.min=min;control.max=max;}
      settings[key]=Math.min(max,Math.max(min,Number(settings[key])));
    }
    $('#model-name').textContent=state.ready?state.model:'English · 125M · Preview';
    $('#connection-label').textContent=ready?'模型已连接':'界面预览 · 未加载模型';
    $('#status-dot').classList.toggle('ready',ready);
    if(state.preview)notice('当前是界面预览。训练完成后，用模型服务命令启动即可聊天。');
    paintSettings();
  } catch {$('#connection-label').textContent='服务器未连接';notice('无法连接模型服务，请稍后重试。',true);}
  const clock=()=>{$('#local-clock').textContent=new Date().toLocaleTimeString('en-GB',{hour12:false});};clock();setInterval(clock,1000);
}
init();
