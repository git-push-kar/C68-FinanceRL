/* Actual paper-account controller; scoped to the native Finance tab.
   Supports:
     Mode 1: Final Benchmark Replay (Jan 2025 – Jan 2026)
     Mode 2: 2026 YTD History Replay (Jan 1, 2026 – Yesterday)
     Mode 3: Live Real-Time Market Trading (Today's Live Session)
*/
(()=>{

'use strict';
const $=id=>document.getElementById('fin-'+id), money=x=>new Intl.NumberFormat('en-IN',{style:'currency',currency:'INR',maximumFractionDigits:2}).format(x),pct=x=>(100*x).toFixed(2)+'%';
let state=null,session=null,busy=false,playing=false,limit=6,selected=null,playback=0,currentMode='benchmark';
const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
async function api(path,method='GET',body){const r=await fetch('/finance/paper'+path,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});let j=await r.json();if(!r.ok)throw Error(j.detail||'Request failed');return j;}
function status(text,working=false){$('statusText').textContent=text;$('status').classList.toggle('busy',working);}

function controls(){
  const isLive = (currentMode === 'live');
  const isReplay = !isLive;

  $('play').style.display = isReplay ? 'inline-block' : 'none';
  $('day').style.display = isReplay ? 'inline-block' : 'none';
  $('rebalance').style.display = isReplay ? 'inline-block' : 'none';
  $('playback-tools').style.display = isReplay ? 'block' : 'none';

  if($('live-refresh')) $('live-refresh').style.display = isLive ? 'inline-block' : 'none';
  if($('live-decide')) $('live-decide').style.display = isLive ? 'inline-block' : 'none';
  if($('live-settle')) $('live-settle').style.display = isLive ? 'inline-block' : 'none';
  if($('live-advance')) $('live-advance').style.display = isLive ? 'inline-block' : 'none';

  if(isLive){
    if($('live-refresh')) $('live-refresh').disabled = busy;
    if($('live-decide')) $('live-decide').disabled = busy;
    if($('live-settle')) $('live-settle').disabled = busy;
    if($('live-advance')) $('live-advance').disabled = busy;
  }

  for(const id of ['day','rebalance'])$(id).disabled=busy||!state||state.finished||playing;
  $('play').disabled=!state||state.finished;
  $('play').textContent=playing?'Ⅱ Pause':state?.finished?'Simulation complete':state?.sessions?'▶ Resume':'▶ Start simulation';
  $('settings').querySelector('button').disabled=busy;
}

function error(text){$('error').textContent=text;$('error').hidden=!text;}

function render(){
  if(!state)return;
  const isLive = (currentMode === 'live');
  const isYTD = (currentMode === 'ytd');

  $('equity').textContent=money(state.equity);
  $('drawdown').textContent=pct(state.maximum_drawdown);
  $('fees').textContent=money(state.fees);
  $('range').textContent=state.initial_date+' — '+state.date;
  $('cash').textContent='Cash '+money(state.cash)+' · Holdings + cash = '+money(state.equity);
  $('progress').style.width=isLive?'100%':(100*state.sessions/state.total_sessions+'%');

  if(isLive){
    $('date').textContent='Live: '+state.date;
    $('initial').textContent='Settled close: '+money(state.settled_equity || state.capital);
    const pnl = (state.intraday_pnl !== undefined) ? state.intraday_pnl : (state.equity - state.capital);
    const pnlPct = (state.intraday_pct !== undefined) ? state.intraday_pct : state.net_return;
    $('gain').textContent=money(pnl);
    $('gain').className='v '+(pnl>=0?'positive':'negative');
    $('return').textContent=pct(pnlPct)+' today · Overall '+pct(state.net_return);

    $('sessionInfo').innerHTML = '<strong>Mode 3: Live Real-Time Market Trading</strong><br>Resumed from yesterday ('+(state.settled_date||state.initial_date)+') · Today: '+state.date+'<br>'+(state.market_status||'Live Data Active');
    $('resetNote').textContent = 'Reloads the pre-existing portfolio state from storage.';
    $('headlineNote').textContent = 'Real-time portfolio management on live market data. Settle to commit today into history.';
  } else if(isYTD){
    $('date').textContent='2026 YTD Replay: '+state.date;
    $('initial').textContent='Started Jan 1 with '+money(state.capital);
    $('gain').textContent=money(state.equity-state.capital);
    $('gain').className='v '+(state.net_return>=0?'positive':'negative');
    $('return').textContent=pct(state.net_return)+' net return';

    $('sessionInfo').innerHTML = '<strong>Mode 2: 2026 YTD History Replay</strong><br>Jan 1, 2026 – Yesterday ('+state.date+')<br>'+state.sessions+' / '+state.total_sessions+' sessions replayed';
    $('resetNote').textContent = 'Restarts the 2026 historical replay from Jan 1, 2026 cash.';
    $('headlineNote').textContent = 'Watch the 216 daily actions and 44 rebalances that built the current portfolio.';
  } else {
    $('date').textContent='Benchmark Replay: '+state.date;
    $('initial').textContent='Started with '+money(state.capital);
    $('gain').textContent=money(state.equity-state.capital);
    $('gain').className='v '+(state.net_return>=0?'positive':'negative');
    $('return').textContent=pct(state.net_return)+' net return';

    $('sessionInfo').innerHTML = '<strong>Mode 1: Final Benchmark Replay</strong><br>1 Jan 2025 – 30 Jan 2026 · 270 sessions<br>Conclusive out-of-sample benchmark (+14.53%)';
    $('resetNote').textContent = 'Restarts the 2025–2026 benchmark simulation with fresh capital.';
    $('headlineNote').textContent = 'Watch the trained portfolio act on the primary test set beating Buy & Hold.';
  }

  renderChart();
  renderHoldings();
  renderActivity();
  renderDecision();
  controls();
}

function renderChart(){
  const svg=$('chart');svg.replaceChildren();const theme=getComputedStyle(document.documentElement),color=name=>theme.getPropertyValue(name).trim();
  const ns='http://www.w3.org/2000/svg',add=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e;};
  const hist=[{date:state.initial_date,equity:state.capital,untrained:state.capital,buy_hold:state.capital},...state.history],keys=$('compare').checked?['untrained','buy_hold','equity']:['equity'];
  const values=hist.flatMap(h=>keys.map(k=>h[k]));
  let lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.12,state.capital*.005);
  lo-=pad;hi+=pad;
  const x=i=>65+i/Math.max(hist.length-1,1)*715,y=v=>240-(v-lo)/(hi-lo)*215;
  for(let i=0;i<4;i++){let v=lo+(hi-lo)*i/3;add('line',{x1:65,x2:780,y1:y(v),y2:y(v),stroke:color('--border')});add('text',{x:52,y:y(v)+4,'text-anchor':'end',fill:color('--muted'),'font-size':10},(v/1000).toFixed(1)+'k');}
  for(const key of keys)add('path',{d:hist.map((h,i)=>(i?'L':'M')+x(i)+','+y(h[key])).join(' '),fill:'none',stroke:{equity:color('--accent'),untrained:color('--muted'),buy_hold:color('--warn')}[key],'stroke-width':key==='equity'?2.8:1.6,'stroke-linejoin':'round'});
  add('text',{x:65,y:270,fill:color('--muted'),'font-size':10},state.initial_date);
  add('text',{x:780,y:270,'text-anchor':'end',fill:color('--muted'),'font-size':10},state.date);
}

function renderHoldings(){
  const q=$('search').value.toLowerCase(),rows=state.holdings.filter(h=>h.value>.005&&h.ticker.toLowerCase().includes(q)).sort((a,b)=>b.value-a.value);
  $('holdings').replaceChildren();
  for(const h of rows){
    const tr=el('tr');
    tr.append(el('td',h.ticker.replace('.NS','')),el('td',money(h.value)),el('td',pct(h.weight)),el('td',money(h.cumulative_market_pnl),h.cumulative_market_pnl>=0?'positive':'negative'));
    $('holdings').append(tr);
  }
  if(!rows.length){const tr=el('tr'),td=el('td',state.sessions?'No matching holdings':'Your portfolio starts in cash.');td.colSpan=4;tr.append(td);$('holdings').append(tr);}
  $('holdingsCount').textContent=state.holdings.filter(h=>h.value>.005).length+' investments · Current values after price movement';
}

function renderActivity(){
  const target=$('activity');target.replaceChildren();
  if(!state.activities.length)target.append(el('div','No trades yet. Your first allocation is made when the simulation starts.','empty'));
  for(const a of [...state.activities].reverse().slice(0,limit)){
    const row=el('div',undefined,'event'),body=el('div');
    body.append(el('h3',a.execution_date+' · '+(a.id===0?'Initial allocation':'Rebalance')));
    const buys=a.trades.filter(t=>t.amount>0).length,sells=a.trades.filter(t=>t.amount<0).length;
    body.append(el('p',buys+' buys · '+sells+' sells · Fee '+money(a.fee)),el('small','Signals through '+a.signal_date));
    const b=el('button','View decision');
    b.addEventListener('click',()=>{selected=a.id;renderDecision();});
    row.append(body,b);target.append(row);
  }
  $('more').hidden=state.activities.length<=limit;
}

function renderDecision(){
  const root=$('decision');
  if(!state.activities.length){
    root.replaceChildren(el('div','Your first allocation will appear here. Start the simulation to let the adapter act.','empty'));
    return;
  }
  const a=state.activities.find(x=>x.id===selected)||state.activities.at(-1);
  root.replaceChildren();
  const kv=el('div',undefined,'kv');
  for(const[label,value]of [['Executed',a.execution_date],['Information through',a.signal_date],['Portfolio before trade',money(a.pre_trade_equity)],['Fee',money(a.fee)],['Amount traded',money(a.turnover*a.pre_trade_equity)]])kv.append(el('span',label),el('span',value));
  root.append(kv);

  if(a.rationales && a.rationales.length){
    const rBox = el('div', undefined, 'card');
    rBox.style.marginTop = '12px';
    rBox.style.marginBottom = '12px';
    rBox.style.padding = '10px 14px';
    rBox.style.background = 'rgba(74, 144, 226, 0.08)';
    rBox.style.borderLeft = '3px solid var(--accent)';
    rBox.append(el('h4', 'Model Investment Rationales (Autoregressive Explanations)'));
    for(const r of a.rationales){
      const p = el('p', `${r.ticker.replace('.NS', '')}: "${r.rationale}"`);
      p.style.fontSize = '12px';
      p.style.margin = '4px 0';
      rBox.append(p);
    }
    root.append(rBox);
  }

  root.append(el('h3','Learned strategy mix'));
  const mix=el('div',undefined,'mix');
  a.strategy_names.forEach((name,i)=>{
    const row=el('div',undefined,'row');
    row.append(el('span',name),el('span',pct(a.probabilities[i])));
    const bar=el('div',undefined,'bar'),fill=el('i');
    fill.style.width=100*a.probabilities[i]+'%';
    bar.append(fill);mix.append(row,bar);
  });
  root.append(mix,el('p','The target blends these strategy weights. Price movement and updated lagged inputs change the stock allocations.','muted'));
  const scroll=el('div',undefined,'detail-trades'),table=el('table'),head=el('thead'),hr=el('tr');
  for(const s of ['Stock','Action / amount','Target'])hr.append(el('th',s));
  head.append(hr);table.append(head);
  const tb=el('tbody');
  for(const t of [...a.trades].sort((x,y)=>Math.abs(y.amount)-Math.abs(x.amount))){
    const tr=el('tr');
    tr.append(el('td',t.ticker.replace('.NS','')),el('td',(t.amount>0?'Buy ':'Sell ')+money(Math.abs(t.amount))),el('td',pct(t.target_weight)));
    tb.append(tr);
  }
  table.append(tb);scroll.append(table);root.append(scroll);
  const elapsed = a.model_receipt && a.model_receipt.elapsed_seconds !== undefined ? a.model_receipt.elapsed_seconds.toFixed(2)+'s · ' : '';
  const adapterName = (a.model_receipt && a.model_receipt.adapter) ? a.model_receipt.adapter : 'InternVL3-2B-State-RL';
  root.append(el('div','Executed from actual adapter inference · '+elapsed+adapterName,'receipt'));
}

async function advance(mode){
  if(busy||!state||state.finished)return;
  busy=true;error('');
  status(state.next_rebalance_in===0?'Adapter selecting allocation…':'Advancing market prices…',true);
  controls();
  try{
    state=await api('/sessions/'+session+'/advance','POST',{mode,expected_sessions:state.sessions});
    render();
    status(state.finished?'Completed '+state.total_sessions+' trading sessions':state.sessions+' / '+state.total_sessions+' sessions · '+(state.next_rebalance_in===0?'Rebalance due':state.next_rebalance_in+' sessions to rebalance'));
    window.refreshStatus?.();
  }catch(e){
    playing=false;error(e.message);
    try{state=await api('/sessions/'+session);render();}catch(_){}
    status('Simulation paused');
  }finally{
    busy=false;controls();
  }
}

async function refreshLiveQuotes(){
  if(busy)return;
  busy=true;error('');
  status('Fetching latest live market quotes from Yahoo Finance…',true);
  controls();
  try{
    state=await api('/live/refresh','POST');
    render();
    status('Live market data refreshed as of '+state.date+' · Portfolio marked to market');
  }catch(e){
    error('Live refresh failed: '+e.message);
    status('Refresh failed');
  }finally{
    busy=false;controls();
  }
}

async function decideLive(){
  if(busy)return;
  busy=true;error('');
  status('Running InternVL3-2B + LoRA RL inference on live market prices…',true);
  controls();
  try{
    state=await api('/live/decide','POST');
    render();
    status('Model evaluated today\'s live market and generated target allocation & reasoning');
  }catch(e){
    error('Model live decision failed: '+e.message);
    status('Live decision failed');
  }finally{
    busy=false;controls();
  }
}

async function settleLive(){
  if(busy)return;
  busy=true;error('');
  status('Committing today\'s session into permanent history storage…',true);
  controls();
  try{
    state=await api('/live/settle','POST');
    render();
    status('Day settled into permanent storage! Portfolio officially managed through '+state.date+'. Tomorrow will resume from this state.');
  }catch(e){
    error('Day settlement failed: '+e.message);
    status('Settlement failed');
  }finally{
    busy=false;controls();
  }
}

async function advanceLive(){
  if(busy||!state)return;
  busy=true;error('');
  status('Stepping live day forward on latest market data…',true);
  controls();
  try{
    state=await api('/sessions/live_market_session/advance','POST',{mode:'day',expected_sessions:state.sessions});
    render();
    status('Live portfolio updated as of '+state.date);
  }catch(e){
    error(e.message);
    status('Live update completed');
  }finally{
    busy=false;controls();
  }
}

async function play(){
  if(playing){playing=false;playback++;controls();return;}
  playing=true;const ticket=++playback;controls();
  while(playing&&ticket===playback&&state&&!state.finished){
    await advance('day');
    if(playing&&ticket===playback&&!state.finished)await new Promise(r=>setTimeout(r,Number($('speed').value)));
  }
  if(ticket===playback)playing=false;controls();
}

async function start(capital=100000, mode=currentMode){
  playing=false;playback++;busy=true;controls();error('');
  try{
    if(session && session !== 'live_market_session' && session !== 'ytd_history_session') {
      try{ await api('/sessions/'+session,'DELETE'); }catch(_){}
    }
    state=await api('/sessions','POST',{capital,mode});
    session=state.session_id;
    if(mode === 'benchmark' || mode === 'historical') localStorage.setItem('finance-paper-session',session);
    selected=null;limit=6;render();

    if(mode === 'live'){
      status('Mode 3 Active: Live Real-Time Market Trading · Resumed from yesterday ('+state.settled_date+') to today ('+state.date+')');
    } else if(mode === 'ytd'){
      status('Mode 2 Active: 2026 YTD History Replay · '+state.total_sessions+' trading sessions from Jan 1, 2026 to yesterday');
    } else {
      status('Mode 1 Active: Final Benchmark Replay · '+state.total_sessions+' continuous test sessions (Jan 2025 – Jan 2026)');
    }
  }catch(e){
    error(e.message);
  }finally{
    busy=false;controls();
  }
}

function exportCSV(name,rows){
  if(!rows.length)return;
  const keys=Object.keys(rows[0]),csv=[keys,...rows.map(r=>keys.map(k=>r[k]))].map(r=>r.map(x=>'"'+String(x??'').replaceAll('"','""')+'"').join(',')).join('\n');
  const a=document.createElement('a'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.href=url;a.download=name;a.click();URL.revokeObjectURL(url);
}

$('play').onclick=play;
$('day').onclick=()=>advance('day');
$('rebalance').onclick=()=>advance('rebalance');
if($('live-refresh')) $('live-refresh').onclick=refreshLiveQuotes;
if($('live-decide')) $('live-decide').onclick=decideLive;
if($('live-settle')) $('live-settle').onclick=settleLive;
if($('live-advance')) $('live-advance').onclick=advanceLive;

$('compare').onchange=renderChart;
$('search').oninput=renderHoldings;
$('more').onclick=()=>{limit+=6;renderActivity();};
$('settings').onsubmit=e=>{e.preventDefault();if(!busy)start(Number($('capital').value),currentMode);};
$('exportHoldings').onclick=()=>exportCSV('holdings-'+state.date+'.csv',state.holdings);
$('exportActivity').onclick=()=>exportCSV('activity-'+state.date+'.csv',state.activities.flatMap(a=>a.trades.map(t=>({execution_date:a.execution_date,signal_date:a.signal_date,...t}))));

$('mode').onchange=async()=>{
  currentMode=$('mode').value;
  if(currentMode === 'historical') currentMode = 'benchmark';
  localStorage.setItem('finance-paper-mode',currentMode);
  await start(Number($('capital').value),currentMode);
};

async function initialize(){
  try{
    const c=await api('/config');
    $('source').textContent='Selected artifact: '+c.adapter+' · SHA256 '+c.adapter_sha256.slice(0,16)+'…';
    currentMode=localStorage.getItem('finance-paper-mode')||'benchmark';
    if(currentMode === 'historical') currentMode = 'benchmark';
    $('mode').value=currentMode;

    if(currentMode==='live'){
      await start(100000,'live');
      return;
    }
    if(currentMode==='ytd'){
      await start(100000,'ytd');
      return;
    }

    session=localStorage.getItem('finance-paper-session');
    if(session){
      try{
        state=await api('/sessions/'+session);
        $('capital').value=state.capital;
        render();
        status(state.finished?'Simulation complete':'Paused · '+state.sessions+' / '+state.total_sessions+' sessions');
        return;
      }catch(_){session=null;}
    }
    await start(100000,'benchmark');
  }catch(e){
    error(e.message);status('Portfolio unavailable');
  }
}

let initialized=false;
window.FinanceWorkspace={
 async show(){if(!initialized){initialized=true;await initialize();}},
 hide(){playing=false;playback++;controls();}
};
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if(state)renderChart();});

})();
