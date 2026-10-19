import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {finalizePresentation} from 'file:///C:/Users/zjh/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/artifact_tool_utils.mjs';
const p=Presentation.create({slideSize:{width:858,height:858}}),s=p.slides.add();s.background.fill='#FFFFFF';
const blue='#08479F',cyan='#68D4F5', S=3;
function sh(g,x,y,w,h,fill='none',stroke=blue,lw=.8,name=g){return s.shapes.add({geometry:g,name,position:{left:x*S,top:y*S,width:w*S,height:h*S},fill,line:{fill:stroke,width:lw*S,style:'solid'}})}
function path(d,fill='none',stroke=blue,lw=.8,name='可编辑曲线') {let cmds=[];for(let a of d){if(a[0]==='M')cmds.push({moveTo:{x:a[1],y:a[2]}});if(a[0]==='L')cmds.push({lineTo:{x:a[1],y:a[2]}});if(a[0]==='C')cmds.push({cubicBezTo:{x1:a[1],y1:a[2],x2:a[3],y2:a[4],x:a[5],y:a[6]}});if(a[0]==='Z')cmds.push({close:{}})}return s.shapes.add({geometry:'custom',name,position:{left:0,top:0,width:858,height:858},fill,line:{fill:stroke,width:lw*S,style:'solid'},customPaths:[{width:286,height:286,commands:cmds}]})}
function txt(t,x,y,w,h,size,rot=0){let a=sh('textbox',x,y,w,h,'none','none',0,t);a.position.rotation=rot;a.text=t;a.text.style={typeface:'Microsoft YaHei',fontSize:size*S,bold:true,color:blue,alignment:'center',verticalAlignment:'middle'};a.text.insets=0;return a}
sh('ellipse',12.5,12.5,261,261,'#FFFFFF',blue,.95,'外圆环');
sh('ellipse',53.5,51.5,180,180,'#FFFFFF',blue,.9,'内圆环');
const top='智能感知与光电信息处理';Array.from(top).forEach((c,i)=>{let a=(-151+i*122/(top.length-1))*Math.PI/180;let x=143+108*Math.cos(a),y=143+108*Math.sin(a);txt(c,x-12,y-12,24,24,20,a*180/Math.PI+90)});
const bot='研究生创新论坛';Array.from(bot).forEach((c,i)=>{let a=(140-i*100/(bot.length-1))*Math.PI/180;let x=143+103*Math.cos(a),y=143+103*Math.sin(a);txt(c,x-13,y-13,26,26,21,a*180/Math.PI-90)});

// Optical sensor and light rays.
for(let i=0;i<24;i++)sh('rect',77+i*1.5,97,1.6,23,'#'+[Math.round(94+(255-94)*i/24),Math.round(200+(255-200)*i/24),255].map(n=>n.toString(16).padStart(2,'0')).join(''),'none',0,'光束');
let sensor=sh('ellipse',66.5,95,17.5,30,cyan,blue,.9,'光传感器');sensor.position.rotation=12;
let lens=sh('ellipse',71,100,7.5,19,'#087ABF',blue,.8,'传感器透镜');lens.position.rotation=12;
// Brain silhouette, left and right hemispheres.
path([['M',176,65],['C',168,61,161,64,160,71],['C',152,69,145,75,147,82],['C',138,84,139,91,143,95],['C',136,102,140,110,148,110],['C',153,107,157,112,163,109],['C',168,115,174,111,177,110],['L',177,68],['Z']],cyan,blue,1.1,'左脑');
path([['M',182,68],['C',188,65,194,69,198,75],['C',204,76,211,84,211,90],['C',218,95,215,103,211,107],['C',214,113,208,119,201,119],['L',201,125],['L',180,125],['L',180,69],['Z']],blue,blue,.8,'右脑');
for(const d of [
[['M',164,72],['C',157,76,165,80,158,83],['C',149,81,146,86,150,91]],
[['M',171,70],['C',175,77,166,77,168,83],['C',170,88,163,91,158,89],['C',151,88,145,93,148,99]],
[['M',176,88],['C',169,87,165,94,169,99],['C',164,100,162,105,163,109]],
[['M',143,103],['C',149,97,155,101,158,105]],
[['M',176,103],['C',173,99,169,100,167,102]]])path(d,'none',blue,1.1,'脑回曲线');
for(const d of [
[['M',186,70],['L',186,88],['L',194,88],['L',199,94],['L',207,94]],
[['M',192,77],['L',186,81]],
[['M',181,100],['L',189,100],['L',194,105],['L',205,105]],
[['M',182,122],['L',188,116],['L',188,109]],
[['M',198,78],['L',202,83],['L',202,88]]])path(d,'none','#FFFFFF',1.2,'脑电路');
for(const [x,y]of [[186,71],[194,77],[207,94],[181,100],[205,105],[188,109],[202,88]])sh('ellipse',x-1.5,y-1.5,3,3,'#FFFFFF','none',0,'电路节点');
// Pulse waveform.
path([['M',70,185],['L',79,185],['C',84,185,84,168,88,169],['C',93,169,92,194,97,194],['C',100,194,101,178,105,180],['C',109,180,108,187,113,186],['C',117,186,119,182,123,182],['L',135,182]],'none',blue,.9,'感知波形');
// Editable monitor, chip and individual chip pixels.
sh('roundRect',135,126,86,69,'#FFFFFF',blue,1.1,'显示器边框');
sh('rect',137,189,82,4,blue,'none',0,'显示器下边');
path([['M',162,195],['L',189,195],['L',193,201],['L',157,201],['Z']],cyan,'none',0,'显示器支架');
path([['M',157,203],['L',199,203]],'none',blue,1,'显示器底座');
sh('rect',155,135,47,37,'#047FC4',blue,.5,'芯片');
for(let r=0;r<9;r++)for(let c=0;c<11;c++){let dx=c-5,dy=r-4;let on=((Math.abs(dx)+Math.abs(dy)<5)&&((r+c)%2===0))||(Math.abs(dx)<2&&Math.abs(dy)<2);if(on)sh('rect',157+c*3.9,137+r*3.7,3,2.8,(Math.abs(dx)+Math.abs(dy)<2)?'#06D8E8':'#00C7E5','none',0,'芯片像素');}
for(const y of [135,145,156])sh('rect',142,y,3,3,blue,'none',0,'接口');
for(const y of [139,150])sh('rect',210,y,3,3,blue,'none',0,'接口');
let wave=[['M',137,180]];for(let x=137;x<208;x+=10)wave.push(['C',x+2,180,x+2,173,x+5,174],['C',x+8,175,x+7,184,x+10,181]);path(wave,'none',cyan,.8,'显示器信号曲线');
await fs.writeFile('D:/zhangjh100/AttCo/.ppt-build/preview.png',new Uint8Array(await (await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
const candidate='D:/zhangjh100/AttCo/.ppt-build/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
await finalizePresentation({workspaceDir:'D:/zhangjh100/AttCo',candidatePath:candidate,finalPath:'D:/zhangjh100/AttCo/output/可编辑论坛标志_无年份版.pptx',pythonExecutable:'C:/Users/zjh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:'C:/Users/zjh/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:'C:/Users/zjh/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','8172450,8172450'],explicitTotalSlideCount:1,fontPolicy:{basis:'design',families:['Microsoft YaHei']},verifyArtifactToolImport:true,receiptPath:'D:/zhangjh100/AttCo/.ppt-build/validation-no-year.json'});
console.log('DONE');




