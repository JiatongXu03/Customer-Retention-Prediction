"""Execute standard Python notebook cells sequentially, without a Jupyter kernel.

Only for trusted notebooks in this project. Outputs are genuine stdout/figures.
The same .ipynb files can also be opened and run normally in Jupyter/VS Code.
"""
from pathlib import Path
import base64, contextlib, io, json, os, sys, time, traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def execute(path):
    nb=json.loads(path.read_text(encoding='utf-8')); env={'__name__':'__main__'};count=0
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
    start=time.time()
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        count+=1;cell['execution_count']=count;cell['outputs']=[];capture=io.StringIO()
        source=''.join(cell['source'])
        try:
            with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
                exec(compile(source,f'{path.name}:cell{count}','exec'),env)
        except Exception:
            cell['outputs'].append({'output_type':'error','ename':'ExecutionError','evalue':str(sys.exc_info()[1]),'traceback':traceback.format_exc().splitlines()})
            path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
            raise
        if capture.getvalue():cell['outputs'].append({'output_type':'stream','name':'stdout','text':capture.getvalue().splitlines(True)})
        for num in plt.get_fignums():
            buf=io.BytesIO();plt.figure(num).savefig(buf,format='png',bbox_inches='tight',dpi=140)
            cell['outputs'].append({'output_type':'display_data','data':{'image/png':base64.b64encode(buf.getvalue()).decode(),'text/plain':['Matplotlib figure']},'metadata':{}})
        plt.close('all')
        print(path.name,'cell',count,'OK',flush=True)
    nb['metadata']['execution']={'method':'Sequential Python exec; stdout and Matplotlib captured; no Jupyter kernel','seconds':round(time.time()-start,2)}
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return {'notebook':path.name,'code_cells':count,'seconds':round(time.time()-start,2),'status':'passed'}

if __name__=='__main__':
    paths=[ROOT/'notebooks'/n for n in sys.argv[1:]] if len(sys.argv)>1 else sorted((ROOT/'notebooks').glob('*.ipynb'))
    results=[execute(p) for p in paths]
    log=ROOT/'outputs/execution_log.json'
    previous=json.loads(log.read_text()) if log.exists() else []
    merged={r['notebook']:r for r in previous}
    merged.update({r['notebook']:r for r in results})
    log.write_text(json.dumps([merged[k] for k in sorted(merged)],indent=2),encoding='utf-8')
