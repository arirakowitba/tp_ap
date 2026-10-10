"""Execute project notebooks without requiring local sockets."""
import argparse,os,base64,io
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
def execute(nb):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell=InteractiveShell.instance();previous_show=plt.show;images=[]
    def show(*args,**kwargs):
        for number in plt.get_fignums():
            figure=plt.figure(number);buffer=io.BytesIO();figure.savefig(buffer,format='png',bbox_inches='tight')
            images.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode()},metadata={}))
        plt.close('all')
    plt.show=show
    try:
        execution=0
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            execution+=1;images=[]
            source=cell.source.replace("get_ipython().run_line_magic('matplotlib','inline')","pass  # executor captures matplotlib figures")
            with capture_output() as captured:result=shell.run_cell(source,store_history=False)
            outputs=[]
            if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
            for item in captured.outputs:outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
            outputs.extend(images);cell.outputs=outputs;cell.execution_count=execution
            if result.error_before_exec or result.error_in_exec:raise RuntimeError(f'Celda {execution}: {result.error_before_exec or result.error_in_exec}')
    finally:plt.show=previous_show
    nb.metadata['execution_verification']={'executor':'in-process','backend':'Agg','note':'Analytical code executed; only inline backend setup replaced.'}
    return nb
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True)
    p.add_argument('--output',default=str(ROOT/'results/round_v2'));p.add_argument('--mode',choices=['full','report'],default='report')
    p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    os.environ.update(AP2026Q2_DATA_DIR=args.data_dir,AP_OUTPUT_DIR=args.output,AP_RUN_MODE=args.mode,
                      AP_ALLOW_PARTIAL='1' if args.allow_partial else '0',MPLBACKEND='Agg')
    os.chdir(ROOT)
    for name in ['21_maestro_pipeline.ipynb','22_caso_desde_cero.ipynb']:
        path=ROOT/'notebooks'/name;nb=nbformat.read(path,as_version=4);execute(nb);nbformat.validate(nb);nbformat.write(nb,path)
        print('EXECUTED',name,flush=True)

