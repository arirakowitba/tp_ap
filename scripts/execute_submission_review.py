"""Execute all cells in a fresh Python process, retaining rich outputs.

Use --in-process for restricted environments that cannot open Jupyter sockets.
"""
from pathlib import Path
import argparse,json,sys,time,os,tempfile
import nbformat

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--in-process",action="store_true")
args=parser.parse_args()
path=ROOT/"notebooks/23_revision_submissions_y_flips.ipynb"
nb=nbformat.read(path,as_version=4)
start=time.time()
runtime=Path(tempfile.gettempdir())/"ap2026q2_notebook_runtime"
runtime.mkdir(exist_ok=True)
os.environ["JUPYTER_RUNTIME_DIR"]=str(runtime)
os.environ["IPYTHONDIR"]=str(runtime/"ipython")
os.environ["MPLCONFIGDIR"]=str(runtime/"matplotlib")
os.chdir(ROOT)
try:
    if args.in_process:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        os.environ["MPLBACKEND"]="module://matplotlib_inline.backend_inline"
        shell=InteractiveShell.instance()
        import matplotlib
        matplotlib.use("module://matplotlib_inline.backend_inline")
        from matplotlib_inline.backend_inline import configure_inline_support
        configure_inline_support(shell,matplotlib.get_backend())
        count=0
        for index,cell in enumerate(nb.cells):
            if cell.cell_type!="code":
                continue
            count+=1
            print(f"Executing code cell {count} (notebook cell {index+1})",flush=True)
            cell.outputs=[]
            cell.execution_count=count
            with capture_output() as captured:
                result=shell.run_cell(cell.source,store_history=True)
            for name,value in (("stdout",captured.stdout),("stderr",captured.stderr)):
                if value:
                    cell.outputs.append(nbformat.v4.new_output("stream",name=name,text=value))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data",data=output.data,metadata=output.metadata))
            error=result.error_before_exec or result.error_in_exec
            if error:
                cell.outputs.append(nbformat.v4.new_output("error",ename=type(error).__name__,evalue=str(error),traceback=[]))
                raise error
        mode="IPython, fresh Python process, sequential Run All (no sockets)"
    else:
        from nbclient import NotebookClient
        client=NotebookClient(nb,timeout=180,kernel_name="python3",allow_errors=False,
                              startup_timeout=45,resources={"metadata":{"path":str(ROOT)}})
        client.on_cell_start=lambda cell,cell_index: print(f"Cell {cell_index+1}/{len(nb.cells)}",flush=True)
        client.execute()
        mode="nbclient, fresh Python kernel, Run All"
except Exception:
    nbformat.write(nb,path)
    raise
nbformat.validate(nb)
errors=[out for c in nb.cells if c.cell_type=="code" for out in c.get("outputs",[]) if out.output_type=="error"]
assert not errors
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=="code")
nbformat.write(nb,path)
summary={"cells":len(nb.cells),"code_cells":sum(c.cell_type=="code" for c in nb.cells),
         "figures":sum("image/png" in o.get("data",{}) for c in nb.cells for o in c.get("outputs",[])),
         "errors":len(errors),"seconds":time.time()-start,"notebook_bytes":path.stat().st_size,
         "execution":mode}
(ROOT/"results/revision_submissions/execution_report.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("NOTEBOOK EXECUTED",summary,flush=True)

