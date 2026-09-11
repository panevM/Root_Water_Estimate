"""Fresh-kernel Run All with outbound network disabled inside notebook Python."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    # Keep all Jupyter/plotting runtime writes inside this standalone project.
    runtime=ROOT/'outputs'/'runtime'
    runtime.mkdir(parents=True,exist_ok=True)
    for key in ('JUPYTER_RUNTIME_DIR','IPYTHONDIR','MPLCONFIGDIR'):
        os.environ[key]=str(runtime/key.lower())
    notebook=nbformat.read(ROOT/'Root_Zone_Water_Estimation_with_Satellite_Data.ipynb',as_version=4)
    blocker=nbformat.v4.new_code_cell('''import socket
_original_connect = socket.socket.connect
def _offline_connect(self, address):
    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise RuntimeError('Cached notebook attempted outbound network access')
    return _original_connect(self, address)
socket.socket.connect = _offline_connect
''')
    notebook.cells.insert(0,blocker)
    NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    notebook.cells.pop(0)
    for cell in notebook.cells:
        if cell.cell_type=='code' and cell.execution_count is not None: cell.execution_count-=1
    nbformat.write(notebook,ROOT/'Root_Zone_Water_Estimation_with_Satellite_Data.ipynb')
    print('Fresh-kernel cached Run All completed; outbound network blocked.')
